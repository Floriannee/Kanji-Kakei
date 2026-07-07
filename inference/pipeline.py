import os
import io
import json
import time
import base64
import logging
import requests
from PIL import Image

# Initialize logger
logger = logging.getLogger("KanjiKakei.Inference")

# The exact system instruction/prompt to pass to LLM providers
PROMPT = """You are a Japanese convenience store & supermarket culture expert helping
foreign students in Japan understand their receipts.
For each item provide:
- Full proper name (resolve ALL katakana abbreviations e.g. ｱｰﾓﾝﾄﾞﾁｮｺ  明治 アーモンドチョコレート)
- English translation
- category: classify the item into EXACTLY ONE of these categories:
  Groceries, Drink, Snack, Dining Out, Daily Essentials, Clothes, Personal Care, Stationery, Leisure, Souvenirs, Tax, Other.
- note: explain WHAT it is, WHY it's popular, any student tips
(e.g. "ナナチキ  7-Eleven's iconic fried chicken sold hot at the register,
crispy outside juicy inside, ~250, staple for students on a budget")

CRITICAL: 
1. Only extract actual physical products purchased in the "items" list.
2. Do NOT extract tax breakdowns, tax totals, subtotals, change, payment details, or point balances as individual items in the "items" list. For example, lines like "8%対象", "10%対象", "消費税", "内消費税", "非課税" must NEVER be listed as items in the "items" list.
3. Extract the total tax amount (sum of all taxes, or the value next to "消費税", "内消費税", or "税") and put it in the root-level "tax_amount" field.
4. Ensure each physical product is only listed ONCE in the "items" list. Do NOT list the same product more than once unless multiple separate units were actually purchased. If the receipt repeats product names or prices in tax calculation sections, do NOT duplicate them.
5. The "category" field must be exactly one of: Groceries, Drink, Snack, Dining Out, Daily Essentials, Clothes, Personal Care, Stationery, Leisure, Souvenirs, Tax, Other.

Return ONLY raw JSON, no markdown, no explanation:
{
"store_name": "Japanese + English name",
"total_amount": 1234,
"tax_amount": 91,
"items": [
{
"japanese_name": "商品名",
"english_name": "English translation",
"category": "Food",
"price": 198,
"note": "Cultural context and student tips"
}
],
"savings_advice": "One actionable money-saving tip for foreign students in Japan"
}
"""


class GroqRateLimitError(RuntimeError):
    """Raised when Groq returns HTTP 429 (rate limited)."""
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class ReceiptParser:
    def __init__(self):
        """Initialize receipt parser loading environment credentials."""
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")

        # Groq's own server-side processing time (seconds) for the most recent
        # successful call, read from the API response's usage.total_time. This
        # excludes our network round-trip and any client-side 429 backoff, so
        # the evaluator can measure "time when Groq is ready and working"
        # rather than wall-clock. None if the last result came from Simulation.
        self.last_groq_total_time = None

        # Log active credentials
        if self.groq_api_key:
            logger.info("Groq API key detected. Provider 1 (Primary) is active.")
        else:
            logger.warning("No GROQ_API_KEY environment variable detected.")

    def pil_to_base64_data_url(self, pil_image: Image.Image) -> str:
        """Convert a PIL image to a base64 encoded string data URL."""
        # Resize large images to reduce payload size and avoid upload timeouts
        max_dim = 1280
        w, h = pil_image.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            pil_image = pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    def parse_receipt_image(
        self,
        pil_image: Image.Image,
        allow_simulation: bool = True,
        max_retries: int = 5,
    ) -> dict:
        """
        Multimodal inference wrapper with a 2-tier fallback architecture:
        - TIER 1: Groq API with Llama-4-Scout (with rate-limit retry/backoff)
        - TIER 2: Local Simulation Mode (FamilyMart Mock response fallback)

        allow_simulation:
            True  -> on failure, fall back to the mock FamilyMart response (app/GUI behaviour).
            False -> on failure, raise instead of returning fake data. Use this when
                     building a dataset/ground truth, so a silent simulated receipt
                     never pollutes your labels.
        max_retries:
            How many times to retry Groq on HTTP 429 (rate limit) before giving up.
        """
        # Reset per-call timing so a failed/simulated result never reports a
        # stale value left over from a previous image.
        self.last_groq_total_time = None

        # Convert image to base64 data URL
        base64_url = self.pil_to_base64_data_url(pil_image)

        # ----------------- TIER 1: GROQ API -----------------
        if self.groq_api_key:
            try:
                return self._call_groq_vision_with_retry(base64_url, max_retries=max_retries)
            except Exception as e:
                logger.error(f"Tier 1 (Groq) execution failed: {e}. "
                             f"{'Falling back to Simulation Mode...' if allow_simulation else 'Raising (simulation disabled).'}")
                if not allow_simulation:
                    raise
        else:
            if not allow_simulation:
                raise RuntimeError("GROQ_API_KEY is not set and simulation fallback is disabled.")
            logger.warning("Groq API key not found. Skipping Tier 1...")

        # ----------------- TIER 2: SIMULATION MODE -----------------
        logger.warning("All API providers failed or are unconfigured. Entering TIER 2 (Simulation Mode)...")
        response = self._get_simulated_response()
        logger.info("[Pipeline]  Success via Simulation")
        return response

    def _call_groq_vision_with_retry(self, base64_url: str, max_retries: int = 5) -> dict:
        """
        Call Groq, retrying on HTTP 429 with exponential backoff. Without this,
        a burst of receipts trips Groq's free-tier rate limit and (in the app)
        every subsequent call falls through to the simulated FamilyMart receipt.
        The backoff sleeps here are deliberately NOT counted in last_groq_total_time.
        """
        for attempt in range(max_retries):
            try:
                response = self._call_groq_vision(base64_url)
                logger.info("[Pipeline]  Success via Groq")
                return response
            except GroqRateLimitError as e:
                if attempt >= max_retries - 1:
                    raise
                # Prefer the server-provided retry-after; otherwise exponential backoff.
                wait = e.retry_after if e.retry_after is not None else (2 ** attempt)
                logger.warning(
                    f"Groq rate limited (429). Waiting {wait:.1f}s before retry "
                    f"{attempt + 1}/{max_retries - 1}..."
                )
                time.sleep(wait)
        # Should not reach here.
        raise RuntimeError("Exhausted Groq retries without a result.")

    def _call_groq_vision(self, base64_url: str) -> dict:
        """Execute HTTP POST call to Groq vision endpoints."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_url
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(url, json=payload, headers=headers, timeout=20)

        if response.status_code == 200:
            result = response.json()

            # Capture Groq's own server-side processing time (seconds). This is
            # queue + prompt + completion time as measured on Groq's side, with
            # no client network latency or retry-backoff included. Used by the
            # evaluator for honest timing. Swap to prompt_time + completion_time
            # if you want to exclude queue time as well.
            usage = result.get("usage", {}) or {}
            self.last_groq_total_time = usage.get("total_time")

            content = result["choices"][0]["message"]["content"]
            return json.loads(content)

        # Rate limited: surface a typed error so the retry layer can back off.
        if response.status_code == 429:
            retry_after = None
            header_val = response.headers.get("retry-after")
            if header_val is not None:
                try:
                    retry_after = float(header_val)
                except ValueError:
                    retry_after = None
            raise GroqRateLimitError(
                f"Groq API rate limited (HTTP 429): {response.text}",
                retry_after=retry_after,
            )

        raise RuntimeError(f"Groq API returned HTTP {response.status_code}: {response.text}")

    def _get_simulated_response(self) -> dict:
        """Hardcoded simulated FamilyMart JSON response fallback."""
        logger.info("Loading offline mock response block.")
        return {
            "store_name": "ファミリーマート 渋谷二丁目店 (FamilyMart)",
            "total_amount": 533,
            "tax_amount": 40,
            "items": [
                {
                    "japanese_name": "ファミチキ (骨なし)",
                    "english_name": "FamiChiki (Boneless Fried Chicken)",
                    "category": "Food",
                    "price": 220,
                    "note": "FamilyMart's signature boneless fried chicken, highly popular among students as a quick hot snack."
                },
                {
                    "japanese_name": "お茶 600ml",
                    "english_name": "Green Tea 600ml",
                    "category": "Drink",
                    "price": 160,
                    "note": "Standard unsweetened bottled green tea sold at all Japanese convenience stores."
                },
                {
                    "japanese_name": "カレーパン",
                    "english_name": "Curry Bread",
                    "category": "Food",
                    "price": 150,
                    "note": "Deep-fried dough filled with thick Japanese curry paste, a staple bakery snack."
                },
                {
                    "japanese_name": "レジ袋 M",
                    "english_name": "Shopping Bag M",
                    "category": "Household",
                    "price": 3,
                    "note": "Standard plastic bag fee introduced in Japan in 2020."
                }
            ],
            "savings_advice": "You spent ¥220 on convenience store fried chicken; buying raw chicken at Gyomu Super can save you over 70%!"
        }
