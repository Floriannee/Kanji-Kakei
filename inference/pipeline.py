import os
import io
import json
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
  Food, Drink, Snack, Household, Personal Care, Stationery, Other
- note: explain WHAT it is, WHY it's popular, any student tips
(e.g. "ナナチキ  7-Eleven's iconic fried chicken sold hot at the register,
crispy outside juicy inside, ~250, staple for students on a budget")

CRITICAL: 
1. Only extract actual physical products purchased in the "items" list.
2. Do NOT extract tax breakdowns, tax totals, subtotals, change, payment details, or point balances as individual items in the "items" list. For example, lines like "8%対象", "10%対象", "消費税", "内消費税", "非課税" must NEVER be listed as items in the "items" list.
3. Extract the total tax amount (sum of all taxes, or the value next to "消費税", "内消費税", or "税") and put it in the root-level "tax_amount" field.
4. Ensure each physical product is only listed ONCE in the "items" list. Do NOT list the same product more than once unless multiple separate units were actually purchased. If the receipt repeats product names or prices in tax calculation sections, do NOT duplicate them.
5. The "category" field must be exactly one of: Food, Drink, Snack, Household, Personal Care, Stationery, Other.

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

class ReceiptParser:
    def __init__(self):
        """Initialize receipt parser loading environment credentials."""
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        
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

    def parse_receipt_image(self, pil_image: Image.Image) -> dict:
        """
        Multimodal inference wrapper with a 2-tier fallback architecture:
        - TIER 1: Groq API with Llama-4-Scout
        - TIER 2: Local Simulation Mode (FamilyMart Mock response fallback)
        """
        # Convert image to base64 data URL
        base64_url = self.pil_to_base64_data_url(pil_image)

        # ----------------- TIER 1: GROQ API -----------------
        if self.groq_api_key:
            try:
                logger.info("Invoking TIER 1: Groq API (meta-llama/llama-4-scout-17b-16e-instruct)...")
                response = self._call_groq_vision(base64_url)
                logger.info("[Pipeline]  Success via Groq")
                return response
            except Exception as e:
                logger.error(f"Tier 1 (Groq) execution failed: {e}. Falling back to Simulation Mode...")
        else:
            logger.warning("Groq API key not found. Skipping Tier 1...")

        # ----------------- TIER 2: SIMULATION MODE -----------------
        logger.warning("All API providers failed or are unconfigured. Entering TIER 2 (Simulation Mode)...")
        response = self._get_simulated_response()
        logger.info("[Pipeline]  Success via Simulation")
        return response

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
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        else:
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
