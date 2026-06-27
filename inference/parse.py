"""Step 5: parse structure, then categorize.

Parsing (rule-based): split OCR lines into header (store name), footer
(subtotal/tax/total), and item lines (name + price). Price detection uses
regex for yen patterns. No model.

Categorizing (embedding-based): embed each item name and each category
prototype with a multilingual sentence-embedding model, assign the
highest-cosine category, or "Other/Uncategorized" below threshold.

If sentence-transformers is unavailable, categorization degrades to
"Other/Uncategorized" with a logged warning so the rest of the pipeline
still runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

# Price like "¥1,280", "1280円", or a trailing integer "  248".
_PRICE_RE = re.compile(r"(?:¥|￥)?\s*([0-9][0-9,]{0,8})\s*(?:円)?\s*[*※]?$")
_TAX_KEYWORDS = ("税", "消費税", "内税", "外税")
_TOTAL_KEYWORDS = ("合計", "小計", "計", "お買上げ", "総額")
_STORE_HINTS = ("店", "店舗", "マート", "ストア", "スーパー")


@dataclass
class Item:
    name: str
    price: float | None
    category: str
    name_confidence: float       # OCR confidence for the source line
    category_confidence: float   # cosine similarity of the best category


@dataclass
class ParsedReceipt:
    store: str | None
    store_confidence: float
    tax: float | None
    total: float | None
    items: list[Item]


def _extract_price(text: str) -> float | None:
    m = _PRICE_RE.search(text.replace(" ", ""))
    if not m:
        return None
    digits = m.group(1).replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    # Reject implausible values (timestamps, phone fragments).
    if value <= 0 or value > 1_000_000:
        return None
    return value


def _looks_like_total(text: str) -> bool:
    return any(k in text for k in _TOTAL_KEYWORDS)


def _looks_like_tax(text: str) -> bool:
    return any(k in text for k in _TAX_KEYWORDS)


def parse_lines(lines: list) -> ParsedReceipt:
    """Turn cleaned OCRLine objects into a structured receipt."""
    store = None
    store_conf = 0.0
    tax = None
    total = None
    items: list[Item] = []

    # Heuristic: the store name is usually among the first 3 lines and
    # contains a store hint or is simply the first non-numeric line.
    for line in lines[:3]:
        if any(h in line.text for h in _STORE_HINTS) or (
            store is None and not _extract_price(line.text) and len(line.text) >= 2
        ):
            store = line.text
            store_conf = line.confidence
            break

    for line in lines:
        text = line.text
        price = _extract_price(text)

        if _looks_like_tax(text) and price is not None:
            tax = price
            continue
        if _looks_like_total(text) and price is not None:
            total = price
            continue
        if price is not None:
            # Item name = the line with the trailing price stripped.
            name = _PRICE_RE.sub("", text).strip(" 　:：-")
            if not name:
                # Price-only line; skip (likely a column artifact).
                continue
            items.append(
                Item(
                    name=name,
                    price=price,
                    category="Other/Uncategorized",
                    name_confidence=line.confidence,
                    category_confidence=0.0,
                )
            )

    logger.info("Parsed %d item lines (store=%r)", len(items), store)
    categorize(items)
    return ParsedReceipt(
        store=store,
        store_confidence=store_conf,
        tax=tax,
        total=total,
        items=items,
    )


# --- categorization ----------------------------------------------------------
@lru_cache(maxsize=1)
def _load_categories() -> dict[str, list[str]]:
    path = Path(settings.parse.categories_file)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("categories", {})


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    model = settings.parse.embedding_model
    logger.info("Loading embedding model %s", model)
    return SentenceTransformer(model)


@lru_cache(maxsize=1)
def _category_prototypes():
    """Return (names, matrix) of L2-normalized category prototype vectors."""
    cats = _load_categories()
    model = _embedder()
    names: list[str] = []
    seeds: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for name, words in cats.items():
        if not words:
            continue  # Other/Uncategorized has no prototype
        names.append(name)
        # e5 models expect a "query:"/"passage:" prefix.
        seeds.extend(f"passage: {w}" for w in words)
        spans.append((cursor, cursor + len(words)))
        cursor += len(words)

    if not seeds:
        return [], np.zeros((0, 1))

    emb = model.encode(seeds, normalize_embeddings=True)
    protos = np.stack([emb[a:b].mean(axis=0) for (a, b) in spans])
    # Re-normalize the averaged prototypes.
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9
    return names, protos


def categorize(items: list[Item]) -> None:
    """Assign a category + confidence to each item, in place."""
    if not items:
        return
    try:
        names, protos = _category_prototypes()
        model = _embedder()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Categorization disabled (%s); items left uncategorized.", exc)
        return
    if len(names) == 0:
        return

    queries = [f"query: {it.name}" for it in items]
    vecs = model.encode(queries, normalize_embeddings=True)
    sims = vecs @ protos.T  # cosine, both normalized
    threshold = settings.parse.category_threshold
    for it, row in zip(items, sims):
        best = int(row.argmax())
        score = float(row[best])
        if score >= threshold:
            it.category = names[best]
            it.category_confidence = score
        else:
            it.category = "Other/Uncategorized"
            it.category_confidence = score
