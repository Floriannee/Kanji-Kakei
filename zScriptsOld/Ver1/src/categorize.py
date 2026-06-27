"""
categorize.py

Stage 5 of the receipt-reading pipeline: separate item lines from
header/footer noise, then assign each item a product category.

Plan (not yet implemented):
    Parsing (rule-based, no model):
        Regex over cleaned OCR lines to find price-like patterns
        (e.g. r"\\d{2,5}" near a yen sign or at line-end) and split each
        matching line into an (item_name, price) pair. Lines with no
        price-like pattern are treated as header/footer noise (store name,
        date, subtotal, tax, total, payment method) and excluded from the
        item list.

    Categorizing (pretrained, zero-shot, no fine-tuning):
        Embed each item name with a pretrained multilingual sentence
        embedding model (e.g. multilingual-e5), embed the category list
        from data/categories.json, and assign the category with the
        highest cosine similarity. Below a confidence threshold, assign
        "other" instead of forcing a bad guess.
"""

import json
from pathlib import Path

CATEGORIES_PATH = Path(__file__).parent.parent / "data" / "categories.json"


def load_categories() -> list[dict]:
    """Load the category taxonomy from data/categories.json."""
    with open(CATEGORIES_PATH, encoding="utf-8") as f:
        return json.load(f)["categories"]


def parse_items(lines: list[dict]) -> list[dict]:
    """
    Separate item lines (name + price) from header/footer noise.

    Args:
        lines: cleaned OCR output, e.g. [{"text": "牛乳 1L 248", ...}, ...]

    Returns:
        A list of {"item": str, "price": int} dicts.
    """
    raise NotImplementedError(
        "Item parsing not yet implemented. Planned: regex price-pattern "
        "split, see module docstring."
    )


def categorize_items(items: list[dict], confidence_threshold: float = 0.5) -> list[dict]:
    """
    Assign a product category to each parsed item.

    Args:
        items: output of parse_items().
        confidence_threshold: minimum cosine similarity to accept a match;
            below this, the item is assigned the "other" category.

    Returns:
        items, each with added "category" and "confidence" keys.
    """
    raise NotImplementedError(
        "Categorization not yet implemented. Planned: multilingual-e5 "
        "embedding similarity against data/categories.json, see module "
        "docstring."
    )
