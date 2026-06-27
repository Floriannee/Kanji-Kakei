"""CSV output.

Columns required by the spec: product name, price, tax, store — plus
confidence scores. We emit one row per line item; the store and tax columns
repeat the receipt-level values on each row so the CSV is self-contained.

Confidence columns:
  name_confidence     OCR confidence for the product-name line
  category            assigned category
  category_confidence cosine similarity for the category
"""

from __future__ import annotations

import csv
from pathlib import Path

from utils.logging_setup import get_logger

logger = get_logger(__name__)

FIELDNAMES = [
    "product_name",
    "price",
    "tax",
    "store",
    "name_confidence",
    "category",
    "category_confidence",
]


def write_csv(parsed, out_path: str | Path) -> Path:
    """Write a ParsedReceipt to CSV. Returns the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so Excel on Windows shows Japanese correctly.
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        if not parsed.items:
            logger.warning("No items to write for %s", out_path)
        for it in parsed.items:
            writer.writerow(
                {
                    "product_name": it.name,
                    "price": "" if it.price is None else f"{it.price:.0f}",
                    "tax": "" if parsed.tax is None else f"{parsed.tax:.0f}",
                    "store": parsed.store or "",
                    "name_confidence": f"{it.name_confidence:.3f}",
                    "category": it.category,
                    "category_confidence": f"{it.category_confidence:.3f}",
                }
            )
    logger.info("Wrote CSV %s (%d rows)", out_path, len(parsed.items))
    return out_path
