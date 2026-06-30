import os
import csv
import logging
from datetime import datetime

from config.settings import CSV_FILE

# Initialize logger
logger = logging.getLogger("KanjiKakei.CSV")

# Column order for the continually-updated item ledger. One row per purchased item.
FIELDNAMES = [
    "date",
    "store_name",
    "japanese_name",
    "english_name",
    "category",
    "price",
    "note",
    "receipt_total",
    "tax_amount",
    "savings_advice",
    "image_path",
]


def init_csv():
    """Create the CSV ledger with a header row if it does not already exist."""
    parent_dir = os.path.dirname(CSV_FILE)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        logger.info(f"Created new CSV ledger at path: {CSV_FILE}")


def append_items_to_csv(receipt_data: dict, image_path: str = "") -> int:
    """
    Append one row per line item from a confirmed receipt to the CSV ledger.
    This is the "continually updated" store that the main window list is rendered from.

    Returns the number of item rows written.
    """
    init_csv()

    store_name = receipt_data.get("store_name", "Unknown Store")
    total_amount = receipt_data.get("total_amount", 0)
    tax_amount = receipt_data.get("tax_amount", 0)
    savings_advice = receipt_data.get("savings_advice", "")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    items = receipt_data.get("items", [])

    rows_written = 0
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for item in items:
            writer.writerow({
                "date": date_str,
                "store_name": store_name,
                "japanese_name": item.get("japanese_name", ""),
                "english_name": item.get("english_name", ""),
                "category": item.get("category") or "Other",
                "price": item.get("price", 0),
                "note": item.get("note", ""),
                "receipt_total": total_amount,
                "tax_amount": tax_amount,
                "savings_advice": savings_advice,
                "image_path": image_path,
            })
            rows_written += 1

    logger.info(f"Appended {rows_written} item row(s) to CSV ledger for store: {store_name}")
    return rows_written


def load_all_items() -> list:
    """Load every stored item row from the CSV ledger as a list of dicts."""
    init_csv()
    items = []
    with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items
