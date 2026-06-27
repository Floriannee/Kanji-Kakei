"""
storage.py

Stage 6 of the receipt-reading pipeline: persist parsed, categorized
receipts to a local SQLite database.

Plan (not yet implemented):
    Two tables:
        receipts(id, store_name, date, total, source_image_path)
        line_items(id, receipt_id, item_name, price, category, confidence)

    One file, no server -- good fit for a personal project, and easy to
    query later (e.g. spending by category, spending over time).
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "output" / "receipts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    store_name TEXT,
    date TEXT,
    total INTEGER,
    source_image_path TEXT
);

CREATE TABLE IF NOT EXISTS line_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,
    price INTEGER,
    category TEXT,
    confidence REAL,
    FOREIGN KEY (receipt_id) REFERENCES receipts (id)
);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database file and tables if they don't already exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


def save_receipt(receipt: dict, items: list[dict], db_path: Path = DB_PATH) -> int:
    """
    Insert a receipt and its line items into the database.

    Args:
        receipt: {"store_name": str, "date": str, "total": int, "source_image_path": str}
        items: output of categorize.categorize_items()
        db_path: path to the SQLite database file.

    Returns:
        The new receipt's id.
    """
    raise NotImplementedError(
        "Storage not yet implemented. Schema is ready (see SCHEMA above); "
        "just needs the insert logic wired in."
    )
