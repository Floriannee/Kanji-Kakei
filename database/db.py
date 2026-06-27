"""Step 6: SQLite storage.

Two tables:
  receipts(id, store, date, tax, total, source_image, created_at)
  line_items(id, receipt_id, name, price, category,
             name_confidence, category_confidence)

One file, no server. The same parsed data also goes to CSV (see
inference/csv_writer.py); the DB is for later querying / spending summaries.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    store         TEXT,
    date          TEXT,
    tax           REAL,
    total         REAL,
    source_image  TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS line_items (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id            INTEGER NOT NULL,
    name                  TEXT NOT NULL,
    price                 REAL,
    category              TEXT,
    name_confidence       REAL,
    category_confidence   REAL,
    FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_items_receipt ON line_items(receipt_id);
CREATE INDEX IF NOT EXISTS idx_items_category ON line_items(category);
"""


class ReceiptDB:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.debug("Database ready at %s", self.path)

    def insert_receipt(self, parsed, source_image: str, date: str | None = None) -> int:
        """Insert a ParsedReceipt and its items. Returns the receipt id."""
        created = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO receipts (store, date, tax, total, source_image, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (parsed.store, date, parsed.tax, parsed.total, source_image, created),
            )
            receipt_id = int(cur.lastrowid)
            conn.executemany(
                "INSERT INTO line_items (receipt_id, name, price, category,"
                " name_confidence, category_confidence) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        receipt_id,
                        it.name,
                        it.price,
                        it.category,
                        it.name_confidence,
                        it.category_confidence,
                    )
                    for it in parsed.items
                ],
            )
        logger.info("Stored receipt id=%d with %d items", receipt_id, len(parsed.items))
        return receipt_id

    def spending_by_category(self) -> list[tuple[str, float]]:
        """Convenience query for later presentation."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT category, SUM(price) AS total FROM line_items"
                " WHERE price IS NOT NULL GROUP BY category ORDER BY total DESC"
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]
