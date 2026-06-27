"""Tests for CSV output and the SQLite storage layer."""

from __future__ import annotations

import csv
from pathlib import Path

from database import ReceiptDB
from inference.csv_writer import FIELDNAMES, write_csv
from inference.parse import Item, ParsedReceipt


def _sample_receipt() -> ParsedReceipt:
    return ParsedReceipt(
        store="セブンイレブン",
        store_confidence=0.95,
        tax=32.0,
        total=430.0,
        items=[
            Item("牛乳 1L", 248.0, "Dairy & Eggs", 0.91, 0.82),
            Item("食パン", 150.0, "Bakery", 0.88, 0.79),
        ],
    )


def test_csv_columns_and_rows(tmp_path):
    out = tmp_path / "r.csv"
    write_csv(_sample_receipt(), out)
    assert out.exists()
    with out.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDNAMES
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["product_name"] == "牛乳 1L"
    assert rows[0]["price"] == "248"
    assert rows[0]["tax"] == "32"
    assert rows[0]["store"] == "セブンイレブン"
    assert float(rows[0]["name_confidence"]) > 0
    assert rows[0]["category"] == "Dairy & Eggs"


def test_csv_empty_items(tmp_path):
    empty = ParsedReceipt(store="X", store_confidence=0.5, tax=None, total=None, items=[])
    out = tmp_path / "empty.csv"
    write_csv(empty, out)
    with out.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_db_insert_and_query(tmp_path):
    db = ReceiptDB(tmp_path / "test.db")
    rid = db.insert_receipt(_sample_receipt(), source_image="photo.jpg")
    assert isinstance(rid, int) and rid > 0
    by_cat = dict(db.spending_by_category())
    assert by_cat.get("Dairy & Eggs") == 248.0
    assert by_cat.get("Bakery") == 150.0
