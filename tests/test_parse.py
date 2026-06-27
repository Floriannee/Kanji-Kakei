"""Tests for inference.parse using synthetic OCRLine objects.

Categorization needs sentence-transformers; we don't require it here, so we
assert on parsing structure (store/tax/total/items) and let categorization
default to Other/Uncategorized when the model is absent.
"""

from __future__ import annotations

from inference.ocr import OCRLine
from inference.parse import _extract_price, parse_lines


def _line(text, conf=0.9, y=0):
    return OCRLine(text=text, confidence=conf, box=(0, y, 100, 20))


def test_extract_price_variants():
    assert _extract_price("¥1,280") == 1280
    assert _extract_price("1280円") == 1280
    assert _extract_price("牛乳 248") == 248
    assert _extract_price("no price here") is None
    assert _extract_price("0") is None


def test_parse_basic_receipt():
    lines = [
        _line("セブンイレブン", y=0),
        _line("牛乳 1L 248", y=20),
        _line("食パン 150", y=40),
        _line("消費税 32", y=60),
        _line("合計 430", y=80),
    ]
    parsed = parse_lines(lines)
    assert parsed.store is not None
    assert parsed.tax == 32
    assert parsed.total == 430
    names = {it.name for it in parsed.items}
    assert any("牛乳" in n for n in names)
    assert any("パン" in n for n in names)
    # Tax and total lines must not become items.
    assert all("合計" not in it.name for it in parsed.items)


def test_price_only_line_skipped():
    lines = [_line("ローソン", y=0), _line("248", y=20)]
    parsed = parse_lines(lines)
    assert parsed.items == []
