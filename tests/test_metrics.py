"""Tests for utils.metrics."""

from __future__ import annotations

from utils.metrics import (
    character_accuracy,
    detection_iou_accuracy,
    end_to_end_score,
    item_extraction_f1,
)


def test_character_accuracy_perfect():
    assert character_accuracy(["牛乳", "パン"], ["牛乳", "パン"]) == 1.0


def test_character_accuracy_partial():
    # One char wrong out of two -> 0.5 for that line.
    acc = character_accuracy(["牛乳"], ["牛肉"])
    assert 0.4 < acc < 0.6


def test_detection_iou():
    preds = [(0, 0, 100, 100)]
    gts = [(10, 10, 100, 100)]
    assert detection_iou_accuracy(preds, gts, threshold=0.5) == 1.0
    assert detection_iou_accuracy(preds, gts, threshold=0.9) == 0.0


def test_item_f1_exact():
    pred = [("牛乳", 248), ("パン", 150)]
    gt = [("牛乳", 248), ("パン", 150)]
    prf = item_extraction_f1(pred, gt)
    assert prf.f1 == 1.0


def test_item_f1_partial():
    pred = [("牛乳", 248), ("卵", 200)]
    gt = [("牛乳", 248), ("パン", 150)]
    prf = item_extraction_f1(pred, gt)
    assert 0.0 < prf.f1 < 1.0


def test_end_to_end():
    pred = [{"store": "A", "total": 430, "items": [("牛乳", 248), ("パン", 150)]}]
    gt = [{"store": "A", "total": 430, "items": [("牛乳", 248), ("パン", 150)]}]
    assert end_to_end_score(pred, gt) == 1.0
