"""Tests for text normalization and config loading."""

from __future__ import annotations

from config import Config, settings
from inference.clean import normalize_text


def test_nfkc_fullwidth_digits():
    # Full-width digits and ASCII should fold to half-width.
    assert normalize_text("１２３ＡＢＣ") == "123ABC"


def test_normalize_strips_whitespace():
    assert normalize_text("  牛乳  ") == "牛乳"


def test_normalize_empty():
    assert normalize_text("") == ""


def test_config_defaults():
    cfg = Config()
    assert cfg.training.seed == 42
    assert cfg.ocr.engine in ("easyocr", "paddleocr")
    assert 0 < cfg.parse.category_threshold < 1


def test_settings_singleton_loaded():
    # The module-level settings should have parsed config.yaml.
    assert settings.detection.min_area_ratio > 0
    assert settings.database_path.endswith(".db")
