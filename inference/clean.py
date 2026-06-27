"""Step 4: clean and normalize OCR text.

Rule-based normalization chain (cheap, deterministic, always on):
  unicodedata NFKC -> neologdn -> jaconv/mojimoji width normalization.

Optional BERT fill-mask correction is gated behind ParseConfig.use_bert_correction
because it pulls a large model and is slow; it is off by default.

All optional dependencies degrade gracefully: if neologdn/jaconv are missing,
the function still returns NFKC-normalized text and logs a warning once.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _optional_imports():
    mods = {}
    try:
        import neologdn

        mods["neologdn"] = neologdn
    except ImportError:
        logger.warning("neologdn not installed; skipping that normalization step.")
    try:
        import jaconv

        mods["jaconv"] = jaconv
    except ImportError:
        logger.warning("jaconv not installed; skipping kana width normalization.")
    return mods


def normalize_text(text: str) -> str:
    """Normalize a single OCR line. Pure function, safe to call per-line."""
    if not text:
        return text
    # NFKC folds full-width ASCII/digits to half-width and composes kana.
    out = unicodedata.normalize("NFKC", text)
    mods = _optional_imports()
    if "neologdn" in mods:
        out = mods["neologdn"].normalize(out)
    if "jaconv" in mods:
        # Half-width katakana -> full-width; keep ASCII half-width.
        out = mods["jaconv"].h2z(out, kana=True, ascii=False, digit=False)
    return out.strip()


@lru_cache(maxsize=1)
def _bert_pipeline():
    from transformers import pipeline

    model = settings.parse.bert_model
    logger.info("Loading BERT fill-mask model %s for correction", model)
    return pipeline("fill-mask", model=model)


def correct_text(text: str) -> str:
    """Optional masked-LM correction. No-op unless enabled in config.

    This is a placeholder hook: a production correction strategy would
    detect low-confidence characters and mask them individually. We keep
    the interface so it can be enabled without touching callers.
    """
    if not settings.parse.use_bert_correction:
        return text
    try:
        _ = _bert_pipeline()  # warm the model; real masking left to extend
    except Exception as exc:  # noqa: BLE001 - correction is best-effort
        logger.warning("BERT correction unavailable: %s", exc)
    return text


def clean_lines(lines: list) -> list:
    """Apply normalize (+ optional correct) to a list of OCRLine objects.

    Mutates each line's ``text`` in place and returns the same list.
    """
    for line in lines:
        line.text = correct_text(normalize_text(line.text))
    return lines
