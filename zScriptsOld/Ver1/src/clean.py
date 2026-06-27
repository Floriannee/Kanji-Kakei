"""
clean.py

Stage 4 of the receipt-reading pipeline: normalize and correct the raw
OCR text before it's parsed and categorized.

Plan (not yet implemented -- design already worked out in conversation):
    Rule-based layer (no model):
        unicodedata.normalize("NFKC", text)
        -> neologdn.normalize(text)
        -> jaconv / mojimoji for remaining width/script cleanup
        -> optional SudachiPy normalized_form() for canonical item names

    Contextual layer (pretrained, zero-shot, no fine-tuning):
        Pass the rule-cleaned text through a pretrained Japanese masked-
        language model (e.g. cl-tohoku/bert-base-japanese-v3) in fill-mask
        mode to catch likely character-substitution OCR errors that the
        rules alone don't fix.

    Also worth building over time: a small custom dictionary of receipt-
    specific OCR confusion pairs (e.g. ロ/口/0, ソ/ン, シ/ツ), built
    empirically from real OCR output on your own receipts.
"""


def normalize(text: str) -> str:
    """
    Apply rule-based normalization to a single line of raw OCR text.
    """
    raise NotImplementedError(
        "Rule-based normalization not yet wired in. Planned: unicodedata "
        "NFKC -> neologdn -> jaconv/mojimoji, see module docstring."
    )


def correct(text: str) -> str:
    """
    Apply contextual (language-model-based) correction to already
    rule-normalized text.
    """
    raise NotImplementedError(
        "LM-based correction not yet wired in. Planned: pretrained Japanese "
        "BERT in fill-mask mode, see module docstring."
    )
