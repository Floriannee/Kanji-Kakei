"""
ocr.py

Stage 3 of the receipt-reading pipeline: read text from the cropped,
deskewed receipt image produced by preprocess.py.

Plan (not yet implemented):
    - Use PaddleOCR or EasyOCR (pretrained, supports Japanese out of the box)
      to detect text line bounding boxes and recognize the characters in each.
    - Optional: fine-tune the recognition module on the CORD dataset if
      accuracy on real thermal-paper receipts isn't good enough out of the box.
      That fine-tuning run happens separately, on a free cloud GPU notebook
      (Colab/Kaggle) -- see training/README.md.

Expected output shape: a list of dicts, one per detected text line, e.g.
    [{"text": "牛乳 1L", "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "confidence": 0.94}, ...]
roughly in top-to-bottom reading order.
"""

import numpy as np


def read_text(image: np.ndarray) -> list[dict]:
    """
    Run OCR on a cropped/deskewed receipt image.

    Args:
        image: a cropped, deskewed receipt image (BGR, as returned by
            preprocess.crop_and_deskew).

    Returns:
        A list of {"text": str, "box": list, "confidence": float} dicts,
        one per detected line, in roughly top-to-bottom order.
    """
    raise NotImplementedError(
        "OCR stage not yet implemented. Planned: PaddleOCR or EasyOCR, "
        "see module docstring."
    )
