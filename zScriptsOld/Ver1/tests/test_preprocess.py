"""
test_preprocess.py

Basic sanity test for stage 2 (crop & deskew), using the synthetic
sample image in data/sample/.

Run with:
    pytest tests/test_preprocess.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import crop_and_deskew  # noqa: E402

SAMPLE_IMAGE = Path(__file__).parent.parent / "data" / "sample" / "sample_receipt.jpg"


def test_crop_and_deskew_runs_and_shrinks_the_image():
    original_path = SAMPLE_IMAGE
    result = crop_and_deskew(str(original_path))

    # Should return a real image
    assert result is not None
    assert result.shape[0] > 0 and result.shape[1] > 0

    # The cropped receipt should be smaller than the full photo, since the
    # full photo includes background around the receipt
    import cv2
    original = cv2.imread(str(original_path))
    assert result.shape[0] * result.shape[1] < original.shape[0] * original.shape[1]


def test_crop_and_deskew_produces_a_roughly_upright_rectangle():
    result = crop_and_deskew(str(SAMPLE_IMAGE))
    height, width = result.shape[:2]

    # The synthetic sample receipt is taller than it is wide
    assert height > width
