"""Tests for inference.detect."""

from __future__ import annotations

import numpy as np
import pytest

from inference.detect import _order_points, crop_and_deskew
from utils.exceptions import MultipleReceiptsError, ReceiptNotFoundError
from utils.image_io import load_image


def test_order_points():
    pts = np.array([[10, 10], [100, 12], [98, 90], [8, 88]], dtype="float32")
    ordered = _order_points(pts)
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0]          # top-left left of top-right
    assert tl[1] < bl[1]          # top-left above bottom-left
    assert br[0] > bl[0]          # bottom-right right of bottom-left


def test_crop_returns_rectangle(receipt_on_table):
    img = load_image(receipt_on_table)
    crop = crop_and_deskew(img)
    assert crop.ndim == 3
    # The crop should be taller than wide (receipt shape) and reasonably sized.
    assert crop.shape[0] > 50 and crop.shape[1] > 30


def test_no_receipt_raises():
    # Uniform noise with no large rectangular contour.
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    with pytest.raises(ReceiptNotFoundError):
        crop_and_deskew(img)


def test_multiple_receipts_raises(two_receipts):
    img = load_image(two_receipts)
    with pytest.raises(MultipleReceiptsError):
        crop_and_deskew(img)
