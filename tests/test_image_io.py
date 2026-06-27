"""Tests for utils.image_io and the required input failure modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.exceptions import (
    CorruptedImageError,
    EmptyImageError,
    UnsupportedFileError,
)
from utils.image_io import load_image, save_image


def test_load_valid_image(receipt_on_table):
    img = load_image(receipt_on_table)
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_unsupported_extension(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(UnsupportedFileError):
        load_image(p)


def test_zero_byte_file(tmp_path):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(EmptyImageError):
        load_image(p)


def test_corrupted_image(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nthis is not a real png")
    with pytest.raises(CorruptedImageError):
        load_image(p)


def test_blank_image_rejected(blank_image):
    with pytest.raises(EmptyImageError):
        load_image(blank_image)


def test_missing_file(tmp_path):
    with pytest.raises(CorruptedImageError):
        load_image(tmp_path / "does_not_exist.png")


def test_save_roundtrip(tmp_path):
    img = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
    out = tmp_path / "sub" / "out.png"
    save_image(img, out)
    assert out.exists()
    reloaded = load_image(out)
    assert reloaded.shape == img.shape
