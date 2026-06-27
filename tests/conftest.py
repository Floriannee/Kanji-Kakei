"""Shared test fixtures.

We synthesize images with OpenCV so the deterministic parts of the pipeline
(loading, detection geometry, parsing, metrics, CSV, DB) can be tested
without downloading any model or dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make the project importable when tests run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_image_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def receipt_on_table(tmp_path) -> Path:
    """A dark background with a bright, slightly rotated white receipt."""
    img = np.full((800, 600, 3), 40, dtype=np.uint8)  # dark table
    # White receipt rectangle, rotated a few degrees.
    box = np.array([[180, 150], [430, 170], [410, 640], [160, 620]], dtype=np.int32)
    cv2.fillPoly(img, [box], (245, 245, 245))
    # Add some dark "text" lines so it isn't blank.
    for y in range(200, 600, 40):
        cv2.line(img, (200, y), (390, y), (30, 30, 30), 3)
    path = tmp_path / "receipt.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def blank_image(tmp_path) -> Path:
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def two_receipts(tmp_path) -> Path:
    img = np.full((800, 900, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (60, 150), (360, 640), (245, 245, 245), -1)
    cv2.rectangle(img, (520, 150), (820, 640), (245, 245, 245), -1)
    for x0 in (60, 520):
        for y in range(200, 600, 40):
            cv2.line(img, (x0 + 20, y), (x0 + 260, y), (30, 30, 30), 3)
    path = tmp_path / "two.png"
    cv2.imwrite(str(path), img)
    return path
