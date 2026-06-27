"""Robust image loading.

``load_image`` is the single gate every photo passes through. It enforces the
required failure modes before any CV/OCR work happens:

  unsupported file  -> UnsupportedFileError
  empty image       -> EmptyImageError
  corrupted image   -> CorruptedImageError

Decoding is done with cv2.imdecode on raw bytes (not cv2.imread) because
imread silently returns None on failure and also mishandles non-ASCII
Windows paths. Reading bytes ourselves avoids both problems.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.exceptions import (
    CorruptedImageError,
    EmptyImageError,
    UnsupportedFileError,
)
from utils.logging_setup import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    """Load and validate an image from disk.

    Returns a BGR uint8 array. Raises a specific ReceiptPipelineError
    subclass for each failure mode.
    """
    path = Path(path)
    logger.debug("Loading image %s", path)

    if not path.exists():
        raise CorruptedImageError(f"File does not exist: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported extension '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise CorruptedImageError(f"Could not read bytes from {path}: {exc}") from exc

    if raw.size == 0:
        raise EmptyImageError(f"File is zero bytes: {path}")

    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise CorruptedImageError(f"Could not decode image (corrupted?): {path}")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise EmptyImageError(f"Decoded image has zero area: {path}")

    # A fully uniform frame (e.g. a blank scan) carries no receipt.
    if float(image.std()) < 1.0:
        raise EmptyImageError(f"Image appears blank (near-zero variance): {path}")

    logger.debug("Loaded image %s shape=%s", path, image.shape)
    return image


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Write an image, handling non-ASCII paths on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise CorruptedImageError(f"Failed to encode image for {path}")
    buf.tofile(str(path))
    logger.debug("Saved image to %s", path)
