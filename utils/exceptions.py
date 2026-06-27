"""Exception hierarchy.

Each required failure mode maps to a specific exception so callers (and the
CSV writer) can distinguish them cleanly instead of catching bare Exception.
"""

from __future__ import annotations


class ReceiptPipelineError(Exception):
    """Base class for every error this pipeline raises on purpose."""


class EmptyImageError(ReceiptPipelineError):
    """The image decoded but has zero area / is blank."""


class UnsupportedFileError(ReceiptPipelineError):
    """File extension or container is not a supported image type."""


class CorruptedImageError(ReceiptPipelineError):
    """The file exists but cannot be decoded as an image."""


class ReceiptNotFoundError(ReceiptPipelineError):
    """Crop/deskew could not locate a receipt-shaped region."""


class MultipleReceiptsError(ReceiptPipelineError):
    """More than one receipt-sized region was detected.

    The spec assumes one receipt per photo; we surface this rather than
    silently picking one.
    """


class OCRFailureError(ReceiptPipelineError):
    """OCR ran but produced no usable text."""
