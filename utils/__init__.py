"""Shared utilities: logging, seeding, image I/O, exceptions."""

from .exceptions import (
    ReceiptPipelineError,
    EmptyImageError,
    UnsupportedFileError,
    CorruptedImageError,
    ReceiptNotFoundError,
    MultipleReceiptsError,
    OCRFailureError,
)
from .logging_setup import setup_logging, get_logger
from .seed import seed_everything
from .image_io import load_image, save_image, SUPPORTED_EXTENSIONS

__all__ = [
    "ReceiptPipelineError",
    "EmptyImageError",
    "UnsupportedFileError",
    "CorruptedImageError",
    "ReceiptNotFoundError",
    "MultipleReceiptsError",
    "OCRFailureError",
    "setup_logging",
    "get_logger",
    "seed_everything",
    "load_image",
    "save_image",
    "SUPPORTED_EXTENSIONS",
]
