"""CLI: read receipts from photos and write CSV.

Usage:
    python -m inference.run path/to/photo.jpg
    python -m inference.run path/to/folder --batch
    python -m inference.run photo.jpg --csv out.csv --no-db --save-crop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.image_io import SUPPORTED_EXTENSIONS
from utils.logging_setup import setup_logging, get_logger

logger = get_logger(__name__)


def _gather(path: Path, batch: bool) -> list[Path]:
    if path.is_dir() or batch:
        return sorted(
            p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return [path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Japanese receipt OCR -> CSV")
    parser.add_argument("input", help="Image file or folder")
    parser.add_argument("--batch", action="store_true", help="Treat input as a folder")
    parser.add_argument("--csv", help="Output CSV path (single-image mode)")
    parser.add_argument("--no-db", action="store_true", help="Skip SQLite storage")
    parser.add_argument("--save-crop", action="store_true", help="Save the deskewed crop")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    # Import after logging is configured so engine init is logged.
    from inference.pipeline import process_image, process_batch

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input does not exist: %s", input_path)
        return 2

    paths = _gather(input_path, args.batch)
    if not paths:
        logger.error("No supported images found at %s", input_path)
        return 2

    if len(paths) == 1 and not args.batch:
        result = process_image(
            paths[0],
            save_crop=args.save_crop,
            store_in_db=not args.no_db,
            csv_path=args.csv,
        )
        if result.success:
            logger.info("OK -> %s", result.csv_path)
            return 0
        logger.error("FAILED (%s): %s", result.error_type, result.error_message)
        return 1

    results = process_batch(paths, store_in_db=not args.no_db)
    failures = [r for r in results if not r.success]
    for r in failures:
        logger.error("FAILED %s (%s): %s", r.source, r.error_type, r.error_message)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
