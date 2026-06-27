"""End-to-end pipeline: photo -> CSV (+ optional DB).

This is the single function the inference CLI and tests call. It wires the
steps together and maps every failure to a clear result rather than crashing
the process. Each step's failure mode is caught and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from database import ReceiptDB
from inference.clean import clean_lines
from inference.csv_writer import write_csv
from inference.detect import crop_and_deskew
from inference.ocr import OCREngine, get_default_engine
from inference.parse import ParsedReceipt, parse_lines
from utils.exceptions import ReceiptPipelineError
from utils.image_io import load_image, save_image
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    success: bool
    source: str
    csv_path: str | None = None
    receipt_id: int | None = None
    parsed: ParsedReceipt | None = None
    error_type: str | None = None
    error_message: str | None = None


def process_image(
    image_path: str | Path,
    *,
    engine: OCREngine | None = None,
    save_crop: bool = False,
    store_in_db: bool = True,
    csv_path: str | Path | None = None,
) -> PipelineResult:
    """Run the full pipeline on one photo.

    Returns a PipelineResult; never raises for the documented failure modes.
    """
    image_path = Path(image_path)
    src = str(image_path)
    logger.info("=== Processing %s ===", src)

    try:
        image = load_image(image_path)                       # empty/unsupported/corrupted
        crop = crop_and_deskew(image)                        # not found / multiple
        if save_crop:
            crop_out = Path(settings.outputs_dir) / "crops" / f"{image_path.stem}_crop.png"
            save_image(crop, crop_out)

        engine = engine or get_default_engine()
        lines = engine.read(crop)                            # OCR failure
        lines = clean_lines(lines)
        parsed = parse_lines(lines)

        out_csv = Path(csv_path) if csv_path else (
            Path(settings.outputs_dir) / f"{image_path.stem}.csv"
        )
        write_csv(parsed, out_csv)

        receipt_id = None
        if store_in_db:
            receipt_id = ReceiptDB().insert_receipt(parsed, source_image=src)

        logger.info("=== Done %s -> %s ===", src, out_csv)
        return PipelineResult(
            success=True,
            source=src,
            csv_path=str(out_csv),
            receipt_id=receipt_id,
            parsed=parsed,
        )

    except ReceiptPipelineError as exc:
        logger.error("%s failed: %s: %s", src, type(exc).__name__, exc)
        return PipelineResult(
            success=False,
            source=src,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - last-resort guard
        logger.exception("Unexpected error processing %s", src)
        return PipelineResult(
            success=False,
            source=src,
            error_type="UnexpectedError",
            error_message=str(exc),
        )


def process_batch(
    paths: list[str | Path], *, store_in_db: bool = True
) -> list[PipelineResult]:
    """Process many photos reusing one OCR engine."""
    engine = get_default_engine()
    results = []
    for p in paths:
        results.append(process_image(p, engine=engine, store_in_db=store_in_db))
    ok = sum(r.success for r in results)
    logger.info("Batch complete: %d/%d succeeded", ok, len(results))
    return results
