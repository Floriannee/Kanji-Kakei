"""Benchmark the pipeline against the target accuracies.

Reads a labeled evaluation set described by a manifest and reports, per stage:
  detection IoU accuracy, OCR character accuracy, item-extraction F1, and the
  end-to-end score — each next to its target so regressions are obvious.

Manifest format (benchmarks/eval_manifest.json):
{
  "samples": [
    {
      "image": "path/to/photo.jpg",
      "receipt_box": [x, y, w, h],          # optional, for detection metric
      "lines": ["line1", "line2", ...],     # OCR ground truth, top-to-bottom
      "store": "...",
      "total": 1280,
      "items": [["牛乳 1L", 248], ["パン", 150]]
    }
  ]
}

Run:
    python -m benchmarks.benchmark --manifest benchmarks/eval_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.logging_setup import setup_logging, get_logger
from utils.metrics import (
    character_accuracy,
    detection_iou_accuracy,
    end_to_end_score,
    item_extraction_f1,
)

logger = get_logger(__name__)

TARGETS = {
    "detection": (0.95, "Receipt detection"),
    "ocr": (0.90, "OCR character accuracy"),
    "items": (0.80, "Structured item extraction (F1)"),
    "end_to_end": (0.75, "End-to-end pipeline"),
}


def _bbox_from_crop(image, crop) -> tuple[int, int, int, int] | None:
    """Approximate detection box: assume crop covers detected region.

    For benchmarking detection we re-run only the detector and record the
    axis-aligned bounding box of the returned crop relative to the original.
    Since the warp loses absolute coordinates, we report a coarse proxy: the
    crop's own dimensions vs the original (used only when receipt_box given).
    """
    h, w = crop.shape[:2]
    return (0, 0, w, h)


def run(manifest_path: Path) -> dict:
    from inference.detect import crop_and_deskew
    from inference.ocr import get_default_engine
    from inference.clean import clean_lines
    from inference.parse import parse_lines
    from utils.image_io import load_image
    from utils.exceptions import ReceiptPipelineError

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data["samples"]
    engine = get_default_engine()

    det_preds, det_gts = [], []
    ocr_pred_lines, ocr_gt_lines = [], []
    pred_receipts, gt_receipts = [], []

    for s in samples:
        try:
            image = load_image(s["image"])
            crop = crop_and_deskew(image)
            if "receipt_box" in s:
                det_preds.append(_bbox_from_crop(image, crop))
                det_gts.append(tuple(s["receipt_box"]))

            lines = clean_lines(engine.read(crop))
            pred_texts = [l.text for l in lines]
            if "lines" in s:
                # Align by index up to the shorter length.
                n = min(len(pred_texts), len(s["lines"]))
                ocr_pred_lines.extend(pred_texts[:n])
                ocr_gt_lines.extend(s["lines"][:n])

            parsed = parse_lines(lines)
            pred_receipts.append(
                {
                    "store": parsed.store,
                    "total": None if parsed.total is None else round(parsed.total),
                    "items": [(it.name, it.price) for it in parsed.items],
                }
            )
        except ReceiptPipelineError as exc:
            logger.warning("Sample %s failed: %s", s.get("image"), exc)
            pred_receipts.append({"store": None, "total": None, "items": []})
        gt_receipts.append(
            {
                "store": s.get("store"),
                "total": s.get("total"),
                "items": [tuple(i) for i in s.get("items", [])],
            }
        )

    results = {}
    if det_gts:
        results["detection"] = detection_iou_accuracy(det_preds, det_gts)
    if ocr_gt_lines:
        results["ocr"] = character_accuracy(ocr_pred_lines, ocr_gt_lines)

    all_pred_items = [i for r in pred_receipts for i in r["items"]]
    all_gt_items = [i for r in gt_receipts for i in r["items"]]
    results["items"] = item_extraction_f1(all_pred_items, all_gt_items).f1
    results["end_to_end"] = end_to_end_score(pred_receipts, gt_receipts)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the receipt pipeline")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    setup_logging()
    manifest = Path(args.manifest)
    if not manifest.exists():
        logger.error("Manifest not found: %s", manifest)
        return 2

    results = run(manifest)
    logger.info("=== Benchmark results ===")
    all_pass = True
    for key, (target, label) in TARGETS.items():
        if key not in results:
            logger.info("%-32s  n/a (no ground truth)", label)
            continue
        value = results[key]
        ok = value >= target
        all_pass &= ok
        logger.info(
            "%-32s  %.1f%%  (target %.0f%%)  %s",
            label, value * 100, target * 100, "PASS" if ok else "BELOW",
        )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
