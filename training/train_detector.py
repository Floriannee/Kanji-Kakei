"""Optionally train a receipt detector (the classical-CV fallback).

Most receipt-on-table-or-hand photos are handled by the classical crop/deskew
in inference/detect.py. Train this only if low-contrast backgrounds defeat it.

We fine-tune an Ultralytics YOLO model to box "receipt" regions. You provide
images + YOLO-format labels (or convert SROIE/your own annotations). After
training, set detection.fallback_model_path in config.yaml to the produced
weights and the classical detector will defer to it when it finds nothing.

Run:
    python -m training.train_detector --data datasets/detect/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import settings
from utils.logging_setup import setup_logging, get_logger
from utils.seed import seed_everything

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a receipt detector")
    parser.add_argument(
        "--data",
        required=True,
        help="Ultralytics data.yaml describing train/val images + labels",
    )
    parser.add_argument("--base", default="yolov8n.pt", help="Pretrained base weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--seed", type=int, default=settings.training.seed)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    seed_everything(args.seed)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("pip install ultralytics to train the detector.")
        return 3

    out_dir = Path(settings.models_dir) / "detector"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fine-tuning %s on %s for %d epochs", args.base, args.data, args.epochs)
    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        seed=args.seed,
        deterministic=True,
        project=str(out_dir),
        name="receipt",
        exist_ok=True,
    )

    best = out_dir / "receipt" / "weights" / "best.pt"
    logger.info("Best detector weights at %s", best)
    logger.info("Set detection.fallback_model_path: %s in config.yaml to use it.", best)

    if args.export_onnx and best.exists():
        YOLO(str(best)).export(format="onnx", opset=17)
        logger.info("Exported detector to ONNX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
