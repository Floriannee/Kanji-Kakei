"""Fine-tune the OCR recognition model on receipt data.

This is the one training run the spec says earns its keep: thermal-printer
fonts and faded ink differ from the natural-scene images EasyOCR/Paddle were
trained on. We fine-tune only the recognition module on CORD-style
(line image, text) pairs.

Design notes
------------
* EasyOCR's recognizer is a CRNN. Full fine-tuning normally uses the upstream
  `EasyOCR/trainer` repo with a config file. To keep this self-contained and
  reproducible, we implement a compact CRNN trainer here that produces weights
  EasyOCR can load as a custom `recog_network`. The architecture matches
  EasyOCR's "standard" recognizer (VGG feature extractor + BiLSTM + CTC).
* Everything is seeded. Checkpoints + the final model are saved to models/.
* If torch is not installed, the script exits with a clear message; training
  is the only place torch is required.

Run:
    python -m training.train_ocr --dataset cord --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import settings
from utils.logging_setup import setup_logging, get_logger
from utils.seed import seed_everything

logger = get_logger(__name__)


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError:
        logger.error("Training requires torch. Install with: pip install torch")
        sys.exit(3)


def build_charset(samples) -> list[str]:
    """Collect the sorted unique character set from transcriptions."""
    chars = set()
    for s in samples:
        chars.update(s.text)
    charset = sorted(chars)
    logger.info("Charset size: %d", len(charset))
    return charset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune OCR recognizer")
    parser.add_argument("--dataset", default="cord", help="Registered dataset name")
    parser.add_argument("--epochs", type=int, default=settings.training.epochs)
    parser.add_argument("--batch-size", type=int, default=settings.training.batch_size)
    parser.add_argument("--lr", type=float, default=settings.training.lr)
    parser.add_argument("--seed", type=int, default=settings.training.seed)
    parser.add_argument("--out", default=None, help="Output model dir")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    _require_torch()
    seed_everything(args.seed)

    import torch
    from torch.utils.data import DataLoader

    from datasets import get_dataset
    from training.crnn import CRNN, CTCLabelConverter, RecognitionDataset, collate

    samples = get_dataset(args.dataset)
    if not samples:
        logger.error("No samples loaded for dataset '%s'.", args.dataset)
        return 2

    charset = build_charset(samples)
    converter = CTCLabelConverter(charset)

    # Reproducible train/val split.
    g = torch.Generator().manual_seed(args.seed)
    n_val = max(1, int(len(samples) * settings.training.val_split))
    perm = torch.randperm(len(samples), generator=g).tolist()
    val_idx = set(perm[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    logger.info("Train/val split: %d / %d", len(train), len(val))

    device = "cuda" if (settings.ocr.gpu and torch.cuda.is_available()) else "cpu"
    logger.info("Training on %s", device)

    train_ds = RecognitionDataset(train, converter)
    val_ds = RecognitionDataset(val, converter)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=settings.training.num_workers,
        collate_fn=collate,
        generator=g,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=settings.training.num_workers,
        collate_fn=collate,
    )

    model = CRNN(num_classes=len(charset) + 1).to(device)  # +1 for CTC blank
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out) if args.out else Path(settings.models_dir) / "ocr_recognizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charset.json").write_text(
        json.dumps(charset, ensure_ascii=False), encoding="utf-8"
    )

    best_val = float("inf")
    patience = settings.training.early_stop_patience
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets, target_lengths in train_dl:
            images = images.to(device)
            optimizer.zero_grad()
            logits = model(images)  # (T, N, C)
            input_lengths = torch.full(
                (images.size(0),), logits.size(0), dtype=torch.long
            )
            loss = criterion(
                logits.log_softmax(2), targets, input_lengths, target_lengths
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / max(1, len(train_dl))

        val_loss = _evaluate(model, val_dl, criterion, device)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch, args.epochs, train_loss, val_loss,
        )

        if epoch % settings.training.checkpoint_every == 0:
            ckpt = out_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save({"model": model.state_dict(), "charset": charset}, ckpt)

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(
                {"model": model.state_dict(), "charset": charset},
                out_dir / "best.pth",
            )
            logger.info("New best val_loss=%.4f -> saved best.pth", best_val)
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Final native save.
    final = out_dir / "recognizer.pth"
    torch.save({"model": model.state_dict(), "charset": charset}, final)
    logger.info("Saved final recognizer to %s", final)

    if args.export_onnx:
        _export_onnx(model, out_dir, device)

    logger.info(
        "Done. Point config ocr.recognizer_weights at %s to use these weights.",
        final,
    )
    return 0


def _evaluate(model, loader, criterion, device) -> float:
    import torch

    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            logits = model(images)
            input_lengths = torch.full(
                (images.size(0),), logits.size(0), dtype=torch.long
            )
            loss = criterion(
                logits.log_softmax(2), targets, input_lengths, target_lengths
            )
            total += loss.item()
    return total / max(1, len(loader))


def _export_onnx(model, out_dir: Path, device) -> None:
    import torch

    model.eval()
    dummy = torch.randn(1, 1, 32, 256, device=device)  # (N, C, H, W)
    onnx_path = out_dir / "recognizer.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 3: "width"}},
        opset_version=17,
    )
    logger.info("Exported ONNX to %s", onnx_path)


if __name__ == "__main__":
    sys.exit(main())
