"""Dataset registry and loaders.

The pipeline needs data for two trainable stages:

  1. OCR recognition fine-tuning  -> needs (line image, transcription) pairs.
  2. Optional receipt detector    -> needs (photo, receipt bounding box).

Recommended datasets (all reusable, and easy to swap):

  * CORD  (Consolidated Receipt Dataset) — receipt photos with word-level
    boxes + transcriptions + field labels. Best fit for OCR recognition
    fine-tuning. HuggingFace id: "naver-clova-ix/cord-v2".
  * SROIE (ICDAR 2019 Task) — scanned receipts with box+text; English-heavy
    but useful for layout. Good for detection/parse evaluation.
  * Your own photos — drop them in datasets/custom/ following the same
    manifest format (see DatasetSpec below) and point --dataset at it.

Swapping datasets: every loader returns the same RecognitionSample /
DetectionSample dataclass, so training code never depends on the source.
Register a new source by adding a DatasetSpec to REGISTRY or by passing a
local manifest path on the command line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RecognitionSample:
    """One cropped text line and its ground-truth transcription."""

    image_path: str
    text: str


@dataclass
class DetectionSample:
    """One full photo and the receipt bounding box(es)."""

    image_path: str
    boxes: list[tuple[int, int, int, int]]  # x, y, w, h


@dataclass
class DatasetSpec:
    name: str
    kind: str  # "recognition" or "detection"
    loader: Callable[[Path], Iterator]
    hf_id: str | None = None
    note: str = ""


# --- CORD loader (HuggingFace) ----------------------------------------------
def _load_cord(root: Path) -> Iterator[RecognitionSample]:
    """Yield line-level recognition samples from CORD.

    Downloads via the `datasets` library on first call and caches crops under
    datasets/cord/lines/. Each CORD example has an image plus ground-truth
    word boxes; we crop each word/line and pair it with its text.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "pip install datasets to download CORD, or supply a local manifest."
        ) from exc

    import cv2
    import numpy as np

    lines_dir = root / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / "recognition_manifest.jsonl"

    if manifest.exists():
        logger.info("Using cached CORD manifest %s", manifest)
        yield from _read_manifest(manifest)
        return

    logger.info("Downloading CORD (naver-clova-ix/cord-v2)...")
    ds = load_dataset("naver-clova-ix/cord-v2", split="train")
    count = 0
    with manifest.open("w", encoding="utf-8") as mf:
        for i, ex in enumerate(ds):
            image = np.array(ex["image"].convert("RGB"))[:, :, ::-1]  # to BGR
            gt = json.loads(ex["ground_truth"])
            for j, line in enumerate(gt.get("valid_line", [])):
                for k, word in enumerate(line.get("words", [])):
                    q = word["quad"]
                    xs = [q["x1"], q["x2"], q["x3"], q["x4"]]
                    ys = [q["y1"], q["y2"], q["y3"], q["y4"]]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs)) - x, int(max(ys)) - y
                    if w <= 2 or h <= 2:
                        continue
                    crop = image[y : y + h, x : x + w]
                    if crop.size == 0:
                        continue
                    text = word["text"]
                    out = lines_dir / f"{i}_{j}_{k}.png"
                    ok, buf = cv2.imencode(".png", crop)
                    if not ok:
                        continue
                    buf.tofile(str(out))
                    rec = {"image_path": str(out), "text": text}
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
    logger.info("Prepared %d CORD recognition crops", count)
    yield from _read_manifest(manifest)


def _read_manifest(path: Path) -> Iterator[RecognitionSample]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield RecognitionSample(rec["image_path"], rec["text"])


def _load_local_recognition(root: Path) -> Iterator[RecognitionSample]:
    """Load a user-provided recognition manifest.

    Expected file: <root>/recognition_manifest.jsonl with one JSON object per
    line: {"image_path": "...", "text": "..."}.
    """
    manifest = root / "recognition_manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"No recognition manifest at {manifest}. See datasets.py docstring."
        )
    yield from _read_manifest(manifest)


REGISTRY: dict[str, DatasetSpec] = {
    "cord": DatasetSpec(
        name="cord",
        kind="recognition",
        loader=_load_cord,
        hf_id="naver-clova-ix/cord-v2",
        note="Receipt photos with word boxes + text. Best for OCR fine-tuning.",
    ),
    "local": DatasetSpec(
        name="local",
        kind="recognition",
        loader=_load_local_recognition,
        note="Your own recognition_manifest.jsonl under datasets/custom/.",
    ),
}


def get_dataset(name: str, root: Path | None = None) -> list:
    """Materialize a registered dataset into a list of samples."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(REGISTRY)}")
    spec = REGISTRY[name]
    root = root or (Path(settings.datasets_dir) / name)
    root.mkdir(parents=True, exist_ok=True)
    logger.info("Loading dataset '%s' from %s", name, root)
    samples = list(spec.loader(root))
    logger.info("Dataset '%s' -> %d samples", name, len(samples))
    return samples
