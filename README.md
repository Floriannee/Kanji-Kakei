# Japanese Receipt OCR Pipeline

Read a photo of a Japanese receipt, locate and flatten the receipt, OCR the
text, parse it into structured items, categorize each item, and export a CSV
(plus an SQLite record). Built for Python 3.13 on Windows 11, reusing
pretrained models and fine-tuning only the OCR recognizer.

## What it does

```
photo.jpg
  -> load + validate            (utils/image_io.py)
  -> crop + deskew  [OpenCV]    (inference/detect.py)
  -> OCR            [EasyOCR]   (inference/ocr.py)
  -> clean/normalize            (inference/clean.py)
  -> parse + categorize         (inference/parse.py)
  -> CSV + SQLite               (inference/csv_writer.py, database/db.py)
```

Assumptions: one receipt per photo, good lighting. CSV columns are
`product_name, price, tax, store` plus confidence scores
(`name_confidence`, `category`, `category_confidence`).

## Install (Windows 11, Python 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# For training, dataset download, ONNX, the optional detector, and tests:
pip install -r requirements-extra.txt
```

Notes on extensions you need and why:

- **opencv-python** — all classical crop/deskew (grayscale, Canny, contours,
  perspective warp). No model involved.
- **easyocr** — the default OCR engine; bundles a pretrained Japanese+English
  text detector and recognizer. Pulls in **torch**/**torchvision**. For a GPU
  build, install torch from pytorch.org *first*, then easyocr.
- **neologdn**, **jaconv** (and optional **mojimoji**, **sudachipy**) —
  Japanese text normalization (width, kana, symbol folding).
- **sentence-transformers** — multilingual-e5 embeddings for category
  assignment by cosine similarity.
- **pyyaml** — config and category taxonomy files.
- **datasets** — downloads CORD for OCR fine-tuning (training only).
- **ultralytics** — optional YOLO detector used only as a fallback when the
  classical crop fails on low-contrast backgrounds (training + inference).
- **onnx**, **onnxruntime** — ONNX export of trained models.
- **fugashi**, **unidic-lite** — tokenizer backend for the optional Japanese
  BERT correction step.

PaddleOCR is supported as an alternative engine (`ocr.engine: paddleocr` in
`config/config.yaml`); install `paddlepaddle` + `paddleocr` if you want it.

## Quick start (inference)

```powershell
# One photo -> outputs/<name>.csv (and a row in database/receipts.db)
python -m inference.run path\to\receipt.jpg

# A whole folder
python -m inference.run path\to\folder --batch

# Custom CSV path, skip the DB, also save the deskewed crop
python -m inference.run receipt.jpg --csv out.csv --no-db --save-crop
```

The first run downloads EasyOCR's pretrained weights (cached afterward) and,
on first categorization, the e5 embedding model.

## Training

Fine-tune the OCR recognizer on receipt data (the one run that matters):

```powershell
python -m training.train_ocr --dataset cord --epochs 30 --export-onnx
```

This downloads CORD, crops line images, trains a CRNN recognizer with fixed
seeds, saves checkpoints and a final `models/ocr_recognizer/recognizer.pth`
(+ `.onnx`), and writes the charset. To use the fine-tuned weights, set in
`config/config.yaml`:

```yaml
ocr:
  recognizer_weights: models/ocr_recognizer/recognizer.pth
  recognizer_network: standard
```

Optional receipt detector (only if classical detection struggles on your
photos):

```powershell
python -m training.train_detector --data datasets\detect\data.yaml --epochs 50
```

Then set `detection.fallback_model_path` in the config to the produced
weights.

## Using other datasets

Everything goes through `datasets/datasets.py`. To plug in your own OCR data,
create `datasets/custom/recognition_manifest.jsonl` with one object per line:

```json
{"image_path": "C:/data/lines/0001.png", "text": "牛乳 1L"}
```

then train with `--dataset local`. Add new sources by registering a
`DatasetSpec` in `REGISTRY`; the training code never depends on the source
format.

## Evaluation and benchmarks

Metrics are defined precisely in `utils/metrics.py`, one per stage. Run the
benchmark against a labeled set:

```powershell
python -m benchmarks.benchmark --manifest benchmarks\eval_manifest.json
```

It reports each stage next to its target:

| Stage                         | Target  |
| ----------------------------- | ------- |
| Receipt detection (IoU >=0.5) | 95%+    |
| OCR character accuracy        | 90-95%  |
| Structured item extraction F1 | 80-90%  |
| End-to-end pipeline           | 75-85%  |

See `benchmarks/eval_manifest.example.json` for the manifest format.

## Tests

The deterministic parts (loading, detection geometry, parsing, metrics, CSV,
DB, config) are covered and run without downloading any model:

```powershell
pip install -r requirements-extra.txt
pytest -q
```

## Configuration

All paths and tunables live in `config/config.py` with defaults mirrored in
`config/config.yaml`. Override per-run with environment variables
`RECEIPT_SEED`, `RECEIPT_LOG_LEVEL`, `RECEIPT_GPU`. The category taxonomy is
in `config/categories.yaml` and is read at runtime, so editing categories
needs no code change.

## Failure handling

Each documented failure maps to a specific exception (`utils/exceptions.py`)
and is reported, not crashed, by `process_image`:

| Failure            | Exception              |
| ------------------ | ---------------------- |
| receipt not found  | ReceiptNotFoundError   |
| OCR failure        | OCRFailureError        |
| empty image        | EmptyImageError        |
| unsupported file   | UnsupportedFileError   |
| corrupted image    | CorruptedImageError    |
| multiple receipts  | MultipleReceiptsError  |

## Logging

Standard library `logging` throughout (no print). Logs stream to stderr and
to `outputs/logs/pipeline.log`. Set level via `--log-level` or
`RECEIPT_LOG_LEVEL`.

## Project layout

```
project/
├── datasets/      dataset registry + loaders (CORD, local, swappable)
├── models/        saved weights, checkpoints, ONNX exports
├── training/      train_ocr.py (CRNN fine-tune), train_detector.py, crnn.py
├── inference/     detect, ocr, clean, parse, csv_writer, pipeline, run (CLI)
├── utils/         logging, seeding, image I/O, exceptions, metrics
├── config/        config.py, config.yaml, categories.yaml
├── outputs/       CSVs, crops, logs
├── database/      SQLite store (receipts.db)
├── benchmarks/    benchmark.py + eval manifest
├── tests/         pytest suite (no model downloads needed)
├── requirements.txt
└── requirements-extra.txt
```
