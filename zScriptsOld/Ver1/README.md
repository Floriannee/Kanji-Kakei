# Receipt reader

Reads a photo of a receipt, extracts the line items, categorizes each
product, and stores the result for later review.

## Pipeline

| Stage | File | Status | Approach |
|---|---|---|---|
| 1. Input photo | -- | -- | a photo from disk |
| 2. Crop & deskew | `src/preprocess.py` | **done** | OpenCV contour detection + perspective transform, no model |
| 3. Read text (OCR) | `src/ocr.py` | stub | PaddleOCR/EasyOCR, pretrained, optional fine-tune on CORD |
| 4. Clean text | `src/clean.py` | stub | rule-based normalization + pretrained Japanese LM, zero-shot |
| 5. Parse & categorize | `src/categorize.py` | stub | regex parsing + multilingual sentence-embedding similarity |
| 6. Store | `src/storage.py` | stub | SQLite |

`src/pipeline.py` is the end-to-end script that chains every stage
together; right now it only runs stage 2.

## Setup

```bash
pip install -r requirements.txt
```

Only the stage-2 dependencies are uncommented for now. Uncomment each
stage's block in `requirements.txt` as it gets implemented.

## Usage

Try it on the included sample image:

```bash
python src/pipeline.py data/sample/sample_receipt.jpg
```

Or run just the crop/deskew step directly on your own photo:

```bash
python src/preprocess.py path/to/your/photo.jpg
```

## Project layout

```
receipt-reader/
├── data/
│   ├── categories.json     # product category taxonomy (bilingual EN/JA)
│   └── sample/              # synthetic test photo for trying the pipeline
├── src/
│   ├── preprocess.py        # stage 2 - done
│   ├── ocr.py                # stage 3 - stub
│   ├── clean.py               # stage 4 - stub
│   ├── categorize.py          # stage 5 - stub
│   ├── storage.py             # stage 6 - stub
│   └── pipeline.py             # orchestrator
├── training/
│   └── README.md             # notes on the optional cloud-GPU OCR fine-tune
└── requirements.txt
```

## Notes

- No local GPU is required anywhere in this pipeline. The one optional
  training step (OCR fine-tuning) is meant to run on a free cloud GPU
  notebook -- see `training/README.md`.
- `data/categories.json` is a starting taxonomy; edit it freely, the
  categorization approach (embedding similarity) needs no retraining when
  categories change.
