# Model Evaluation

Benchmarks `ReceiptParser` against a folder of receipt images you've already
labeled with the correct answers, and reports three accuracy numbers:

| Metric | What it measures |
|---|---|
| **Text reading accuracy** | Did the model correctly transcribe what's printed on the receipt: store name, total, tax, and each item's Japanese name + price? |
| **Categorization accuracy** | Did the model assign the correct category (Food, Drink, Household, ...) to each item? |
| **Time accuracy** | What % of receipts were parsed within the time budget (default 3.0s)? Raw avg/median/min/max timings are also reported. |

English item names and the `note` cultural-context field are **not** scored —
those are translation/generation, not text reading off the receipt. If you'd
like them scored too, that's a small addition to `score_image()` in
`evaluate_model.py`.

## 1. Label your images

Ground truth is stored as a single JSON file, `evaluation/ground_truth.json`,
mapping each image **filename** to its correct answer key. See
`ground_truth.example.json` for a fully worked example.

The easiest way to create it:

```bash
python evaluation/generate_template.py --images-dir path/to/your/receipts
```

This scans the folder and writes a blank entry per image into
`evaluation/ground_truth.json`:

```json
{
  "receipt_001.jpg": {
    "store_name": "",
    "total_amount": 0,
    "tax_amount": 0,
    "items": [
      {"japanese_name": "", "english_name": "", "category": "", "price": 0}
    ]
  }
}
```

Open the file and fill in the true values for each image (add/remove item
objects as needed — one per line item actually on that receipt). It's safe to
re-run `generate_template.py` later if you add more images to the folder;
it only appends new blank entries and never touches ones you've already
filled in.

Valid categories (must match exactly, case-insensitive) are whatever the
model is instructed to use in `inference/pipeline.py`'s `PROMPT` — currently:
`Food, Drink, Snack, Household, Personal Care, Stationery, Other`. Change the
list in both places together if you want different categories.

## 2. Run the evaluation

```bash
python evaluation/evaluate_model.py --images-dir path/to/your/receipts
```

Optional flags:

```bash
python evaluation/evaluate_model.py \
  --images-dir path/to/your/receipts \
  --ground-truth evaluation/ground_truth.json \   # default shown
  --time-threshold 3.0 \                          # seconds; default shown
  --output-dir outputs                             # default shown
```

This actually calls `ReceiptParser.parse_receipt_image()` for every labeled
image — if `GROQ_API_KEY` is set it hits the real Groq API (so it uses your
API quota and needs network access); if not, it runs in the app's existing
offline Simulation Mode.

Images present in the folder but missing from `ground_truth.json` are
skipped with a warning (not scored). Images that fail to load or parse are
reported as failures and excluded from the timing stats, but still count as
0% for that image's text/categorization fields.

## 3. Read the results

A summary prints to the console, e.g.:

```
Text reading accuracy   : 91.2%  (156/171 fields)
Categorization accuracy : 84.0%  (42/50 items)
Time accuracy (<= 3.0s)  : 96.0%  (24/25 images)
Avg / median / min / max time: 1.842s / 1.71s / 0.93s / 4.02s
```

Two files are also written to `outputs/`:
- `evaluation_report_<timestamp>.csv` — one row per image (open in Excel/Sheets)
- `evaluation_details_<timestamp>.json` — the same data plus a `mismatches`
  list per image explaining exactly which fields were wrong and what the
  model returned instead of the expected value. This is the file to check
  when you want to know *why* accuracy is lower than expected.

## How matching works

Items aren't compared by list position — the model might extract them in a
different order, or miss/duplicate one. Each ground-truth item is paired with
whichever predicted item has the most similar `japanese_name` (using
`difflib`), so a missing or reordered item doesn't cascade into every item
after it being marked wrong. Predicted items that don't match anything in
your ground truth are logged as "extra items" (possible hallucinations or
duplicate extractions) but aren't folded into the accuracy percentage, so the
denominator always stays anchored to your labeled answer key.

Text fields use fuzzy matching (85% similarity, or a plain substring match)
so formatting differences like added spacing or an appended English name
don't count as a miss. Numbers and categories require an exact match.
