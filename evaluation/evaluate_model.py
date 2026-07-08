"""
evaluation/evaluate_model.py

Benchmarks ReceiptParser against a folder of labeled receipt images and
reports accuracy across three axes:

  1. Text reading accuracy   - did the model correctly transcribe what's
                                printed on the receipt? (store name, total,
                                tax, each item's Japanese name and price)
  2. Categorization accuracy - did the model assign the correct category
                                (Groceries, Drink, Dining Out, ...) to each item?
  3. Time accuracy           - what fraction of receipts were parsed within
                                the time budget (default 3.0s), plus raw
                                timing stats (avg/median/min/max).

Timing note: this measures the SAME thing the desktop app's stopwatch does.
In main.py the app records time.time() right after deskewing (at the start of
the scan) and again when the result returns, i.e. plain wall-clock around
parse_receipt_image() including network round-trip, Groq queue, and inference.
This evaluator mirrors that exactly: deskew/crop happen OUTSIDE the timer, and
wall-clock is measured around the parse call. It deliberately does NOT use
Groq's server-side usage.total_time, so the numbers match what a user sees.

Ground truth format: see evaluation/ground_truth.json (or run
evaluation/generate_template.py to scaffold one from your image folder).

Usage:
    python evaluation/evaluate_model.py --images-dir path/to/receipts
    python evaluation/evaluate_model.py --images-dir path/to/receipts \
        --ground-truth evaluation/ground_truth.json --time-threshold 3.0
"""

import argparse
import csv
import difflib
import json
import os
import statistics
import sys
import time
from datetime import datetime

# Make the project root importable regardless of how this script is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processing import deskew_and_crop, opencv_to_pil
from inference.pipeline import ReceiptParser

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

# HEIC/HEIF support (iPhone photos). pillow-heif is a listed requirement; if
# it's somehow missing we degrade gracefully to the base formats.
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    SUPPORTED_EXTENSIONS.extend([".heic", ".heif"])
except Exception:
    pass

SUPPORTED_EXTENSIONS = tuple(SUPPORTED_EXTENSIONS)

TEXT_SIMILARITY_THRESHOLD = 0.85  # how close two strings must be to count as "correct"
ITEM_MATCH_THRESHOLD = 0.5        # minimum similarity to pair a predicted item with a GT item


def _normalize(s):
    return (s or "").strip().lower()


def text_similar(a, b, threshold=TEXT_SIMILARITY_THRESHOLD):
    """
    Fuzzy string comparison so minor spacing/formatting/extra-wording
    differences (e.g. model appending an English name) don't count as wrong.
    """
    a, b = _normalize(a), _normalize(b)
    if not a and not b:
        return True
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def numbers_equal(a, b):
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return a == b


def match_items(predicted_items, expected_items):
    """
    Greedily pair each expected (ground-truth) item with the best-matching
    predicted item by japanese_name similarity, so item order/count
    differences don't break scoring.

    Returns:
        pairs: list of (expected_item, predicted_item_or_None), one per expected item
        extra_items: predicted items that didn't match any expected item
                      (possible hallucinated/duplicated extractions)
    """
    remaining = list(enumerate(predicted_items))
    pairs = []
    for exp_item in expected_items:
        exp_name = (exp_item.get("japanese_name") or "").strip()
        best_pos, best_score = None, 0.0
        for pos, (_, pred_item) in enumerate(remaining):
            pred_name = (pred_item.get("japanese_name") or "").strip()
            score = difflib.SequenceMatcher(None, exp_name, pred_name).ratio()
            if score > best_score:
                best_score, best_pos = score, pos
        if best_pos is not None and best_score >= ITEM_MATCH_THRESHOLD:
            pairs.append((exp_item, remaining.pop(best_pos)[1]))
        else:
            pairs.append((exp_item, None))
    extra_items = [item for _, item in remaining]
    return pairs, extra_items


def score_image(predicted: dict, expected: dict):
    """Compare one model prediction against ground truth. Returns a dict of counts + notes."""
    text_correct = text_total = 0
    cat_correct = cat_total = 0
    mismatches = []

    # ---- Header fields (text reading) ----
    header_checks = [
        ("store_name", text_similar),
        ("total_amount", numbers_equal),
        ("tax_amount", numbers_equal),
    ]
    for field, cmp_fn in header_checks:
        text_total += 1
        pred_val, exp_val = predicted.get(field), expected.get(field)
        if cmp_fn(pred_val, exp_val):
            text_correct += 1
        else:
            mismatches.append(f"{field}: expected={exp_val!r} got={pred_val!r}")

    # ---- Items ----
    predicted_items = predicted.get("items", []) or []
    expected_items = expected.get("items", []) or []
    pairs, extra_items = match_items(predicted_items, expected_items)

    for i, (exp_item, pred_item) in enumerate(pairs):
        label = exp_item.get("japanese_name") or f"item[{i}]"

        # Text reading fields: what's literally printed on the receipt
        for field, cmp_fn in (("japanese_name", text_similar), ("price", numbers_equal)):
            text_total += 1
            exp_val = exp_item.get(field)
            pred_val = pred_item.get(field) if pred_item else None
            if pred_item is not None and cmp_fn(pred_val, exp_val):
                text_correct += 1
            else:
                mismatches.append(f"{label}.{field}: expected={exp_val!r} got={pred_val!r}")

        # Categorization field: a judgment call, not read directly off the paper
        cat_total += 1
        exp_cat = exp_item.get("category") or ""
        pred_cat = (pred_item.get("category") if pred_item else None) or ""
        if pred_item is not None and _normalize(pred_cat) == _normalize(exp_cat):
            cat_correct += 1
        else:
            mismatches.append(f"{label}.category: expected={exp_cat!r} got={pred_cat!r}")

    if extra_items:
        names = ", ".join(str(it.get("japanese_name", "?")) for it in extra_items)
        mismatches.append(f"{len(extra_items)} unexpected extra item(s) extracted: {names}")

    return {
        "text_correct": text_correct,
        "text_total": text_total,
        "cat_correct": cat_correct,
        "cat_total": cat_total,
        "extra_items": len(extra_items),
        "mismatches": mismatches,
    }


def run_evaluation(images_dir, ground_truth_path, time_threshold, output_dir, delay=2.5):
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    parser = ReceiptParser()

    image_files = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )
    skipped = [f for f in image_files if f not in ground_truth]
    unused_gt = [k for k in ground_truth if k not in image_files]
    evaluated_files = [f for f in image_files if f in ground_truth]

    if skipped:
        print(f"[WARN] {len(skipped)} image(s) have no ground-truth entry and will be skipped:")
        for f in skipped:
            print(f"    - {f}")
    if unused_gt:
        print(f"[WARN] {len(unused_gt)} ground-truth entr(y/ies) have no matching image file:")
        for f in unused_gt:
            print(f"    - {f}")
    if not evaluated_files:
        print("[ERROR] No images with matching ground-truth entries were found. Nothing to evaluate.")
        sys.exit(1)

    per_image_results = []
    parse_times = []

    for idx, filename in enumerate(evaluated_files):
        # Pace requests like a human using the app: pause BETWEEN receipts so a
        # tight loop doesn't trip Groq's per-minute rate limit. Once tripped,
        # Groq throttles/queues later requests server-side, which is exactly why
        # the first ~7 receipts are fast (~2s) and the rest balloon to 7-9s.
        # This sleep is outside the timing block, so it never affects parse_time.
        if idx > 0 and delay > 0:
            time.sleep(delay)

        image_path = os.path.join(images_dir, filename)
        expected = ground_truth[filename]
        print(f"Processing {filename} ...")

        try:
            # Deskew/crop happen OUTSIDE the timer, exactly like the app:
            # main.py preprocesses first, then starts its stopwatch.
            cv_img = deskew_and_crop(image_path)
            pil_img = opencv_to_pil(cv_img)

            # Wall-clock around the parse call, matching the app's stopwatch
            # (time.time() at scan start -> time.time() when the result returns).
            # Includes network round-trip, Groq queue, and inference -- i.e.
            # the real latency a user of the interface experiences.
            start = time.time()
            predicted = parser.parse_receipt_image(pil_img)
            elapsed = time.time() - start
            parse_times.append(elapsed)

            scores = score_image(predicted, expected)
            scores.update({
                "image": filename,
                "parse_time_sec": round(elapsed, 3),
                "time_ok": elapsed <= time_threshold,
                "error": "",
            })
            per_image_results.append(scores)

        except Exception as e:
            per_image_results.append({
                "image": filename,
                "text_correct": 0, "text_total": 0,
                "cat_correct": 0, "cat_total": 0,
                "extra_items": 0,
                "parse_time_sec": None, "time_ok": False,
                "mismatches": [], "error": str(e),
            })
            print(f"    [ERROR] {e}")

    # ---------------- Aggregate ----------------
    total_text_correct = sum(r["text_correct"] for r in per_image_results)
    total_text_total = sum(r["text_total"] for r in per_image_results)
    total_cat_correct = sum(r["cat_correct"] for r in per_image_results)
    total_cat_total = sum(r["cat_total"] for r in per_image_results)
    time_oks = [r["time_ok"] for r in per_image_results if r["parse_time_sec"] is not None]
    num_failures = sum(1 for r in per_image_results if r["error"])

    text_accuracy = (total_text_correct / total_text_total * 100) if total_text_total else 0.0
    cat_accuracy = (total_cat_correct / total_cat_total * 100) if total_cat_total else 0.0
    time_accuracy = (sum(time_oks) / len(time_oks) * 100) if time_oks else 0.0

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "images_dir": images_dir,
        "ground_truth_path": ground_truth_path,
        "num_images_evaluated": len(evaluated_files),
        "num_images_skipped_no_gt": len(skipped),
        "num_failures": num_failures,
        "time_threshold_sec": time_threshold,
        "time_source": "wall_clock (matches app stopwatch: network + queue + inference)",
        "inter_receipt_delay_sec": delay,
        "text_reading_accuracy_pct": round(text_accuracy, 2),
        "categorization_accuracy_pct": round(cat_accuracy, 2),
        "time_accuracy_pct": round(time_accuracy, 2),
        "avg_time_sec": round(statistics.mean(parse_times), 3) if parse_times else None,
        "median_time_sec": round(statistics.median(parse_times), 3) if parse_times else None,
        "min_time_sec": round(min(parse_times), 3) if parse_times else None,
        "max_time_sec": round(max(parse_times), 3) if parse_times else None,
    }

    # ---------------- Print report ----------------
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images evaluated   : {summary['num_images_evaluated']}")
    print(f"Skipped (no GT)    : {summary['num_images_skipped_no_gt']}")
    print(f"Failed to process  : {summary['num_failures']}")
    print("-" * 60)
    print(f"Text reading accuracy   : {summary['text_reading_accuracy_pct']}%  "
          f"({total_text_correct}/{total_text_total} fields)")
    print(f"Categorization accuracy : {summary['categorization_accuracy_pct']}%  "
          f"({total_cat_correct}/{total_cat_total} items)")
    print(f"Time accuracy (<= {time_threshold}s)  : {summary['time_accuracy_pct']}%  "
          f"({sum(time_oks)}/{len(time_oks)} images)")
    if parse_times:
        print(f"Avg / median / min / max time: {summary['avg_time_sec']}s / "
              f"{summary['median_time_sec']}s / {summary['min_time_sec']}s / {summary['max_time_sec']}s")
    print("  (wall-clock, same as the app stopwatch: network + queue + inference)")
    print("=" * 60 + "\n")

    # ---------------- Save report files ----------------
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "text_correct", "text_total", "text_accuracy_pct",
            "cat_correct", "cat_total", "cat_accuracy_pct",
            "extra_items", "parse_time_sec", "time_ok", "error",
        ])
        for r in per_image_results:
            text_pct = round(r["text_correct"] / r["text_total"] * 100, 1) if r["text_total"] else ""
            cat_pct = round(r["cat_correct"] / r["cat_total"] * 100, 1) if r["cat_total"] else ""
            writer.writerow([
                r["image"], r["text_correct"], r["text_total"], text_pct,
                r["cat_correct"], r["cat_total"], cat_pct,
                r["extra_items"], r["parse_time_sec"], r["time_ok"], r["error"],
            ])

    details_path = os.path.join(output_dir, f"evaluation_details_{timestamp}.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": per_image_results}, f, indent=2, ensure_ascii=False)

    print(f"Per-image CSV report saved to : {csv_path}")
    print(f"Full mismatch details saved to: {details_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ReceiptParser accuracy against labeled receipt images."
    )
    parser.add_argument("--images-dir", required=True, help="Folder containing receipt images")
    parser.add_argument(
        "--ground-truth",
        default=os.path.join("evaluation", "ground_truth.json"),
        help="Path to ground_truth.json (default: evaluation/ground_truth.json)",
    )
    parser.add_argument(
        "--time-threshold", type=float, default=3.0,
        help="Seconds a parse must finish within to count as 'on time' (default: 3.0)",
    )
    parser.add_argument("--output-dir", default="outputs", help="Where to write report files (default: outputs/)")
    parser.add_argument(
        "--delay", type=float, default=5,
        help="Seconds to pause BETWEEN receipts so a fast back-to-back loop doesn't "
             "trip Groq's per-minute rate limit (which throttles later requests and "
             "inflates their time). The pause is OUTSIDE the timer, so it never counts "
             "toward measured parse times. Set 0 to disable. Default: 2.5",
    )
    args = parser.parse_args()

    run_evaluation(args.images_dir, args.ground_truth, args.time_threshold,
                   args.output_dir, delay=args.delay)


if __name__ == "__main__":
    main()
