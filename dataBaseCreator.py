"""
dataBaseCreator.py

Scans a folder of receipt images and produces a single JSON file mapping each
image filename to its parsed receipt, using the project's existing pipeline:

    utils.image_processing.deskew_and_crop   (classical CV preprocessing)
    inference.pipeline.ReceiptParser         (Groq vision, or offline Simulation Mode)

The output is trimmed to the ground-truth schema (store_name, total_amount,
tax_amount, and per-item japanese_name / english_name / category / price).
The verbose `note` and top-level `savings_advice` fields the model returns are
intentionally dropped, since they are generation, not text read off the receipt.

Like generate_template.py, this is safe to re-run: by default existing entries
in the output file are kept and only new images are parsed and appended. Pass
--overwrite to re-parse everything.

Usage:
    python dataBaseCreator.py --images-dir path/to/receipts
    python dataBaseCreator.py --images-dir path/to/receipts --output evaluation/ground_truth.json
    python dataBaseCreator.py --images-dir path/to/receipts --overwrite

Notes:
    * If GROQ_API_KEY is set, this hits the real Groq API (uses quota + network).
      Otherwise ReceiptParser falls back to offline Simulation Mode.
    * .HEIC images (e.g. straight off an iPhone) are supported if the optional
      `pillow-heif` package is installed (pip install pillow-heif). Without it,
      only the formats the app already handles (.png/.jpg/.jpeg/.bmp) are read.
"""

import argparse
import json
import os
import sys

# Make the project root importable regardless of how this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_processing import deskew_and_crop, opencv_to_pil
from inference.pipeline import ReceiptParser

# Base formats the app's image pipeline already supports.
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

# Best-effort HEIC/HEIF support (common for iPhone photos). Optional dependency:
# if pillow-heif is available we register it with Pillow so deskew_and_crop's
# Image.open() can read .heic files; otherwise we just skip them with a warning.
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    SUPPORTED_EXTENSIONS.extend([".heic", ".heif"])
    _HEIC_SUPPORTED = True
except Exception:
    _HEIC_SUPPORTED = False

SUPPORTED_EXTENSIONS = tuple(SUPPORTED_EXTENSIONS)


def _trim_to_schema(parsed: dict) -> dict:
    """
    Reduce a full ReceiptParser result to the ground-truth JSON schema:
    keep store_name / total_amount / tax_amount and, per item, only
    japanese_name / english_name / category / price.
    """
    items = []
    for item in parsed.get("items", []) or []:
        items.append({
            "japanese_name": item.get("japanese_name", ""),
            "english_name": item.get("english_name", ""),
            "category": item.get("category") or "Other",
            "price": item.get("price", 0),
        })

    return {
        "store_name": parsed.get("store_name", ""),
        "total_amount": parsed.get("total_amount", 0),
        "tax_amount": parsed.get("tax_amount", 0),
        "items": items,
    }


def scan_folder(images_dir, output_path, overwrite=False):
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        sys.exit(1)

    # Load any existing results so re-runs only parse new images (unless --overwrite).
    if os.path.exists(output_path) and not overwrite:
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                database = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] {output_path} exists but isn't valid JSON. Starting fresh.")
                database = {}
    else:
        database = {}

    image_files = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )

    if not image_files:
        hint = "" if _HEIC_SUPPORTED else " (install 'pillow-heif' to read .HEIC files)"
        print(f"[ERROR] No supported images found in {images_dir}{hint}.")
        sys.exit(1)

    parser = ReceiptParser()

    parsed_count = 0
    skipped_count = 0
    failed = []

    for filename in image_files:
        # Preserve any already-parsed entries on re-run.
        if filename in database and not overwrite:
            print(f"Skipping {filename} (already in {os.path.basename(output_path)})")
            skipped_count += 1
            continue

        image_path = os.path.join(images_dir, filename)
        print(f"Processing {filename} ...")

        try:
            cv_img = deskew_and_crop(image_path)
            pil_img = opencv_to_pil(cv_img)
            parsed = parser.parse_receipt_image(pil_img)
            database[filename] = _trim_to_schema(parsed)
            parsed_count += 1
        except Exception as e:
            print(f"    [ERROR] Failed to process {filename}: {e}")
            failed.append(filename)

    # Write the combined database (sorted keys for stable, diff-friendly output).
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2, ensure_ascii=False, sort_keys=True)

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"Newly parsed : {parsed_count}")
    print(f"Skipped      : {skipped_count} (already present)")
    print(f"Failed       : {len(failed)}")
    if failed:
        for f_ in failed:
            print(f"    - {f_}")
    print(f"Output       : {output_path}  ({len(database)} receipt(s) total)")

    return database


def main():
    ap = argparse.ArgumentParser(
        description="Scan a folder of receipt images into a ground-truth-style JSON database."
    )
    ap.add_argument("--images-dir", required=True, help="Folder containing receipt images")
    ap.add_argument(
        "--output",
        default=os.path.join("evaluation", "ground_truth.json"),
        help="Where to write the JSON database (default: evaluation/ground_truth.json)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-parse every image, replacing existing entries (default: keep existing, add new).",
    )
    args = ap.parse_args()

    scan_folder(args.images_dir, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
