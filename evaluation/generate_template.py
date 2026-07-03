"""
evaluation/generate_template.py

Scans a folder of receipt images and writes/updates a ground_truth.json
skeleton (one blank entry per image) for you to fill in with the correct
answers. Safe to re-run: existing entries are never overwritten, so you can
add more images to the folder later and just re-run this to append new
blank entries.

Usage:
    python evaluation/generate_template.py --images-dir path/to/receipts
    python evaluation/generate_template.py --images-dir path/to/receipts --output evaluation/ground_truth.json
"""

import argparse
import json
import os

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    parser = argparse.ArgumentParser(
        description="Generate/update a ground_truth.json skeleton from a folder of receipt images."
    )
    parser.add_argument("--images-dir", required=True, help="Folder containing receipt images")
    parser.add_argument(
        "--output",
        default=os.path.join("evaluation", "ground_truth.json"),
        help="Where to write the ground truth file (default: evaluation/ground_truth.json)",
    )
    args = parser.parse_args()

    image_files = sorted(
        f for f in os.listdir(args.images_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )

    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    added = 0
    for filename in image_files:
        if filename not in existing:
            existing[filename] = {
                "store_name": "",
                "total_amount": 0,
                "tax_amount": 0,
                "items": [
                    {
                        "japanese_name": "",
                        "english_name": "",
                        "category": "",
                        "price": 0,
                    }
                ],
            }
            added += 1

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Found {len(image_files)} image(s) in {args.images_dir}")
    print(f"Added {added} new blank entr(y/ies); {len(image_files) - added} already present.")
    print(f"Now open {args.output} and fill in the correct store_name, total_amount, tax_amount, and items.")


if __name__ == "__main__":
    main()
