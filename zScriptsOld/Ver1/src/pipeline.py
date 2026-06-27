"""
pipeline.py

The end-to-end inference script: takes a photo and runs it through every
stage to produce a structured, categorized, stored receipt.

Currently wired up through stage 2 (crop & deskew). Stages 3-6 will be
added here as they're implemented.

Usage:
    python src/pipeline.py path/to/photo.jpg
"""

import argparse

from preprocess import crop_and_deskew

# from ocr import read_text
# from clean import normalize, correct
# from categorize import parse_items, categorize_items
# from storage import init_db, save_receipt


def run(image_path: str) -> None:
    print(f"[1/5] Cropping and deskewing {image_path} ...")
    cropped = crop_and_deskew(image_path)
    print("      done.")

    # print("[2/5] Reading text (OCR) ...")
    # lines = read_text(cropped)

    # print("[3/5] Cleaning text ...")
    # cleaned = [{"text": correct(normalize(l["text"])), **l} for l in lines]

    # print("[4/5] Parsing and categorizing items ...")
    # items = categorize_items(parse_items(cleaned))

    # print("[5/5] Saving to database ...")
    # init_db()
    # receipt_id = save_receipt({...}, items)
    # print(f"      saved as receipt #{receipt_id}")

    print(
        "\nStages 3-6 (OCR, cleaning, categorization, storage) aren't "
        "wired in yet -- only the crop/deskew step ran."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a receipt photo through the full reading pipeline."
    )
    parser.add_argument("image", help="Path to the input photo")
    args = parser.parse_args()
    run(args.image)


if __name__ == "__main__":
    main()
