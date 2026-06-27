"""
preprocess.py

Stage 2 of the receipt-reading pipeline: locate a receipt within a photo,
crop it out, and correct its perspective so it appears as a flat,
top-down rectangle ready for OCR.

Pure classical computer vision (OpenCV) -- no trained model, no GPU needed.

Usage:
    python src/preprocess.py path/to/photo.jpg
    python src/preprocess.py path/to/photo.jpg --output path/to/cropped.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Given 4 (x, y) points in any order, return them ordered as
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest x-y
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest x-y

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform to flatten the quadrilateral region of
    `image` defined by `pts` into a straight, top-down rectangle.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def find_receipt_contour(image: np.ndarray, min_area_ratio: float = 0.1):
    """
    Locate the largest 4-cornered contour in `image`, assumed to be the
    receipt. Returns a (4, 2) float32 array of corner points, or None if
    no suitable contour was found.

    min_area_ratio: the candidate contour must cover at least this fraction
    of the total image area, to reject small false positives (a logo, a
    stray dark patch on the table, etc).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # close small gaps in the edges so contours form a clean loop
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:  # only check the largest few
        area = cv2.contourArea(contour)
        if area < image_area * min_area_ratio:
            break  # sorted descending, so nothing smaller will pass either

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    return None


def crop_and_deskew(image_path: str, output_path: str = None) -> np.ndarray:
    """
    Load an image, locate the receipt, crop and deskew it.

    Falls back to returning the original image unchanged (with a warning
    printed to stderr) if no 4-cornered contour could be found -- e.g. a
    low-contrast background, or a receipt that already fills the frame.

    Returns the resulting image as a numpy array (BGR, OpenCV convention).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    contour = find_receipt_contour(image)

    if contour is None:
        print(
            f"[warning] No receipt contour found in {image_path}; "
            "using the original image unchanged. If this happens often, "
            "the classical-CV step may need the fallback detector "
            "mentioned earlier.",
            file=sys.stderr,
        )
        result = image
    else:
        result = four_point_transform(image, contour)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Crop and deskew a receipt photo using classical computer vision."
    )
    parser.add_argument("image", help="Path to the input photo")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the cropped/deskewed image "
             "(default: <name>_cropped.jpg next to the input)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        in_path = Path(args.image)
        output_path = str(in_path.with_name(f"{in_path.stem}_cropped{in_path.suffix}"))

    crop_and_deskew(args.image, output_path)
    print(f"Saved cropped/deskewed image to: {output_path}")


if __name__ == "__main__":
    main()
