"""Step 2: crop and deskew the receipt — classical OpenCV, no model.

Pipeline: grayscale -> blur -> Canny -> contours -> largest 4-point polygon
-> perspective warp to a top-down rectangle.

Failure handling:
  * No receipt-shaped contour above the area threshold -> ReceiptNotFoundError.
  * More than one large rectangular candidate -> MultipleReceiptsError
    (the spec assumes a single receipt per photo).
  * If a learned fallback detector path is configured, it is tried before
    raising ReceiptNotFoundError.

Everything that can be tuned lives in DetectionConfig.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import settings
from config.config import DetectionConfig
from utils.exceptions import MultipleReceiptsError, ReceiptNotFoundError
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points ordered TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]      # bottom-right has largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right has smallest y-x
    rect[3] = pts[np.argmax(diff)]   # bottom-left has largest y-x
    return rect


def _four_point_warp(image: np.ndarray, pts: np.ndarray, max_side: int) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = int(max(width_a, width_b))
    max_h = int(max(height_a, height_b))
    if max_w == 0 or max_h == 0:
        raise ReceiptNotFoundError("Degenerate receipt quadrilateral.")

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_w, max_h))

    # Downscale very large crops to keep OCR fast and memory sane.
    longest = max(warped.shape[:2])
    if longest > max_side:
        scale = max_side / longest
        warped = cv2.resize(
            warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return warped


def _candidate_quads(
    image: np.ndarray, cfg: DetectionConfig
) -> list[np.ndarray]:
    """Find rectangular contours that could be a receipt."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = cfg.gaussian_kernel | 1  # force odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
    # Close small gaps so receipt borders form continuous contours.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    img_area = image.shape[0] * image.shape[1]
    min_area = cfg.min_area_ratio * img_area

    quads: list[np.ndarray] = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < min_area:
            break  # everything after this is smaller
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, cfg.approx_epsilon_ratio * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype("float32"))
    return quads


def crop_and_deskew(
    image: np.ndarray, cfg: DetectionConfig | None = None
) -> np.ndarray:
    """Return a single top-down crop of the receipt.

    Raises ReceiptNotFoundError or MultipleReceiptsError as appropriate.
    """
    cfg = cfg or settings.detection
    quads = _candidate_quads(image, cfg)

    if len(quads) == 0:
        logger.warning("No receipt-shaped contour found; trying fallback.")
        fallback = _try_fallback(image, cfg)
        if fallback is not None:
            return fallback
        raise ReceiptNotFoundError(
            "No receipt-shaped contour found. The background may be "
            "low-contrast (white receipt on white surface)."
        )

    # Distinct large quads whose areas are comparable => multiple receipts.
    if len(quads) >= 2:
        a0 = cv2.contourArea(quads[0])
        a1 = cv2.contourArea(quads[1])
        if a1 > 0.6 * a0 and not _overlapping(quads[0], quads[1]):
            raise MultipleReceiptsError(
                f"Detected {len(quads)} receipt-sized regions; "
                "this pipeline assumes one receipt per photo."
            )

    warped = _four_point_warp(image, quads[0], cfg.output_max_side)
    logger.info("Cropped receipt to %sx%s", warped.shape[1], warped.shape[0])
    return warped


def _overlapping(q1: np.ndarray, q2: np.ndarray) -> bool:
    """True if two quads' bounding boxes overlap substantially."""
    r1 = cv2.boundingRect(q1.astype("int32"))
    r2 = cv2.boundingRect(q2.astype("int32"))
    x = max(r1[0], r2[0])
    y = max(r1[1], r2[1])
    xx = min(r1[0] + r1[2], r2[0] + r2[2])
    yy = min(r1[1] + r1[3], r2[1] + r2[3])
    inter = max(0, xx - x) * max(0, yy - y)
    smaller = min(r1[2] * r1[3], r2[2] * r2[3])
    return smaller > 0 and inter / smaller > 0.5


def _try_fallback(image: np.ndarray, cfg: DetectionConfig) -> np.ndarray | None:
    """Use a learned detector if one is configured, else None.

    The fallback is an optional Ultralytics YOLO model fine-tuned to box
    receipts. Training it is only needed if classical detection proves
    unreliable on your photos (see training/train_detector.py).
    """
    if not cfg.fallback_model_path:
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("fallback_model_path set but ultralytics not installed.")
        return None

    model = YOLO(cfg.fallback_model_path)
    results = model.predict(image, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    # Take the highest-confidence box and crop axis-aligned.
    best = boxes[int(boxes.conf.argmax())]
    x1, y1, x2, y2 = (int(v) for v in best.xyxy[0].tolist())
    crop = image[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    logger.info("Fallback detector produced a crop %sx%s", crop.shape[1], crop.shape[0])
    return crop
