import cv2
import numpy as np
import logging
from PIL import Image

# Initialize logger
logger = logging.getLogger("KanjiKakei.ImageProcessing")

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in standard clockwise format:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left point has the smallest sum, bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right has the smallest difference, bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def deskew_and_crop(image_path: str) -> np.ndarray:
    """
    Perform fast classical deskew and perspective warp on receipt image.
    If receipt contour cannot be isolated, falls back to original image and logs warning.
    Returns:
        np.ndarray: The preprocessed OpenCV image.
    """
    logger.info(f"Loading receipt image for preprocessing: {image_path}")
    # cv2.imread fails on non-ASCII paths on Windows; use PIL instead
    pil_img = Image.open(image_path).convert("RGB")
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if image is None:
        logger.error(f"Failed to load image from path: {image_path}")
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    orig = image.copy()
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny Edge Detection
    edged = cv2.Canny(blurred, 75, 200)
    
    # 4. Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by size, keep the top 10 largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    screen_cnt = None
    
    # 5. Look for largest 4-point contour that is sufficiently large
    total_area = image.shape[0] * image.shape[1]
    min_area = total_area * 0.05  # Must occupy at least 5% of the image area
    
    for c in contours:
        # Check area first to avoid warping tiny noise/artifacts
        area = cv2.contourArea(c)
        if area < min_area:
            continue
            
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If approximated contour has exactly 4 points, assume it is our receipt outline
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    # 6. Apply perspective transform if contour found; else return fallback
    if screen_cnt is not None:
        try:
            logger.info("4-point receipt contour successfully detected. Applying perspective warp.")
            pts = screen_cnt.reshape(4, 2)
            
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            
            # Compute width of warped image
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))
            
            # Compute height of warped image
            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((bl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))
            
            # Construct bird's-eye view coordinates
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")
            
            # Compute perspective warp matrix and warp
            transform_matrix = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(orig, transform_matrix, (max_width, max_height))
            
            # Return warped image
            return warped
            
        except Exception as warp_err:
            logger.warning(f"Error executing perspective warp: {warp_err}. Falling back to original image.")
            return orig
    else:
        logger.warning("Could not detect receipt contour (low contrast or complex background). Returning original image.")
        return orig

def opencv_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB Image."""
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)
