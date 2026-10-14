"""Stage 30: Stamp Detector.

Detects document stamps (official seals, received stamps):
- Circular / oval contour or Hough Circle transform detection
- Saturated color check (red, purple, blue ink) in HSV color space
- Text presence within circular boundary
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_stamp_region(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """Detects ink stamps / seals in the crop.
    
    Returns:
        (is_stamp, confidence)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, 0.0

    h, w = crop_bgr.shape[:2]
    if h < 15 or w < 15:
        return False, 0.0

    # 1. Color saturation check in HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    # Saturated color threshold (> 60 out of 255)
    saturated_pixels = cv2.countNonZero(cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY)[1])
    sat_ratio = saturated_pixels / (h * w)

    # Detect red or blue hue range
    # Red has hues 0-10 or 160-180; Blue has hues 90-130
    h_channel = hsv[:, :, 0]
    red_mask = ((h_channel < 12) | (h_channel > 165)) & (saturation > 50)
    blue_mask = ((h_channel >= 90) & (h_channel <= 130)) & (saturation > 50)
    colored_stamp_ink = np.count_nonzero(red_mask | blue_mask) / (h * w)

    # 2. Shape check: Hough circles or round contours
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    min_dim = min(h, w)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim // 2,
        param1=50,
        param2=30,
        minRadius=min_dim // 6,
        maxRadius=min_dim // 2,
    )
    has_circle = circles is not None and len(circles[0]) > 0

    # Check for ellipse / oval contour
    _, bin_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_oval = False
    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 0.15 * (h * w):
            ellipse = cv2.fitEllipse(cnt)
            axis1, axis2 = ellipse[1]
            if axis1 > 0 and axis2 > 0:
                ratio = min(axis1, axis2) / max(axis1, axis2)
                if ratio > 0.5:
                    has_oval = True
                    break

    has_round_shape = has_circle or has_oval

    if has_round_shape and (sat_ratio > 0.08 or colored_stamp_ink > 0.05):
        return True, 0.95
    elif has_round_shape and sat_ratio > 0.03:
        return True, 0.80
    elif colored_stamp_ink > 0.15:
        return True, 0.75

    return False, 0.0
