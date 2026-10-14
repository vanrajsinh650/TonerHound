"""Stage 29: Signature Region Detector.

Identifies handwriting/signature fields:
- Connected components analysis (large continuous cursive components)
- High stroke variability (handwriting signal with varying widths)
- Non-rectangular bounding contours
- Classifies as SIGNATURE if 3/3 match.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_signature_region(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """Detects handwritten signature mark in the cropped region.
    
    Returns:
        (is_signature, confidence)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    h, w = gray.shape[:2]
    if h < 10 or w < 20:
        return False, 0.0

    _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # 1. Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return False, 0.0

    # Filter out tiny noise and background
    valid_components = [
        stats[i] for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= 20
    ]
    if not valid_components:
        return False, 0.0

    # Condition 1: Major continuous connected component (cursive flow)
    max_comp_area = max(c[cv2.CC_STAT_AREA] for c in valid_components)
    total_ink = cv2.countNonZero(binary)
    has_large_component = (max_comp_area / max(1, total_ink)) > 0.25

    # Condition 2: Stroke variability (gradient of distances in distance transform)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_distances = dist_transform[binary > 0]
    stroke_var = float(np.var(ink_distances)) if len(ink_distances) > 0 else 0.0
    has_stroke_variability = stroke_var > 0.40

    # Condition 3: Non-rectangular shape (solidity < 0.65 or high aspect ratio)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    non_rectangular = False
    if contours:
        main_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_cnt)
        hull = cv2.convexHull(main_cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1.0, hull_area)
        if solidity < 0.65:
            non_rectangular = True

    # Classify as signature if 3/3 match
    matches = sum([has_large_component, has_stroke_variability, non_rectangular])
    if matches >= 3:
        return True, 0.92
    elif matches == 2:
        return True, 0.70
    return False, 0.0
