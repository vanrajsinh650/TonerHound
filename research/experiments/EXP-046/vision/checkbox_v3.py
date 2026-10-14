"""Stage 28: Checkbox Detection v3.

Advanced visual checkbox detector:
- Lower pixel density threshold (0.04 instead of 0.06)
- Stroke variance check for hand-drawn checks / cross marks
- Harris / Shi-Tomasi corner detection for square checkbox borders
"""

from __future__ import annotations

from typing import Sequence
import cv2
import numpy as np


def detect_checkbox_v3(
    crop_bgr: np.ndarray,
    min_density: float = 0.04,
) -> tuple[bool, str, float]:
    """Detects whether crop contains a checkbox, and its state.
    
    Returns:
        (is_checkbox, state ["CHECKED", "UNCHECKED", "UNKNOWN"], confidence)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, "UNKNOWN", 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    h, w = gray.shape[:2]
    if h < 6 or w < 6:
        return False, "UNKNOWN", 0.0

    # Binarize (ink is 255)
    _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    total_pixels = h * w
    ink_count = cv2.countNonZero(binary)
    density = ink_count / total_pixels

    if density < min_density:
        return False, "UNKNOWN", 0.0

    # 1. Corner detection for square box border
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.05, minDistance=3)
    has_corners = corners is not None and len(corners) >= 3

    # 2. Check inner region vs border
    margin_y = max(1, int(h * 0.2))
    margin_x = max(1, int(w * 0.2))
    inner = binary[margin_y : h - margin_y, margin_x : w - margin_x]
    inner_pixels = max(1, inner.shape[0] * inner.shape[1])
    inner_density = cv2.countNonZero(inner) / inner_pixels

    # 3. Stroke variance for hand-drawn mark
    # Compute horizontal and vertical projections
    h_proj = np.sum(inner == 255, axis=1)
    v_proj = np.sum(inner == 255, axis=0)
    stroke_var = float(np.var(h_proj) + np.var(v_proj))

    if inner_density > 0.08 or stroke_var > 15.0:
        state = "CHECKED"
        conf = 0.92 if has_corners else 0.85
    else:
        state = "UNCHECKED"
        conf = 0.88 if has_corners else 0.80

    return True, state, conf
