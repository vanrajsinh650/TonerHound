"""Stage 24: Region Classifier (Text vs Non-Text).

For regions that remain empty after OCR extraction:
- Computes pixel statistics: edge density, ink coverage, symmetry
- Classifies as VISUAL_NON_TEXT (e.g. checkbox, stamp, signature) or EMPTY
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def classify_region_text_vs_nontext(
    gray_crop: np.ndarray,
    text_token_count: int = 0,
) -> dict[str, Any]:
    """Classifies region image into 'TEXT', 'VISUAL_NON_TEXT', or 'BLANK'.
    
    Returns:
        dict with classification and calculated metrics.
    """
    if gray_crop is None or gray_crop.size == 0:
        return {"category": "BLANK", "edge_density": 0.0, "ink_coverage": 0.0}

    h, w = gray_crop.shape[:2]
    total_pixels = max(1, h * w)

    # 1. Ink coverage (foreground pixels assuming dark ink on light background)
    _, binary = cv2.threshold(gray_crop, 200, 255, cv2.THRESH_BINARY_INV)
    ink_pixels = cv2.countNonZero(binary)
    ink_coverage = ink_pixels / total_pixels

    # 2. Edge density using Canny edge detector
    edges = cv2.Canny(gray_crop, 50, 150)
    edge_pixels = cv2.countNonZero(edges)
    edge_density = edge_pixels / total_pixels

    # 3. Symmetry
    # Horizontal symmetry (flipped horizontally)
    flipped_lr = cv2.flip(binary, 1)
    overlap_lr = cv2.bitwise_and(binary, flipped_lr)
    sym_h = cv2.countNonZero(overlap_lr) / max(1, ink_pixels)

    if ink_coverage < 0.01 and edge_density < 0.01:
        category = "BLANK"
    elif text_token_count > 0:
        category = "TEXT"
    elif edge_density > 0.04 and ink_coverage > 0.02:
        # High edge density, visible ink, but no text tokens -> Visual element (checkbox / stamp / signature)
        category = "VISUAL_NON_TEXT"
    else:
        category = "UNRESOLVED"

    return {
        "category": category,
        "ink_coverage": float(ink_coverage),
        "edge_density": float(edge_density),
        "symmetry_h": float(sym_h),
    }
