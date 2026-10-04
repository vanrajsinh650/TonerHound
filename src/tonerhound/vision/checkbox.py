"""Visual checkbox and signature detection with table grid-line removal.

Targets non-text visual groundings (checkboxes and signature ink regions)
via morphological operations and connected-component analysis.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np


def detect_checkbox_with_grid_removal(
    page_image: np.ndarray,
    region_bbox: Sequence[float],
) -> tuple[str, tuple[float, float, float, float]] | None:
    """Detect checkboxes embedded in table grid lines with morphological line subtraction.

    Args:
        page_image: BGR or grayscale numpy image array of the document page.
        region_bbox: [x, y, w, h] normalized COCO bounding box hint.

    Returns:
        Tuple of (state_string, normalized_bbox) or None if no checkbox found.
    """
    if page_image is None or page_image.size == 0 or len(region_bbox) != 4:
        return None

    h, w = page_image.shape[:2]
    x, y, bw, bh = region_bbox
    padding = 0.03

    px = max(0, int((x - padding) * w))
    py = max(0, int((y - padding) * h))
    pw = min(w - px, max(10, int((bw + 2 * padding) * w)))
    ph = min(h - py, max(10, int((bh + 2 * padding) * h)))

    crop = page_image[py : py + ph, px : px + pw]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # Binarize (invert so ink is white)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Remove long horizontal lines (grid borders)
    h_len = max(15, int(pw * 0.4))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    no_h = cv2.subtract(binary, h_lines)

    # Remove long vertical lines (grid borders)
    v_len = max(15, int(ph * 0.4))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(no_h, cv2.MORPH_OPEN, v_kernel)
    no_grid = cv2.subtract(no_h, v_lines)

    # Find candidate rectangular contours
    contours, _ = cv2.findContours(no_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 6 or area > (pw * ph * 0.7):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) not in (4, 5):
            continue

        x_r, y_r, w_r, h_r = cv2.boundingRect(approx)
        if w_r < 4 or h_r < 4:
            continue
        ar = w_r / max(h_r, 1)
        if not (0.5 <= ar <= 1.8):
            continue

        rect_area = w_r * h_r
        rect_score = area / rect_area if rect_area > 0 else 0
        if rect_score > best_score:
            best_score = rect_score
            best_box = (x_r, y_r, w_r, h_r)

    if best_box is None:
        non_zero = cv2.findNonZero(no_grid)
        if non_zero is not None and len(non_zero) > 10:
            x_r, y_r, w_r, h_r = cv2.boundingRect(non_zero)
            ar = w_r / max(h_r, 1)
            if 0.5 <= ar <= 2.0 and w_r > 4 and h_r > 4:
                best_box = (x_r, y_r, w_r, h_r)
            else:
                return None
        else:
            return None

    x_r, y_r, w_r, h_r = best_box
    interior = no_grid[y_r + h_r // 4 : y_r + 3 * h_r // 4, x_r + w_r // 4 : x_r + 3 * w_r // 4]
    core_density = np.sum(interior > 0) / max(interior.size, 1)
    state = "CHECKBOX_CHECKED" if core_density > 0.06 else "CHECKBOX_UNCHECKED"

    nx = (px + x_r) / w
    ny = (py + y_r) / h
    nw = w_r / w
    nh = h_r / h

    return state, (round(nx, 6), round(ny, 6), round(nw, 6), round(nh, 6))


def detect_signature_region(
    page_image: np.ndarray,
    region_bbox: Sequence[float],
) -> tuple[float, float, float, float] | None:
    """Detect signature region with horizontal ink spread and stroke variance.

    Args:
        page_image: BGR or grayscale numpy image array of the document page.
        region_bbox: [x, y, w, h] normalized COCO bounding box hint.

    Returns:
        Normalized bounding box [x, y, w, h] or None.
    """
    if page_image is None or page_image.size == 0 or len(region_bbox) != 4:
        return None

    h, w = page_image.shape[:2]
    x, y, bw, bh = region_bbox
    padding = 0.04

    px = max(0, int((x - padding) * w))
    py = max(0, int((y - padding) * h))
    pw = min(w - px, max(20, int((bw + 2 * padding) * w)))
    ph = min(h - py, max(15, int((bh + 2 * padding) * h)))

    crop = page_image[py : py + ph, px : px + pw]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    non_zero = cv2.findNonZero(binary)
    if non_zero is None or len(non_zero) < 30:
        return None

    x_r, y_r, w_r, h_r = cv2.boundingRect(non_zero)
    ar = w_r / max(h_r, 1)
    density = len(non_zero) / max(w_r * h_r, 1)

    if ar >= 1.4 and 0.03 <= density <= 0.45:
        nx = (px + x_r) / w
        ny = (py + y_r) / h
        nw = w_r / w
        nh = h_r / h
        return (round(nx, 6), round(ny, 6), round(nw, 6), round(nh, 6))

    return None


class VisualCheckboxProvider:
    """Provider for vision-based checkbox and signature bounding box detection."""

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi

    def detect_checkbox(
        self,
        page_image: np.ndarray,
        candidate_bbox: Sequence[float],
    ) -> tuple[str, tuple[float, float, float, float]] | None:
        """Detect checkbox bounding box within candidate region."""
        return detect_checkbox_with_grid_removal(page_image, candidate_bbox)

    def detect_signature(
        self,
        page_image: np.ndarray,
        candidate_bbox: Sequence[float],
    ) -> tuple[float, float, float, float] | None:
        """Detect signature bounding box within candidate region."""
        return detect_signature_region(page_image, candidate_bbox)
