"""EXP-043 Phase D: Recursive XY-Cut + Morphological Table Cells.

Target: TOKEN_SLICING (23,542 fields).
Performs recursive XY-Cut projection profile decomposition and morphological line opening
to detect discrete table cell geometry and clamp overlapping predictions to true cell boundaries.
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def max_contiguous_gap(indices: np.ndarray) -> int:
    """Find maximum contiguous run of indices."""
    if len(indices) == 0:
        return 0
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return max((len(g) for g in groups), default=0)


def find_cut_center(indices: np.ndarray) -> int:
    """Find midpoint of the longest contiguous valley."""
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    longest = max(groups, key=len)
    return int(np.mean(longest))


def recursive_xy_cut(
    binary_image: np.ndarray,
    min_gap: int = 10,
    depth: int = 0,
    max_depth: int = 6,
) -> list[tuple[int, int, int, int]]:
    """Recursive XY-Cut page segmentation."""
    h, w = binary_image.shape
    if depth >= max_depth or h < 20 or w < 20:
        return [(0, 0, w, h)]

    row_proj = np.sum(binary_image, axis=1)
    col_proj = np.sum(binary_image, axis=0)

    h_valleys = np.where(row_proj < min_gap)[0]
    v_valleys = np.where(col_proj < min_gap)[0]

    if len(h_valleys) == 0 and len(v_valleys) == 0:
        return [(0, 0, w, h)]

    h_gap = max_contiguous_gap(h_valleys) if len(h_valleys) > 0 else 0
    v_gap = max_contiguous_gap(v_valleys) if len(v_valleys) > 0 else 0

    if h_gap >= v_gap and h_gap > 20:
        cut = find_cut_center(h_valleys)
        if 10 < cut < h - 10:
            top = recursive_xy_cut(binary_image[:cut, :], min_gap, depth + 1, max_depth)
            bottom = recursive_xy_cut(binary_image[cut:, :], min_gap, depth + 1, max_depth)
            return [(x, y, cw, ch) for (x, y, cw, ch) in top] + [
                (x, y + cut, cw, ch) for (x, y, cw, ch) in bottom
            ]
    elif v_gap > 20:
        cut = find_cut_center(v_valleys)
        if 10 < cut < w - 10:
            left = recursive_xy_cut(binary_image[:, :cut], min_gap, depth + 1, max_depth)
            right = recursive_xy_cut(binary_image[:, cut:], min_gap, depth + 1, max_depth)
            return [(x, y, cw, ch) for (x, y, cw, ch) in left] + [
                (x + cut, y, cw, ch) for (x, y, cw, ch) in right
            ]

    return [(0, 0, w, h)]


def extract_table_cells_morphological(page_image: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Extract table cells via morphological line detection."""
    if page_image is None or page_image.size == 0:
        return []

    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY) if len(page_image.shape) == 3 else page_image
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Horizontal lines (kernel 25x1)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Vertical lines (kernel 1x25)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Combine
    frame = cv2.bitwise_or(h_lines, v_lines)

    # Find connected cells
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    cells: list[tuple[float, float, float, float]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 20 or ch < 10:
            continue
        cells.append((
            round(x / w, 6),
            round(y / h, 6),
            round(cw / w, 6),
            round(ch / h, 6),
        ))

    return cells
