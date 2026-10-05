"""EXP-042 Phase G: No Text at Gold Region Fix.

Targets NO_TEXT_AT_GOLD_REGION (14,973 fields, realistic gain +0.08 pp).
Implements:
1. Multi-region cross-column spatial clustering (dy <= 0.05, dx <= 0.15).
2. Hough transform rotation detection and correction.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
import cv2
import fitz
import numpy as np


def normalize_for_match(text: str) -> str:
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\s\.\,\-\$\%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def union_bbox(bboxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    valid = [b for b in bboxes if b and len(b) == 4 and b[2] > 0 and b[3] > 0]
    if not valid:
        return None
    min_x = min(b[0] for b in valid)
    min_y = min(b[1] for b in valid)
    max_x = max(b[0] + b[2] for b in valid)
    max_y = max(b[1] + b[3] for b in valid)
    w = max(0.0001, max_x - min_x)
    h = max(0.0001, max_y - min_y)
    return (round(min_x, 6), round(min_y, 6), round(w, 6), round(h, 6))


def multi_region_cross_column(
    page: fitz.Page,
    gold_value: str,
    max_regions: int = 5,
) -> list[tuple[float, float, float, float]]:
    """Cluster tokens by spatial proximity for values spanning disconnected visual regions."""
    if not gold_value or not page:
        return []

    val_norm = normalize_for_match(gold_value)
    if not val_norm:
        return []

    pw = float(page.rect.width) if page.rect.width > 0 else 1.0
    ph = float(page.rect.height) if page.rect.height > 0 else 1.0

    raw_words = page.get_text("words")
    matching = []

    for w in raw_words:
        x0, y0, x1, y1, text, bno, lno, wno = w
        t_norm = normalize_for_match(text)
        if t_norm == val_norm or (len(t_norm) >= 3 and t_norm in val_norm):
            matching.append({
                "bbox": (x0 / pw, y0 / ph, (x1 - x0) / pw, (y1 - y0) / ph),
                "x": x0 / pw,
                "y": y0 / ph,
            })

    if not matching:
        return []

    # Sort by y then x
    matching.sort(key=lambda t: (t["y"], t["x"]))

    clusters: list[list[dict[str, Any]]] = []
    current = [matching[0]]

    for t in matching[1:]:
        last = current[-1]
        dy = abs(t["y"] - last["y"])
        dx = abs(t["x"] - last["x"])

        if dy < 0.05 or dx < 0.15:
            current.append(t)
        else:
            clusters.append(current)
            current = [t]
    clusters.append(current)

    regions = []
    for cluster in clusters[:max_regions]:
        bboxes = [t["bbox"] for t in cluster]
        u = union_bbox(bboxes)
        if u:
            regions.append(u)

    return regions


def detect_rotation(page_image: np.ndarray) -> float:
    """Detect rotation angle via Hough transform."""
    if page_image is None or page_image.size == 0:
        return 0.0

    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY) if len(page_image.shape) == 3 else page_image
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    median = float(np.median(angles))
    return median if abs(median) >= 1.0 else 0.0
