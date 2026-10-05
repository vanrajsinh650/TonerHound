"""EXP-041 Phase D: Multi-Region Assembler V2.

Target class: NO_TEXT_AT_GOLD_REGION / SUB_MULTI_REGION (14,983 fields).
Realistic gain: +0.2–0.3 pp.

Clusters spatially disconnected tokens into unified region bounding boxes supporting both
vertical (multi-line, dy <= 0.05) and horizontal (cross-column, dx <= 0.15) proximity,
with Hough transform rotation detection and correction.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
import cv2
import fitz
import numpy as np


def normalize_for_match(text: str) -> str:
    """Normalize text for robust token clustering."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\s\.\,\-\$\%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def union_bbox(bboxes: Sequence[Sequence[float]]) -> list[float] | None:
    """Compute the union bounding box [x, y, w, h] in normalized COCO coordinates."""
    if not bboxes:
        return None
    valid_boxes = [b for b in bboxes if b and len(b) == 4 and b[2] > 0 and b[3] > 0]
    if not valid_boxes:
        return None

    min_x = min(b[0] for b in valid_boxes)
    min_y = min(b[1] for b in valid_boxes)
    max_x = max(b[0] + b[2] for b in valid_boxes)
    max_y = max(b[1] + b[3] for b in valid_boxes)

    w = max(0.0001, max_x - min_x)
    h = max(0.0001, max_y - min_y)
    return [round(min_x, 6), round(min_y, 6), round(w, 6), round(h, 6)]


def detect_and_correct_rotation(page_image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect page rotation via Hough transform.
    Returns (corrected_image, angle_in_degrees).
    """
    if page_image is None or page_image.size == 0:
        return page_image, 0.0

    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY) if len(page_image.shape) == 3 else page_image
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return page_image, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return page_image, 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 1.0:
        return page_image, 0.0

    h, w = page_image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    corrected = cv2.warpAffine(page_image, M, (w, h), borderValue=(255, 255, 255))
    return corrected, median_angle


class MultiRegionAssemblerV2:
    """Assembles multi-part or multi-region tokens into unified region boxes."""

    def __init__(self, page: fitz.Page) -> None:
        self.page = page
        self.pw = float(page.rect.width) if page.rect.width > 0 else 1.0
        self.ph = float(page.rect.height) if page.rect.height > 0 else 1.0

        raw_words = page.get_text("words")
        self.tokens: list[dict[str, Any]] = []

        for w in raw_words:
            x0, y0, x1, y1, text, bno, lno, wno = w
            norm_box = [
                max(0.0, min(1.0, x0 / self.pw)),
                max(0.0, min(1.0, y0 / self.ph)),
                max(0.0, min(1.0, (x1 - x0) / self.pw)),
                max(0.0, min(1.0, (y1 - y0) / self.ph)),
            ]
            self.tokens.append({
                "text": text,
                "text_lower": text.lower().strip(),
                "bbox": norm_box,
                "x": norm_box[0],
                "y": norm_box[1],
            })

    def assemble_regions(
        self,
        value: str,
        max_regions: int = 5,
        max_gap_y: float = 0.05,
        max_gap_x: float = 0.15,
    ) -> list[list[float]]:
        """
        Cluster tokens into regions by spatial proximity.
        Supports both vertical (multi-line) and horizontal (multi-column) spans.
        """
        if not value or not self.tokens:
            return []

        val_norm = normalize_for_match(value)
        if not val_norm:
            return []

        matching = [
            t for t in self.tokens
            if val_norm == normalize_for_match(t["text"]) or val_norm in normalize_for_match(t["text"])
        ]

        if not matching:
            return []

        # Sort by y then x
        matching.sort(key=lambda t: (t["y"], t["x"]))

        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = [matching[0]]

        for t in matching[1:]:
            last = current[-1]
            dy = abs(t["y"] - last["y"])
            dx = abs(t["x"] - last["x"])

            if dy <= max_gap_y or dx <= max_gap_x:
                current.append(t)
            else:
                clusters.append(current)
                current = [t]
        clusters.append(current)

        clusters = clusters[:max_regions]

        regions = []
        for cluster in clusters:
            bboxes = [t["bbox"] for t in cluster]
            u = union_bbox(bboxes)
            if u:
                regions.append(u)

        return regions
