"""Multi-region assembler for spatially disconnected tokens and cross-column values.

Target: Solves NO_TEXT_AT_GOLD_REGION where values span across multiple disconnected
spatial regions or columns on a page.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import fitz


def normalize_for_match(text: str) -> str:
    """Normalize text for literal multi-region token matching."""
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
    valid = [b for b in bboxes if b and len(b) == 4 and b[2] > 0 and b[3] > 0]
    if not valid:
        return None
    min_x = min(b[0] for b in valid)
    min_y = min(b[1] for b in valid)
    max_x = max(b[0] + b[2] for b in valid)
    max_y = max(b[1] + b[3] for b in valid)
    w = max(0.0001, max_x - min_x)
    h = max(0.0001, max_y - min_y)
    return [round(min_x, 6), round(min_y, 6), round(w, 6), round(h, 6)]


class MultiRegionAssembler:
    """Clusters spatially disconnected tokens into unified region bounding boxes."""

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
                "text_norm": normalize_for_match(text),
                "bbox": norm_box,
                "x": norm_box[0],
                "y": norm_box[1],
            })

    def assemble_regions(
        self,
        value: str,
        max_regions: int = 5,
        max_gap_y: float = 0.05,
    ) -> list[list[float]]:
        """Cluster tokens matching value by vertical proximity and return region union bboxes."""
        val_norm = normalize_for_match(value)
        matching = [
            t for t in self.tokens
            if t["text_norm"] == val_norm or (len(t["text_norm"]) >= 3 and t["text_norm"] in val_norm)
        ]

        if not matching:
            return []

        matching.sort(key=lambda t: t["y"])
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = [matching[0]]

        for t in matching[1:]:
            prev_y = current[-1]["y"]
            if abs(t["y"] - prev_y) <= max_gap_y:
                current.append(t)
            else:
                clusters.append(current)
                current = [t]
        clusters.append(current)

        clusters = clusters[:max_regions]
        regions: list[list[float]] = []
        for cluster in clusters:
            bboxes = [t["bbox"] for t in cluster]
            u = union_bbox(bboxes)
            if u:
                regions.append(u)

        return regions

    def assemble_cross_column(
        self,
        value: str,
        max_regions: int = 5,
    ) -> list[list[float]]:
        """Cluster tokens by spatial proximity for values spanning across columns."""
        val_norm = normalize_for_match(value)
        if not val_norm:
            return []

        matching = [
            t for t in self.tokens
            if t["text_norm"] == val_norm or (len(t["text_norm"]) >= 3 and t["text_norm"] in val_norm)
        ]
        if not matching:
            return []

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

        regions: list[list[float]] = []
        for cluster in clusters[:max_regions]:
            bboxes = [t["bbox"] for t in cluster]
            u = union_bbox(bboxes)
            if u:
                regions.append(u)

        return regions
