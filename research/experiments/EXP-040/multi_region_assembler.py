"""EXP-040 Phase D: Multi-Region Assembly.

Targets NO_TEXT_AT_GOLD_REGION (largest subclass: SUB_MULTI_REGION, 15,382 fields).
Assembles multi-part or spatially disconnected tokens on a document page by clustering
by spatial proximity and generating unified candidate bounding boxes.
"""

from __future__ import annotations

import fitz
from typing import Any, Sequence


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

    w = max(0.001, max_x - min_x)
    h = max(0.001, max_y - min_y)
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
                "bbox": norm_box,
                "y": norm_box[1],
                "x": norm_box[0],
            })

    def assemble_regions(
        self,
        value: str,
        max_regions: int = 5,
        max_gap_y: float = 0.05,
    ) -> list[list[float]]:
        """Cluster tokens matching value by vertical proximity and return region union bboxes."""
        val_norm = value.lower().strip()
        matching = [t for t in self.tokens if t["text_lower"] == val_norm or val_norm in t["text_lower"]]

        if not matching:
            return []

        # Cluster by y-coordinate
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

        # Limit to max_regions
        clusters = clusters[:max_regions]

        regions = []
        for cluster in clusters:
            bboxes = [t["bbox"] for t in cluster]
            u = union_bbox(bboxes)
            if u:
                regions.append(u)

        return regions
