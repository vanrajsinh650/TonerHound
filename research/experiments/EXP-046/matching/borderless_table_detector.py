"""EXP-046 Stage 8: Borderless Table Detector (Whitespace Gap Projection).

Detects column and row boundaries in borderless tables using vertical and horizontal
whitespace projection analysis across page token distributions.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


class BorderlessTableDetector:
    """Infers table grid structure from token whitespace gaps."""

    def __init__(self, min_col_gap: float = 0.02, min_row_gap: float = 0.005) -> None:
        self.min_col_gap = min_col_gap
        self.min_row_gap = min_row_gap

    def detect_table_grid(
        self, page_obj: DocumentPage, region_bbox: BBox | None = None
    ) -> dict[str, Any]:
        """Detect column rails and row bands from page tokens."""
        tokens = page_obj.tokens
        if not tokens:
            return {"columns": [], "rows": [], "detected": False}

        if region_bbox:
            tokens = [
                t for t in tokens
                if region_bbox.x <= (t.bbox.x + t.bbox.width / 2) <= (region_bbox.x + region_bbox.width)
                and region_bbox.y <= (t.bbox.y + t.bbox.height / 2) <= (region_bbox.y + region_bbox.height)
            ]

        if not tokens:
            return {"columns": [], "rows": [], "detected": False}

        # 1. Project horizontal positions to find column rails
        xs = sorted([(t.bbox.x, t.bbox.x + t.bbox.width) for t in tokens], key=lambda x: x[0])
        col_splits = []
        max_r = xs[0][1]
        for l, r in xs[1:]:
            if l - max_r >= self.min_col_gap:
                col_splits.append((max_r + l) / 2.0)
            max_r = max(max_r, r)

        # 2. Cluster tokens into horizontal row bands by y-center
        ys = sorted([t.bbox.y + (t.bbox.height / 2.0) for t in tokens])
        row_bands: list[float] = []
        last_y = ys[0]
        curr_cluster = [last_y]
        for y in ys[1:]:
            if y - last_y >= self.min_row_gap:
                row_bands.append(float(np.mean(curr_cluster)))
                curr_cluster = [y]
            else:
                curr_cluster.append(y)
            last_y = y
        if curr_cluster:
            row_bands.append(float(np.mean(curr_cluster)))

        return {
            "columns": col_splits,
            "rows": row_bands,
            "detected": len(col_splits) >= 1 and len(row_bands) >= 3,
        }

    def match_cell_token(
        self, target_value: Any, page_obj: DocumentPage, row_idx: int, col_idx: int
    ) -> tuple[BBox | None, str | None]:
        """Attempt to locate token in specific (row, col) cell coordinate."""
        grid = self.detect_table_grid(page_obj)
        if not grid["detected"]:
            return None, None

        rows = grid["rows"]
        cols = grid["columns"]
        if row_idx < 0 or row_idx >= len(rows):
            return None, None

        target_y = rows[row_idx]
        target_str = str(target_value).strip().lower()

        # Find token near target_y
        cands = []
        for t in page_obj.tokens:
            cy = t.bbox.y + (t.bbox.height / 2.0)
            if abs(cy - target_y) <= 0.008:
                if target_str in t.text.strip().lower() or t.text.strip().lower() in target_str:
                    cands.append(t)

        if not cands:
            return None, None

        b = cands[0].bbox
        return BBox(b.x, b.y, b.width, b.height, page_obj.page_number), cands[0].text
