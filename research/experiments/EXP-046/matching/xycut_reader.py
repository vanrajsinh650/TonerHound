"""EXP-046 Stage 11: Multi-Column XY-Cut Reader.

Recursive XY-Cut algorithm for detecting column partitions and restoring
proper reading order (column 1 top-to-bottom, then column 2 top-to-bottom).
"""

from __future__ import annotations

from typing import Any, Sequence
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


class XYCutReader:
    """Applies XY-Cut decomposition to partition pages into column reading streams."""

    def __init__(self, min_gap_x: float = 0.03, min_gap_y: float = 0.015) -> None:
        self.min_gap_x = min_gap_x
        self.min_gap_y = min_gap_y

    def partition_columns(self, tokens: Sequence[DocumentToken]) -> list[list[DocumentToken]]:
        """Split tokens into distinct vertical columns by finding significant x-whitespace gaps."""
        if not tokens:
            return []

        # Find x-intervals
        intervals = sorted([(t.bbox.x, t.bbox.x + t.bbox.width) for t in tokens], key=lambda x: x[0])
        gaps = []
        max_r = intervals[0][1]
        for l, r in intervals[1:]:
            if l - max_r >= self.min_gap_x:
                gaps.append((max_r, l))
            max_r = max(max_r, r)

        if not gaps:
            # Single column: sort by y, then x
            return [sorted(tokens, key=lambda t: (t.bbox.y, t.bbox.x))]

        # Choose the widest primary column split
        best_gap = max(gaps, key=lambda g: g[1] - g[0])
        split_x = (best_gap[0] + best_gap[1]) / 2.0

        col_left = [t for t in tokens if (t.bbox.x + t.bbox.width / 2.0) < split_x]
        col_right = [t for t in tokens if (t.bbox.x + t.bbox.width / 2.0) >= split_x]

        sorted_left = sorted(col_left, key=lambda t: (t.bbox.y, t.bbox.x))
        sorted_right = sorted(col_right, key=lambda t: (t.bbox.y, t.bbox.x))

        return [sorted_left, sorted_right]

    def read_tokens_in_xycut_order(self, tokens: Sequence[DocumentToken]) -> list[DocumentToken]:
        """Produce flattened token list ordered column-by-column."""
        cols = self.partition_columns(tokens)
        ordered: list[DocumentToken] = []
        for c in cols:
            ordered.extend(c)
        return ordered

    def match_in_xycut_stream(
        self, target_value: Any, page_obj: DocumentPage
    ) -> tuple[BBox | None, str | None]:
        """Search in column-partitioned stream."""
        target_str = str(target_value).strip().lower()
        if not target_str:
            return None, None

        ordered_tokens = self.read_tokens_in_xycut_order(page_obj.tokens)
        for t in ordered_tokens:
            t_text = t.text.strip().lower()
            if target_str == t_text or (len(target_str) > 3 and target_str in t_text):
                b = t.bbox
                return BBox(b.x, b.y, b.width, b.height, page_obj.page_number), t.text
        return None, None
