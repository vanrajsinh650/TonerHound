"""Stage 19: Column Rail Detector.

Detect column left/right rails from header row or vertical column distributions:
- Column i starts at header[i].x0 and ends at header[i].x1 (or before header[i+1].x0)
- Constrains candidate x-range by assigned column index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from tonerhound.geometry.coordinates import BBox


@dataclass
class ColumnRail:
    col_idx: int
    x_left: float
    x_right: float
    header_text: str = ""

    def contains_x(self, x: float, margin: float = 5.0) -> bool:
        return (self.x_left - margin) <= x <= (self.x_right + margin)

    def clamp_bbox(self, bbox: BBox) -> BBox:
        clamped_x0 = max(bbox.x0, self.x_left)
        clamped_x1 = min(bbox.x1, self.x_right)
        if clamped_x1 <= clamped_x0:
            return bbox
        return BBox.from_xyxy(clamped_x0, bbox.y0, clamped_x1, bbox.y1)


def detect_column_rails(
    header_cells: Sequence[tuple[float, float, float, float] | BBox | dict],
) -> list[ColumnRail]:
    """Detects column rails from ordered header cells."""
    if not header_cells:
        return []

    cells_norm: list[tuple[float, float, str]] = []
    for c in header_cells:
        if isinstance(c, (list, tuple)) and len(c) == 4:
            cells_norm.append((c[0], c[0] + c[2], ""))
        elif isinstance(c, BBox):
            cells_norm.append((c.x0, c.x1, ""))
        elif isinstance(c, dict):
            bb = c.get("bbox") or c.get("bounding_box") or [0, 0, 0, 0]
            txt = c.get("text", "")
            if len(bb) == 4:
                cells_norm.append((bb[0], bb[0] + bb[2], txt))

    # Sort by left x
    cells_norm.sort(key=lambda item: item[0])

    rails: list[ColumnRail] = []
    num_cells = len(cells_norm)

    for i, (x0, x1, txt) in enumerate(cells_norm):
        # The right rail boundary extends until the start of next column if there's whitespace
        if i + 1 < num_cells:
            next_x0 = cells_norm[i + 1][0]
            right_boundary = max(x1, next_x0)
        else:
            right_boundary = x1 + 100.0  # open right margin

        rails.append(
            ColumnRail(
                col_idx=i,
                x_left=x0,
                x_right=right_boundary,
                header_text=txt,
            )
        )

    return rails
