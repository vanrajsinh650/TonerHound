"""Stage 18: Cell Boundary Clamping v2.

When a token or multi-token bbox spans multiple cells:
- Detect cell boundaries from table structure / cell bounding boxes
- Clamp token bbox to containing cell
- If candidate value is contained in exactly one cell, clamp to cell or token inside cell
- If value spans across two or more disjoint cells, reject (multi-cell merge)
"""

from __future__ import annotations

from typing import Sequence
from tonerhound.geometry.coordinates import BBox


def clamp_bbox_to_cells(
    token_bbox: BBox | tuple[float, float, float, float] | list[float],
    cell_bboxes: Sequence[BBox | tuple[float, float, float, float] | list[float]],
) -> tuple[BBox | None, bool]:
    """Clamps token_bbox to the matching cell.
    
    Returns:
        (clamped_bbox, is_valid)
        If token spans across two or more cells, returns (None, False).
    """
    if isinstance(token_bbox, (list, tuple)):
        tb = BBox(token_bbox[0], token_bbox[1], token_bbox[2], token_bbox[3])
    else:
        tb = token_bbox

    intersecting_cells: list[BBox] = []

    for cb in cell_bboxes:
        if isinstance(cb, (list, tuple)):
            cell_b = BBox(cb[0], cb[1], cb[2], cb[3])
        else:
            cell_b = cb

        # Check intersection
        ix0 = max(tb.x0, cell_b.x0)
        iy0 = max(tb.y0, cell_b.y0)
        ix1 = min(tb.x1, cell_b.x1)
        iy1 = min(tb.y1, cell_b.y1)

        if ix1 > ix0 and iy1 > iy0:
            inter_area = (ix1 - ix0) * (iy1 - iy0)
            token_area = tb.width * tb.height
            if token_area > 0 and (inter_area / token_area) >= 0.15:
                intersecting_cells.append(cell_b)

    if not intersecting_cells:
        # No cell intersection found, return original bbox unmodified
        return tb, True

    if len(intersecting_cells) == 1:
        c = intersecting_cells[0]
        # Clamp token bbox within cell boundaries
        nx0 = max(tb.x0, c.x0)
        ny0 = max(tb.y0, c.y0)
        nx1 = min(tb.x1, c.x1)
        ny1 = min(tb.y1, c.y1)
        return BBox.from_xyxy(nx0, ny0, nx1, ny1), True

    # Spans across multiple cells -> rejected
    return None, False
