"""Stage 21: Merged Cell Splitter.

When find_tables() or layout heuristics return a cell spanning multiple columns:
- Detect expected column widths from the table header row
- Split the merged cell horizontally proportional to the header widths
- Assign each resulting sub-cell its own bounding box.
"""

from __future__ import annotations

from typing import Sequence
from tonerhound.geometry.coordinates import BBox


def split_merged_cell(
    merged_cell_bbox: BBox | tuple[float, float, float, float] | list[float],
    header_bboxes: Sequence[BBox | tuple[float, float, float, float] | list[float]],
) -> list[BBox]:
    """Splits a horizontally merged cell across intersecting header columns."""
    if isinstance(merged_cell_bbox, (list, tuple)):
        mb = BBox(merged_cell_bbox[0], merged_cell_bbox[1], merged_cell_bbox[2], merged_cell_bbox[3])
    else:
        mb = merged_cell_bbox

    if not header_bboxes:
        return [mb]

    norm_headers: list[BBox] = []
    for hb in header_bboxes:
        if isinstance(hb, (list, tuple)):
            norm_headers.append(BBox(hb[0], hb[1], hb[2], hb[3]))
        else:
            norm_headers.append(hb)

    # Sort headers by x0
    norm_headers.sort(key=lambda b: b.x0)

    # Find headers overlapping merged cell's horizontal span
    overlapping_headers: list[BBox] = []
    for h in norm_headers:
        overlap_w = min(mb.x1, h.x1) - max(mb.x0, h.x0)
        if overlap_w > 0 and (overlap_w / max(1.0, h.width)) > 0.2:
            overlapping_headers.append(h)

    if len(overlapping_headers) <= 1:
        return [mb]

    sub_cells: list[BBox] = []
    total_header_w = sum(h.width for h in overlapping_headers)
    if total_header_w <= 0:
        return [mb]

    curr_x0 = mb.x0
    total_cell_w = mb.width

    for i, h in enumerate(overlapping_headers):
        ratio = h.width / total_header_w
        if i == len(overlapping_headers) - 1:
            cell_x1 = mb.x1
        else:
            cell_x1 = curr_x0 + (total_cell_w * ratio)

        sub_cells.append(BBox.from_xyxy(curr_x0, mb.y0, cell_x1, mb.y1))
        curr_x0 = cell_x1

    return sub_cells
