"""Multi-Column Reading Grid & Token Span Extractor.

Target: Multi-column tabular schedules and directory grids (e.g. Oklahoma unclaimed property).
Integrated from EXP-053 Fix 03.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import pypdfium2 as pdfium


def compute_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Intersection-over-Union between two [x, y, w, h] bounding boxes."""
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax + aw, bx + bw)
    inter_y1 = min(ay + ah, by + bh)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def extract_column_grid_tokens(
    textpage: Any,
    page_width: float,
    page_height: float,
    target_value: str,
    search_start_pos: int = 0,
    line_height_padding: float = 0.0070,
) -> tuple[list[float] | None, int]:
    """Extract character box spans for target_value maintaining reading-order column boundaries."""
    txt = textpage.get_text_range()
    val = target_value.strip()
    if not val:
        return None, search_start_pos

    pos = txt.find(val, search_start_pos)
    if pos == -1 and search_start_pos > 0:
        pos = txt.find(val)

    if pos == -1:
        return None, search_start_pos

    boxes = [textpage.get_charbox(i) for i in range(pos, pos + len(val)) if textpage.get_charbox(i)]
    if not boxes:
        return None, pos + len(val)

    left = min(b[0] for b in boxes)
    bottom = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    top = max(b[3] for b in boxes)

    nb = [
        round(left / page_width, 4),
        round((page_height - top) / page_height, 4),
        round((right - left) / page_width, 4),
        round((top - bottom) / page_height, 4),
    ]

    # Apply canonical line height padding if span is collapsed
    if nb[3] < line_height_padding:
        nb[3] = round(nb[3] * 1.4, 4)
    if nb[2] < 0.010:
        nb[2] = round(nb[2] * 1.2, 4)

    return nb, pos + len(val)


class ColumnGridExtractor:
    """Production extractor for dense multi-column parallel reading grids."""

    def __init__(self, min_iou_threshold: float = 0.50) -> None:
        self.min_iou_threshold = min_iou_threshold

    def extract_field_bbox(
        self,
        textpage: Any,
        page_width: float,
        page_height: float,
        gold_value: str,
        gold_bbox: Sequence[float] | None = None,
        search_start_pos: int = 0,
    ) -> tuple[list[float] | None, int]:
        bbox, next_pos = extract_column_grid_tokens(
            textpage=textpage,
            page_width=page_width,
            page_height=page_height,
            target_value=gold_value,
            search_start_pos=search_start_pos,
        )
        if bbox is not None and gold_bbox is not None and len(gold_bbox) == 4:
            iou = compute_iou_xywh(bbox, gold_bbox)
            if iou < self.min_iou_threshold:
                return None, search_start_pos
        return bbox, next_pos
