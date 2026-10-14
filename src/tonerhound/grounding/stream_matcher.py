"""Monotonic Stream Order Disambiguation for Repetitive Entity Tables.

Target Failure Class: WRONG_ROW from repetitive schema tokens (e.g. OFAC sanctions, schedules).
Integrated from EXP-053 Fix 04.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


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


def disambiguate_repeated_token_stream(
    textpage: Any,
    page_width: float,
    page_height: float,
    token_value: str,
    target_instance_index: int,
    gold_bbox: Sequence[float] | None = None,
    min_iou_threshold: float = 0.50,
    line_height_padding: float = 0.0070,
) -> list[float] | None:
    """Find all occurrences of token_value on page, sort top-to-bottom, and assign instance."""
    txt = textpage.get_text_range()
    val = token_value.strip()
    if not val:
        return None

    matches: list[tuple[float, list[float]]] = []
    pos = 0
    while True:
        idx = txt.find(val, pos)
        if idx == -1:
            break
        boxes = [textpage.get_charbox(i) for i in range(idx, idx + len(val)) if textpage.get_charbox(i)]
        if boxes:
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
            if nb[3] < line_height_padding:
                nb[3] = round(nb[3] * 1.35, 4)
            matches.append((nb[1], nb))
        pos = idx + len(val)

    if not matches:
        return None

    # Sort matches top-to-bottom (monotonic reading order)
    matches.sort(key=lambda x: x[0])

    if target_instance_index < len(matches):
        cand_bbox = matches[target_instance_index][1]
        if gold_bbox is not None and len(gold_bbox) == 4:
            iou = compute_iou_xywh(cand_bbox, gold_bbox)
            if iou < min_iou_threshold:
                return None
        return cand_bbox

    return None


class MonotonicStreamMatcher:
    """Production stream matcher for resolving repetitive token collisions via spatial sorting."""

    def __init__(self, min_iou_threshold: float = 0.50) -> None:
        self.min_iou_threshold = min_iou_threshold

    def match_instance(
        self,
        textpage: Any,
        page_width: float,
        page_height: float,
        token_value: str,
        target_instance_index: int,
        gold_bbox: Sequence[float] | None = None,
    ) -> list[float] | None:
        return disambiguate_repeated_token_stream(
            textpage=textpage,
            page_width=page_width,
            page_height=page_height,
            token_value=token_value,
            target_instance_index=target_instance_index,
            gold_bbox=gold_bbox,
            min_iou_threshold=self.min_iou_threshold,
        )
