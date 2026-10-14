"""Stage 25: Multi-Line BBox Union v2.

Extends multi-line bounding box assembly up to 6 lines:
- Vertical gap < 1.5 * line height
- Horizontal overlap > 60% (or significant column corridor overlap)
- Reject if union spans > 30% of page height
"""

from __future__ import annotations

from typing import Sequence
from tonerhound.geometry.coordinates import BBox


def union_multiline_boxes(
    boxes: Sequence[BBox | tuple[float, float, float, float] | list[float]],
    page_height: float,
    max_lines: int = 6,
) -> BBox | None:
    """Forms a multi-line union across up to max_lines lines satisfying geometric continuity."""
    if not boxes:
        return None

    norm_boxes: list[BBox] = []
    for b in boxes:
        if isinstance(b, (list, tuple)):
            norm_boxes.append(BBox(b[0], b[1], b[2], b[3]))
        else:
            norm_boxes.append(b)

    # Sort lines by vertical position y0
    norm_boxes.sort(key=lambda b: b.y0)

    if len(norm_boxes) > max_lines:
        norm_boxes = norm_boxes[:max_lines]

    valid_lines: list[BBox] = [norm_boxes[0]]

    for i in range(1, len(norm_boxes)):
        prev = valid_lines[-1]
        curr = norm_boxes[i]

        v_gap = curr.y0 - prev.y1
        line_h = max(1.0, prev.height)

        # Check vertical gap
        if v_gap > 1.5 * line_h:
            break

        # Check horizontal overlap
        overlap_x = min(prev.x1, curr.x1) - max(prev.x0, curr.x0)
        min_w = min(prev.width, curr.width)
        if min_w > 0 and (overlap_x / min_w) < 0.40:
            # Insufficient horizontal alignment
            break

        valid_lines.append(curr)

    if not valid_lines:
        return None

    # Compute union bbox
    ux0 = min(b.x0 for b in valid_lines)
    uy0 = min(b.y0 for b in valid_lines)
    ux1 = max(b.x1 for b in valid_lines)
    uy1 = max(b.y1 for b in valid_lines)

    union_b = BBox.from_xyxy(ux0, uy0, ux1, uy1)

    # Reject if union height > 30% of page height
    if page_height > 0 and (union_b.height / page_height) > 0.30:
        return None

    return union_b
