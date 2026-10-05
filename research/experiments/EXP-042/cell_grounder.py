"""EXP-042 Phase F: Token Slicing Fix.

Targets TOKEN_SLICING (23,542 fields, realistic gain +0.40 pp).
Implements:
1. Exact table cell text bounding (PyMuPDF find_tables strategy="lines_strict").
2. Vertical column rail clamping using vector drawing lines.
"""

from __future__ import annotations

from typing import Any, Sequence
import fitz


def ground_field_in_cell(
    page: fitz.Page,
    gold_value: str,
    gold_bbox: Sequence[float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Find the exact table cell containing the gold value."""
    if not gold_value or not page:
        return None

    val_str = str(gold_value).strip()
    if not val_str:
        return None

    try:
        tables = page.find_tables(strategy="lines_strict")
    except Exception:
        return None

    if not tables or not tables.tables:
        return None

    pw = float(page.rect.width) if page.rect.width > 0 else 1.0
    ph = float(page.rect.height) if page.rect.height > 0 else 1.0

    candidates = []

    for tab in tables.tables:
        for cell_rect in tab.cells:
            # cell_rect: (x0, y0, x1, y1)
            rect_obj = fitz.Rect(cell_rect)
            cell_text = page.get_text("text", clip=rect_obj).strip()
            if not cell_text:
                continue

            cx0, cy0, cx1, cy1 = cell_rect
            norm_box = (cx0 / pw, cy0 / ph, (cx1 - cx0) / pw, (cy1 - cy0) / ph)

            if cell_text == val_str:
                candidates.append((norm_box, 1.0))
            elif val_str in cell_text:
                candidates.append((norm_box, 0.8))

    if not candidates:
        return None

    if len(candidates) == 1:
        b = candidates[0][0]
        return (round(b[0], 6), round(b[1], 6), round(b[2], 6), round(b[3], 6))

    if gold_bbox and len(gold_bbox) == 4:
        gx, gy = gold_bbox[0], gold_bbox[1]
        best = min(candidates, key=lambda c: (c[0][0] - gx) ** 2 + (c[0][1] - gy) ** 2)
        b = best[0]
        return (round(b[0], 6), round(b[1], 6), round(b[2], 6), round(b[3], 6))

    best = max(candidates, key=lambda c: c[1])
    b = best[0]
    return (round(b[0], 6), round(b[1], 6), round(b[2], 6), round(b[3], 6))


def clamp_to_column_rail(
    pred_bbox: Sequence[float],
    page: fitz.Page,
) -> tuple[float, float, float, float] | None:
    """Clamp a predicted bbox to the nearest vertical column rails from vector drawings."""
    if not pred_bbox or len(pred_bbox) != 4 or not page:
        return None

    try:
        drawings = page.get_drawings()
    except Exception:
        return None

    if not drawings:
        return None

    v_lines = []
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":  # line
                p1, p2 = item[1], item[2]
                if abs(p1.x - p2.x) < 2.0:  # vertical line
                    v_lines.append(p1.x)

    if not v_lines:
        return None

    v_lines = sorted(set(v_lines))
    x, y, w, h = pred_bbox
    pw = float(page.rect.width) if page.rect.width > 0 else 1.0

    cx_pdf = (x + w / 2.0) * pw

    lefts = [v for v in v_lines if v <= cx_pdf]
    rights = [v for v in v_lines if v >= cx_pdf]

    if not lefts or not rights:
        return None

    left = max(lefts)
    right = min(rights)

    if right - left < 5.0 or (right - left) / pw > 0.8:
        return None

    new_x = left / pw
    new_w = (right - left) / pw

    return (round(new_x, 6), round(y, 6), round(new_w, 6), round(h, 6))
