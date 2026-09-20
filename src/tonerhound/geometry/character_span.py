"""Safe Character-Span Reconstruction (EXP-015).

Resolves sub-token character spans when an emitted token contains affixes, punctuation,
or footnote markers, adjusting both horizontal offset (X) and width (W) proportionally.
"""

from __future__ import annotations

from typing import Any

from tonerhound.geometry.coordinates import BBox


def reconstruct_safe_character_span(
    pred_box: tuple[float, float, float, float] | BBox,
    raw_token_text: str,
    target_value: Any,
    confidence: float = 0.95,
) -> tuple[float, float, float, float] | BBox:
    """Safe Character-Span Resolver implementing EXP-015 Phase 2 safety conditions."""
    if confidence < 0.80 or not raw_token_text or not target_value:
        return pred_box

    is_bbox_obj = isinstance(pred_box, BBox)
    if is_bbox_obj:
        px, py, pw, ph = pred_box.x, pred_box.y, pred_box.width, pred_box.height
        page = pred_box.page
    else:
        px, py, pw, ph = pred_box[:4]
        page = None

    target_str = str(target_value)
    target_clean = target_str.strip(" -.,;:_()[]{}/'\"")
    pt_clean = raw_token_text.strip(" -.,;:_()[]{}/'\"")

    if not target_clean or not pt_clean:
        return pred_box

    # Substring search
    idx = raw_token_text.lower().find(target_clean.lower())
    if idx == -1:
        idx = pt_clean.lower().find(target_clean.lower())
        if idx != -1:
            raw_text = pt_clean
        else:
            return pred_box
    else:
        raw_text = raw_token_text

    total_len = len(raw_text)
    target_len = len(target_clean)

    # Safety Guard 1: High risk of table dot leaders / row concatenation
    if total_len > 3 * target_len and ("..." in raw_text or "EUR" in raw_text or len(raw_text.split()) > 10):
        return pred_box

    # Safety Guard 2: Almost whole string (>= 90%) requires no trim
    if target_len / max(1, total_len) >= 0.90:
        return pred_box

    start_frac = idx / float(total_len)
    span_frac = target_len / float(total_len)

    new_x = px + pw * start_frac
    new_w = pw * span_frac

    # Clamp bounds
    new_w = max(0.002, min(new_w, 0.98 - new_x))

    if is_bbox_obj:
        return BBox(x=new_x, y=py, width=new_w, height=ph, page=page)
    return (new_x, py, new_w, ph)
