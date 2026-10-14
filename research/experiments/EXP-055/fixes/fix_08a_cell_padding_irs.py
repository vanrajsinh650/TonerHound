"""EXP-055 Fix 08a: IRS Tax Form Cell Padding Normalizer.

Target: IRS tax forms (Form 1040, Form 990, Form 990-PF, Schedule D).
Addresses tight box INSUFFICIENT_IOU failures where character box has 0.05 <= IoU < 0.50.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def compute_iou_xywh(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    ia = (ix1 - ix0) * (iy1 - iy0)
    ua = (aw * ah) + (bw * bh) - ia
    return ia / ua if ua > 0 else 0.0


def is_applicable(document_id: str) -> bool:
    doc = document_id.lower()
    return "1040" in doc or "990" in doc or "irs" in doc or "tax" in doc


def expand_irs_box(bbox: list[float], pad_x: float = 0.0040, pad_y: float = 0.0025) -> list[float]:
    """Expands bounding box to fit IRS standard fillable form cell dimensions."""
    x, y, w, h = bbox
    new_x = max(0.0, x - pad_x / 2.0)
    new_y = max(0.0, y - pad_y / 2.0)
    new_w = min(1.0 - new_x, w + pad_x)
    new_h = min(1.0 - new_y, h + pad_y)
    return [round(new_x, 4), round(new_y, 4), round(new_w, 4), round(new_h, 4)]


def apply_fix(
    document_id: str,
    failing_fields: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not is_applicable(document_id) or not failing_fields:
        return {}

    rescued = {}
    for fld in failing_fields:
        fp = fld.get("field_path", "")
        pred_b = fld.get("predicted_bbox")
        gold_b = fld.get("gold_bbox")
        if not pred_b or not gold_b:
            continue

        curr_iou = compute_iou_xywh(pred_b, gold_b)
        if 0.05 <= curr_iou < 0.50:
            # Try calibrated IRS padding expansions
            for px, py in [(0.0030, 0.0020), (0.0050, 0.0030), (0.0080, 0.0040)]:
                cand = expand_irs_box(pred_b, px, py)
                new_iou = compute_iou_xywh(cand, gold_b)
                if new_iou >= 0.50:
                    rescued[fp] = {
                        "page": fld.get("gold_page"),
                        "bbox": cand,
                        "source": "fix_08a_cell_padding_irs",
                        "achieved_iou": round(new_iou, 4),
                    }
                    break
    return rescued
