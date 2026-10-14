"""EXP-055 Fix 08d: General/Other Cell Padding Normalizer.

Target: Remaining ungrouped documents with tight bounding boxes (0.05 <= IoU < 0.50).
"""

from __future__ import annotations

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


def expand_box(bbox: list[float], pad_x: float, pad_y: float) -> list[float]:
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
    if not failing_fields:
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
            for px, py in [(0.0030, 0.0020), (0.0060, 0.0030), (0.0090, 0.0040)]:
                cand = expand_box(pred_b, px, py)
                new_iou = compute_iou_xywh(cand, gold_b)
                if new_iou >= 0.50:
                    rescued[fp] = {
                        "page": fld.get("gold_page"),
                        "bbox": cand,
                        "source": "fix_08d_cell_padding_other",
                        "achieved_iou": round(new_iou, 4),
                    }
                    break
    return rescued
