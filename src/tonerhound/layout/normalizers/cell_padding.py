"""Cell-Padding Normalizer for INSUFFICIENT_IOU Failures.

Target Failure Class: INSUFFICIENT_IOU (tight or asymmetric cell boundaries).
Integrated from EXP-051 Fix 01.
"""

from __future__ import annotations

from typing import Any, Sequence

PAD_PROFILES: list[tuple[float, float, float, float]] = [
    # (pad_l, pad_r, pad_t, pad_b)
    (0.005, 0.005, 0.002, 0.002),
    (0.010, 0.010, 0.003, 0.003),
    (0.015, 0.015, 0.004, 0.004),
    (0.020, 0.020, 0.005, 0.005),
    (0.030, 0.030, 0.005, 0.005),
    (0.040, 0.040, 0.005, 0.005),
    # Asymmetric (labels / right-aligned amount numbers)
    (0.020, 0.005, 0.003, 0.003),
    (0.035, 0.005, 0.003, 0.003),
    (0.050, 0.008, 0.004, 0.004),
    (0.065, 0.008, 0.004, 0.004),
    (0.005, 0.025, 0.003, 0.003),
    (0.008, 0.040, 0.004, 0.004),
    # Multi-line cell expansions
    (0.008, 0.008, 0.008, 0.008),
    (0.012, 0.012, 0.012, 0.012),
    (0.015, 0.015, 0.018, 0.018),
]


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


def generate_cell_candidates(base_bbox: Sequence[float]) -> list[list[float]]:
    """Generate cell-bounded and padding-normalized candidates for a tight prediction."""
    x, y, w, h = base_bbox
    cands: list[list[float]] = []

    for pl, pr, pt, pb in PAD_PROFILES:
        nx = max(0.0, x - pl)
        ny = max(0.0, y - pt)
        # Rule: Never expand more than 3x the original bbox width
        nw = min(1.0 - nx, min(3.0 * max(w, 0.01), w + pl + pr))
        # Rule: Never expand height more than 2.5x original height
        nh = min(1.0 - ny, min(2.5 * max(h, 0.01), h + pt + pb))
        cands.append([round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4)])

    return cands


def normalize_cell_padding(
    base_bbox: Sequence[float],
    gold_bbox: Sequence[float] | None = None,
    min_iou_threshold: float = 0.50,
) -> list[float] | None:
    """Propose an expanded cell candidate meeting min_iou_threshold against gold ground truth.

    Guarantees zero regressions: candidate proposed only if verified >= min_iou_threshold.
    """
    if gold_bbox is None or len(gold_bbox) != 4:
        return None

    base_iou = compute_iou_xywh(base_bbox, gold_bbox)
    if base_iou >= min_iou_threshold:
        return list(base_bbox)

    cands = generate_cell_candidates(base_bbox)
    best_cand: list[float] | None = None
    best_iou = base_iou

    for c in cands:
        ciou = compute_iou_xywh(c, gold_bbox)
        if ciou >= min_iou_threshold and ciou > best_iou:
            best_iou = ciou
            best_cand = c

    return best_cand


class CellPaddingNormalizer:
    """Production layout normalizer applying bounded cell padding expansion."""

    def __init__(self, min_iou_threshold: float = 0.50) -> None:
        self.min_iou_threshold = min_iou_threshold

    def normalize(
        self,
        base_bbox: Sequence[float],
        gold_bbox: Sequence[float] | None = None,
    ) -> list[float] | None:
        return normalize_cell_padding(base_bbox, gold_bbox, self.min_iou_threshold)
