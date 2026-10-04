"""Hungarian bipartite table assigner for dense tabular fields.

Target: Solves multi-candidate ambiguity in repeated values across table rows
using global Hungarian bipartite matching (scipy.optimize.linear_sum_assignment)
with row monotonicity constraints.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

_ARRAY_PATTERN = re.compile(r"^(.*)\[(\d+)\](.*)$")


def compute_iou_xywh(
    b1: tuple[float, float, float, float] | Sequence[float],
    b2: tuple[float, float, float, float] | Sequence[float],
) -> float:
    """Compute Intersection over Union (IoU) between two [x, y, w, h] bounding boxes."""
    if len(b1) != 4 or len(b2) != 4:
        return 0.0

    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, w1 * h1)
    area2 = max(0.0, w2 * h2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def parse_array_field_path(field_path: str) -> tuple[str, int, str] | None:
    """Parse array index path, e.g., 'holdings[3].shares' -> ('holdings', 3, '.shares')."""
    match = _ARRAY_PATTERN.match(field_path)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    return None


class HungarianTableAssigner:
    """Performs optimal Hungarian bipartite matching for repeated tabular values."""

    def __init__(self, min_iou_threshold: float = 0.30) -> None:
        self.min_iou_threshold = min_iou_threshold

    def assign_array_group(
        self,
        records: list[dict[str, Any]],
        candidate_pool_by_field: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        """Perform optimal bipartite matching for an array group.

        Args:
            records: list of dicts with 'field_path', 'gold_bbox', 'gold_page'
            candidate_pool_by_field: field_path -> list of candidate dicts

        Returns:
            dict of field_path -> assigned candidate dict
        """
        if len(records) < 2:
            return {}

        sorted_records = []
        for r in records:
            parsed = parse_array_field_path(r["field_path"])
            if parsed:
                sorted_records.append((parsed[1], r))
        if not sorted_records:
            return {}

        sorted_records.sort(key=lambda x: x[0])
        ordered_records = [x[1] for x in sorted_records]

        # Collect distinct candidates across all records in this array group
        all_candidates: list[dict[str, Any]] = []
        seen_cand_keys: set[tuple[Any, ...]] = set()
        for r in ordered_records:
            cands = candidate_pool_by_field.get(r["field_path"], [])
            for c in cands:
                key = (c.get("page"), tuple(c.get("bbox", ())))
                if key not in seen_cand_keys:
                    seen_cand_keys.add(key)
                    all_candidates.append(c)

        if len(all_candidates) < 2:
            return {}

        n_records = len(ordered_records)
        n_candidates = len(all_candidates)

        # Build cost matrix: rows = records, cols = candidates
        cost_matrix = np.full((n_records, n_candidates), 1e6)

        for i, rec in enumerate(ordered_records):
            gb = rec.get("gold_bbox")
            gp = rec.get("gold_page")
            if not gb:
                continue

            for j, cand in enumerate(all_candidates):
                cb = cand.get("bbox")
                cp = cand.get("page")
                if cp != gp or not cb:
                    continue

                iou = compute_iou_xywh(gb, cb)
                if iou >= self.min_iou_threshold:
                    cost = 1.0 - iou
                    cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: dict[str, dict[str, Any]] = {}
        for r_idx, c_idx in zip(row_ind, col_ind):
            if cost_matrix[r_idx, c_idx] <= (1.0 - self.min_iou_threshold + 1e-5):
                rec = ordered_records[r_idx]
                assignments[rec["field_path"]] = all_candidates[c_idx]

        return assignments


GlobalTableAssigner = HungarianTableAssigner
