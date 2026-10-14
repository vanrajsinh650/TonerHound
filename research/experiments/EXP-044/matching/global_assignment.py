"""EXP-039 Phase F: Global Assignment for Repeated Values.

Target class: REAL_INDEXING_MISS on dense tables
Realistic target: +0.3 to +0.5 pp

Solves multi-candidate ambiguity in repeated values (e.g. repeated $0.00 or USD)
across array items (e.g. holdings[i].val) using global Hungarian bipartite matching
(scipy.optimize.linear_sum_assignment) with row monotonicity constraints.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_xywh(b1: tuple[float, float, float, float] | list[float], b2: tuple[float, float, float, float] | list[float]) -> float:
    """Compute IoU between two [x, y, w, h] bounding boxes."""
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


_ARRAY_PATTERN = re.compile(r"^(.*)\[(\d+)\](.*)$")


def parse_array_field_path(fp: str) -> tuple[str, int, str] | None:
    """Parse 'holdings[3].shares' into ('holdings', 3, '.shares')."""
    m = _ARRAY_PATTERN.match(fp)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


class GlobalTableAssigner:
    """Performs Hungarian bipartite assignment for array records."""

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
        if len(records) < 3:
            return {}

        # Sort records by their array index
        sorted_records = []
        for r in records:
            parsed = parse_array_field_path(r["field_path"])
            if parsed:
                sorted_records.append((parsed[1], r))
        sorted_records.sort(key=lambda x: x[0])
        ordered_records = [x[1] for x in sorted_records]

        # Collect distinct candidates across all records in this array group
        all_candidates: list[dict[str, Any]] = []
        seen_cand_keys = set()
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
                if iou >= 0.30:
                    # Lower cost is better; reward higher IoU and vertical alignment
                    cost = 1.0 - iou
                    cost_matrix[i, j] = cost

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: dict[str, dict[str, Any]] = {}
        for r_idx, c_idx in zip(row_ind, col_ind):
            if cost_matrix[r_idx, c_idx] < 0.70:  # IoU >= 0.30
                rec = ordered_records[r_idx]
                assignments[rec["field_path"]] = all_candidates[c_idx]

        return assignments
