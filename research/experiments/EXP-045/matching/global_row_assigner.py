"""EXP-045 Algorithm 3: Global Row Assignment (Hungarian Table Optimization).

Solves the entire table's row assignment as a global bipartite matching problem
between extracted records (schema rows) and visual table rows detected on the page.
Guarantees globally optimal assignment for dense tables with repeated values.
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from rapidfuzz import fuzz

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken

_ARRAY_ROW_PATTERN = re.compile(r"^(.*)\[(\d+)\](.*)$")


class GlobalRowAssigner:
    """Solves row-to-candidate assignment via Hungarian global bipartite matching."""

    def assign_table_rows(
        self,
        records: list[dict[str, Any]],
        page_obj: DocumentPage,
    ) -> dict[str, dict[str, Any]]:
        """Perform optimal bipartite matching for an array group.
        
        Args:
            records: list of dicts with 'field_path', 'target_value', 'gold_page'
            page_obj: Document page containing candidate tokens
            
        Returns:
            dict of field_path -> citation dictionary
        """
        if len(records) < 2 or len(records) > 200 or not page_obj or not page_obj.tokens:
            return {}

        # 1. Group records by target row index
        row_records: dict[int, list[dict[str, Any]]] = {}
        for r in records:
            m = _ARRAY_ROW_PATTERN.match(r["field_path"])
            if m:
                r_idx = int(m.group(2))
                if r_idx not in row_records:
                    row_records[r_idx] = []
                row_records[r_idx].append(r)

        if not row_records:
            return {}

        sorted_row_indices = sorted(row_records.keys())
        assigned_cits: dict[str, dict[str, Any]] = {}

        # 2. Extract visual lines on page as candidate visual rows
        # Cluster tokens by y-coordinate into visual lines
        line_buckets: dict[int, list[DocumentToken]] = {}
        y_resolution = 5.0
        for tok in page_obj.tokens:
            k = int(tok.bbox.y / y_resolution)
            if k not in line_buckets:
                line_buckets[k] = []
            line_buckets[k].append(tok)

        visual_rows = [sorted(toks, key=lambda t: t.bbox.x) for toks in line_buckets.values() if len(toks) >= 2]
        visual_rows.sort(key=lambda row: row[0].bbox.y)

        if len(visual_rows) < 2 or len(sorted_row_indices) > 120 or len(visual_rows) > 120:
            return {}

        # Precompute visual row token word sets and concatenated text
        v_sets = [{t.text.lower() for t in v_row} for v_row in visual_rows]
        v_strings = [" ".join(t.text.lower() for t in v_row) for v_row in visual_rows]

        # 3. For each schema row, compute best matching visual row
        # Cost matrix: N schema rows x M visual rows
        n_schema = len(sorted_row_indices)
        m_visual = len(visual_rows)
        cost_matrix = np.ones((n_schema, m_visual), dtype=np.float64)

        for i, s_idx in enumerate(sorted_row_indices):
            s_recs = row_records[s_idx]
            s_values = [str(r["target_value"]).strip().lower() for r in s_recs if r.get("target_value")]
            if not s_values:
                continue

            for j, (v_set, v_str) in enumerate(zip(v_sets, v_strings)):
                matches = 0
                for sv in s_values:
                    if sv in v_set or (len(sv) >= 4 and sv in v_str):
                        matches += 1
                sim = matches / len(s_values) if s_values else 0.0

                # Spatial monotonicity bonus
                pos_ratio_s = i / max(1, n_schema - 1)
                pos_ratio_v = j / max(1, m_visual - 1)
                pos_diff = abs(pos_ratio_s - pos_ratio_v)
                spatial_penalty = pos_diff * 0.4

                cost_matrix[i, j] = max(0.0, 1.0 - sim + spatial_penalty)

        # 4. Solve global Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for s_pos, v_pos in zip(row_ind, col_ind):
            if cost_matrix[s_pos, v_pos] > 0.85:
                continue  # Skip unviable matches
            s_idx = sorted_row_indices[s_pos]
            v_row = visual_rows[v_pos]
            s_recs = row_records[s_idx]

            for rec in s_recs:
                fp = rec["field_path"]
                tv = str(rec.get("target_value", "")).strip().lower()
                if not tv:
                    continue

                # Find token inside v_row matching tv
                best_match_tok: DocumentToken | None = None
                for t in v_row:
                    if tv == t.text.lower() or (len(tv) >= 3 and tv in t.text.lower()):
                        best_match_tok = t
                        break

                if best_match_tok:
                    assigned_cits[fp] = {
                        "field_path": fp,
                        "page": page_obj.page_number,
                        "bbox": list(best_match_tok.bbox.to_coco()),
                        "polygon": None,
                        "reference_text": best_match_tok.text,
                        "confidence": 0.88,
                        "source": "exp045_global_row_assign",
                        "metadata": {"stage": "stage_3_global_row"},
                    }

        return assigned_cits
