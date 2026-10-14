"""EXP-044 Stage 4: Token Set Matching (Hungarian Algorithm).

Requirements:
- Build bipartite cost matrix between target tokens and candidate tokens
- Solve optimal token permutation alignment with scipy.optimize.linear_sum_assignment
- Only activate when token order/permutation is the difference
- Reject alignments when total assignment cost > 3
- Gate: only apply when Stages 1-3 failed.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken
from token_aligner import is_canonical_equal


def token_distance(t1: str, t2: str) -> float:
    """Pairwise token alignment cost."""
    c1 = t1.strip(" ,.;:'\"()[]{}").lower()
    c2 = t2.strip(" ,.;:'\"()[]{}").lower()
    if c1 == c2:
        return 0.0
    if is_canonical_equal(c1, c2):
        return 1.0
    return 5.0


def hungarian_token_set_cost(target_tokens: list[str], cand_tokens: list[str]) -> float:
    """Solve optimal bipartite assignment cost between two token multisets."""
    n = len(target_tokens)
    m = len(cand_tokens)
    if n == 0 or m == 0:
        return 999.0

    size = max(n, m)
    cost_matrix = np.full((size, size), 2.0)  # Gap cost = 2.0 for padding

    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = token_distance(target_tokens[i], cand_tokens[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = float(cost_matrix[row_ind, col_ind].sum())
    return total_cost


def match_token_set_stage4(
    target_value: Any,
    page_obj: DocumentPage,
    max_total_cost: float = 3.0,
) -> tuple[BBox | None, str | None]:
    """Execute Stage 4 Hungarian token set matching.
    
    Identifies unordered token permutations (e.g. "USD 5000" vs "5000 USD",
    "John Doe" vs "Doe, John") within the same visual line or region.
    """
    if target_value is None:
        return None, None

    val_str = str(target_value).strip()
    target_tokens = [t for t in val_str.split() if t]
    if len(target_tokens) < 2:
        return None, None

    n_target = len(target_tokens)
    best_match: tuple[float, BBox, str] | None = None

    for line in page_obj.lines:
        ltoks = line.tokens
        if len(ltoks) < n_target:
            continue

        for start in range(len(ltoks) - n_target + 1):
            sub_toks = ltoks[start : start + n_target]
            cand_tokens = [t.text for t in sub_toks]

            # Sequential order check: if exact sequential order matches, Stage 3 or earlier would have matched
            # Stage 4 specifically targets order permutations
            cost = hungarian_token_set_cost(target_tokens, cand_tokens)
            if cost <= max_total_cost:
                min_x = min(t.bbox.x for t in sub_toks)
                min_y = min(t.bbox.y for t in sub_toks)
                max_x = max(t.bbox.x + t.bbox.width for t in sub_toks)
                max_y = max(t.bbox.y + t.bbox.height for t in sub_toks)
                ubox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y, page=page_obj.page_number)
                matched_str = " ".join(cand_tokens)

                if best_match is None or cost < best_match[0]:
                    best_match = (cost, ubox, matched_str)

    if best_match:
        return best_match[1], best_match[2]

    return None, None
