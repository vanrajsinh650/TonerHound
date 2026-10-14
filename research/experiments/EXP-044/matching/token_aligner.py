"""EXP-044 Stage 3: Needleman-Wunsch Token Sequence Aligner.

Requirements:
- Token-level global alignment via Needleman-Wunsch
- Cost matrix:
    - identical = 0
    - canonical-equal = 1 (e.g. "USD" vs "$", "100.00" vs "100")
    - unrelated = 5
    - gap (insertion/deletion) = 2
- Only activate for multi-token values (length >= 2)
- Reject alignments with total cost > 3
- Gate: only apply when Stages 1 and 2 failed.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def is_canonical_equal(tok1: str, tok2: str) -> bool:
    """Check if two tokens are canonical-equal (cost 1)."""
    t1 = tok1.strip(" ,.;:'\"()[]{}").lower()
    t2 = tok2.strip(" ,.;:'\"()[]{}").lower()
    if t1 == t2:
        return True

    # Punctuation or comma difference e.g. "1,000" vs "1000"
    if t1.replace(",", "") == t2.replace(",", ""):
        return True

    # Trailing .00 e.g. "500.00" vs "500"
    if re.sub(r"\.00$", "", t1) == re.sub(r"\.00$", "", t2):
        return True

    # Currency symbol vs code e.g. "usd" vs "$"
    curr_equiv = [
        {"usd", "$"},
        {"eur", "€"},
        {"gbp", "£"},
        {"jpy", "¥"},
    ]
    for eq in curr_equiv:
        if t1 in eq and t2 in eq:
            return True

    # Percent e.g. "5%" vs "0.05"
    if t1.endswith("%") or t2.endswith("%"):
        try:
            n1 = float(t1.rstrip("%")) / (100.0 if t1.endswith("%") else 1.0)
            n2 = float(t2.rstrip("%")) / (100.0 if t2.endswith("%") else 1.0)
            if abs(n1 - n2) < 1e-4:
                return True
        except ValueError:
            pass

    return False


def needleman_wunsch_token_align(
    target_tokens: list[str],
    candidate_tokens: list[str],
) -> int:
    """Compute Needleman-Wunsch alignment cost between two token sequences.
    
    Cost matrix:
      identical = 0
      canonical-equal = 1
      unrelated = 5
      gap = 2
    """
    n, m = len(target_tokens), len(candidate_tokens)
    # Initialize DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * 2  # gap cost = 2
    for j in range(m + 1):
        dp[0][j] = j * 2  # gap cost = 2

    for i in range(1, n + 1):
        t_tok = target_tokens[i - 1]
        for j in range(1, m + 1):
            c_tok = candidate_tokens[j - 1]

            t_clean = t_tok.strip(" ,.;:'\"()[]{}").lower()
            c_clean = c_tok.strip(" ,.;:'\"()[]{}").lower()

            if t_clean == c_clean:
                sub_cost = 0
            elif is_canonical_equal(t_clean, c_clean):
                sub_cost = 1
            else:
                sub_cost = 5

            dp[i][j] = min(
                dp[i - 1][j] + 2,        # target gap
                dp[i][j - 1] + 2,        # candidate gap
                dp[i - 1][j - 1] + sub_cost # match/sub
            )

    return dp[n][m]


def match_token_aligner_stage3(
    target_value: Any,
    page_obj: DocumentPage,
    max_alignment_cost: int = 3,
) -> tuple[BBox | None, str | None]:
    """Execute Stage 3 Needleman-Wunsch token alignment.
    
    Only activates for multi-token values (token count >= 2).
    Rejects alignments with total cost > max_alignment_cost (default 3).
    """
    if target_value is None:
        return None, None

    val_str = str(target_value).strip()
    target_tokens = [t for t in val_str.split() if t]
    if len(target_tokens) < 2:
        return None, None

    n_target = len(target_tokens)
    best_match: tuple[int, BBox, str] | None = None

    # Slide window over line tokens allowing +/- 1 token size window
    for line in page_obj.lines:
        ltoks = line.tokens
        if len(ltoks) < n_target - 1:
            continue

        for win_len in [n_target, n_target + 1, n_target - 1]:
            if win_len < 1 or win_len > len(ltoks):
                continue
            for start in range(len(ltoks) - win_len + 1):
                sub_toks = ltoks[start : start + win_len]
                cand_token_texts = [t.text for t in sub_toks]

                cost = needleman_wunsch_token_align(target_tokens, cand_token_texts)
                if cost <= max_alignment_cost:
                    min_x = min(t.bbox.x for t in sub_toks)
                    min_y = min(t.bbox.y for t in sub_toks)
                    max_x = max(t.bbox.x + t.bbox.width for t in sub_toks)
                    max_y = max(t.bbox.y + t.bbox.height for t in sub_toks)
                    ubox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y, page=page_obj.page_number)
                    matched_str = " ".join(cand_token_texts)

                    if best_match is None or cost < best_match[0]:
                        best_match = (cost, ubox, matched_str)
                        if cost == 0:
                            return ubox, matched_str

    if best_match:
        return best_match[1], best_match[2]

    return None, None
