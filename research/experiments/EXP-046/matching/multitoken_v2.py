"""EXP-046 Stage 13: Multi-Word Sequence Matcher V2.

Builds token adjacency graphs along visual reading lines and executes Needleman-Wunsch
sequence alignment with gap penalties to handle skip-tokens (e.g. 'Bank [of] America').
"""

from __future__ import annotations

from typing import Any, Sequence
from rapidfuzz import fuzz

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def needleman_wunsch_token_align(seq1: list[str], seq2: list[str]) -> float:
    """Computes normalized Needleman-Wunsch similarity score for token sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0

    match_score = 2
    mismatch_score = -1
    gap_penalty = -1

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(m + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            t1 = seq1[i - 1].lower()
            t2 = seq2[j - 1].lower()
            score = match_score if (t1 == t2 or fuzz.ratio(t1, t2) >= 85) else mismatch_score
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap_penalty,
                dp[i][j - 1] + gap_penalty,
            )

    max_possible = max(n, m) * match_score
    raw_score = dp[n][m]
    return max(0.0, raw_score / max(1.0, max_possible))


class MultiTokenMatcherV2:
    """Aligns arbitrary-length token sequences across visual lines."""

    def __init__(self, min_similarity: float = 0.70) -> None:
        self.min_similarity = min_similarity

    def match_multitoken_sequence(
        self, target_value: Any, page_obj: DocumentPage
    ) -> tuple[BBox | None, str | None]:
        """Align multi-token target against sliding windows of page tokens."""
        target_str = str(target_value).strip()
        target_tokens = target_str.split()
        if len(target_tokens) <= 1:
            return None, None

        page_tokens = sorted(page_obj.tokens, key=lambda t: (t.bbox.y, t.bbox.x))
        k = len(target_tokens)

        best_score = 0.0
        best_window: list[DocumentToken] = []

        # Sliding window with gap expansion allowance
        for i in range(len(page_tokens)):
            window = page_tokens[i : i + k + 2]
            if len(window) < k - 1:
                continue

            # Ensure tokens in window share rough vertical alignment
            y_span = max(t.bbox.y for t in window) - min(t.bbox.y for t in window)
            if y_span > 0.04:
                continue

            win_texts = [t.text.strip() for t in window]
            score = needleman_wunsch_token_align(target_tokens, win_texts)
            if score > best_score:
                best_score = score
                best_window = window

        if best_score >= self.min_similarity and best_window:
            rx0 = min(t.bbox.x for t in best_window)
            ry0 = min(t.bbox.y for t in best_window)
            rw = max(t.bbox.x + t.bbox.width for t in best_window) - rx0
            rh = max(t.bbox.y + t.bbox.height for t in best_window) - ry0
            matched_text = " ".join(t.text for t in best_window)
            return BBox(rx0, ry0, rw, rh, page_obj.page_number), matched_text

        return None, None
