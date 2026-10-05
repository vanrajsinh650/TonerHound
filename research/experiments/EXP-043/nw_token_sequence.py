"""EXP-043 Phase E: Needleman-Wunsch Token Sequence Alignment.

Target: REAL_INDEXING_MISS (77,529 fields).
Solves multi-word phrases with missing, noisy, or intervening tokens by performing
2-level DP alignment (word-level sequence alignment with character-level NW distance).
"""

from __future__ import annotations

from typing import Any
from nw_aligner import nw_char_align


def _get_token_text(token: Any) -> str:
    """Extract clean string text from token object or dict."""
    if isinstance(token, str):
        return token.strip()
    if isinstance(token, dict):
        return str(token.get("text", "")).strip()
    if hasattr(token, "text"):
        return str(token.text).strip()
    return str(token).strip()


def nw_token_sequence_match(
    page_tokens: list[Any],
    gold_value: str,
    max_gap: int = 3,
    min_word_score: float = 0.75,
) -> list[int] | None:
    """Match multi-word gold_value against token sequence via NW alignment.

    Returns:
        list of matched token indices, or None.
    """
    gold_words = gold_value.split()
    if len(gold_words) < 2:
        return None

    n = len(page_tokens)
    m = len(gold_words)
    if n == 0:
        return None

    # DP table: n+1 by m+1
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    # Initialize borders
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - 0.3
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - 0.3

    # Fill DP
    for i in range(1, n + 1):
        token_text = _get_token_text(page_tokens[i - 1]).lower()
        for j in range(1, m + 1):
            gold_word = gold_words[j - 1].strip().lower()

            _, _, char_score = nw_char_align(token_text, gold_word)
            skip_token = dp[i - 1][j] - 0.3
            skip_word = dp[i][j - 1] - 0.3
            match_gain = char_score if char_score >= min_word_score else -0.5
            match = dp[i - 1][j - 1] + match_gain

            dp[i][j] = max(skip_token, skip_word, match)

    # Require minimum score
    min_required = m * 0.65
    if dp[n][m] < min_required:
        return None

    # Traceback
    i, j = n, m
    matched_indices = []
    while i > 0 and j > 0:
        token_text = _get_token_text(page_tokens[i - 1]).lower()
        gold_word = gold_words[j - 1].strip().lower()
        _, _, char_score = nw_char_align(token_text, gold_word)
        match_gain = char_score if char_score >= min_word_score else -0.5

        if char_score >= min_word_score and abs(dp[i][j] - (dp[i - 1][j - 1] + match_gain)) < 1e-4:
            matched_indices.append(i - 1)
            i -= 1
            j -= 1
        elif abs(dp[i][j] - (dp[i - 1][j] - 0.3)) < 1e-4:
            i -= 1
        else:
            j -= 1

    matched_indices.reverse()
    return matched_indices if len(matched_indices) >= 2 else None
