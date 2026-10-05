"""EXP-043 Phase A: Needleman-Wunsch Character Alignment.

Target: NORMALIZATION_MISMATCH (46,156 fields).
Provides optimal global sequence alignment between gold string values and page tokens,
recovering character-level noise, dropped prefixes/suffixes, and formatting discrepancies.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def nw_char_align(
    a: str,
    b: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -1,
) -> tuple[str, str, float]:
    """Needleman-Wunsch character alignment.

    Returns:
        (aligned_a, aligned_b, normalized_score) where score in [0.0, 1.0].
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return a, b, 0.0

    dp = np.zeros((n + 1, m + 1), dtype=float)

    # Initialize borders
    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                diag = dp[i - 1][j - 1] + match
            else:
                diag = dp[i - 1][j - 1] + mismatch
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            dp[i][j] = max(diag, up, left)

    # Traceback
    i, j = n, m
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match if a[i - 1] == b[j - 1] else mismatch
            if abs(dp[i][j] - (dp[i - 1][j - 1] + score)) < 1e-5:
                aligned_a.append(a[i - 1])
                aligned_b.append(b[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + gap)) < 1e-5:
            aligned_a.append(a[i - 1])
            aligned_b.append("-")
            i -= 1
            continue
        if j > 0 and abs(dp[i][j] - (dp[i][j - 1] + gap)) < 1e-5:
            aligned_a.append("-")
            aligned_b.append(b[j - 1])
            j -= 1
            continue
        # Fallback step
        if i > 0:
            i -= 1
        elif j > 0:
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()

    # Normalize score to [0, 1]
    max_score = max(len(a), len(b)) * match
    normalized = max(0.0, dp[n][m] / max_score) if max_score > 0 else 0.0
    return "".join(aligned_a), "".join(aligned_b), float(normalized)


def find_nw_candidates(
    page_tokens: list[Any],
    gold_value: str,
    threshold: float = 0.75,
) -> list[tuple[Any, float]]:
    """Find page tokens matching gold_value via NW alignment.

    Returns:
        [(token, score), ...] sorted by score descending.
    """
    candidates = []
    gold_clean = gold_value.strip().lower()
    gold_len = len(gold_clean)

    for token in page_tokens:
        text = (getattr(token, "text", "") or str(token)).strip()
        tok_clean = text.lower()
        token_len = len(tok_clean)

        # Length gate: reject if lengths differ > 50%
        if abs(token_len - gold_len) > gold_len * 0.5:
            continue
        # Minimum length gate: too short = many false positives
        if gold_len < 4 or token_len < 4:
            continue

        _, _, score = nw_char_align(gold_clean, tok_clean)
        if score >= threshold:
            candidates.append((token, score))

    return sorted(candidates, key=lambda x: -x[1])
