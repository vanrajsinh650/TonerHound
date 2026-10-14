"""EXP-044 Stage 2: Character-Level Bounded Fuzzy Matcher.

Requirements:
- Levenshtein distance with bounded edit cost
- Only apply when edit distance <= 2
- Sliding window over candidate text regions
- Reject matches where character substitutions aren't in the allowed confusion set:
    {0<->O, 1<->l, 1<->I, 5<->S, 8<->B, 2<->Z}
- Gate: only apply when Stage 1 failed.
"""

from __future__ import annotations

from typing import Any
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken

# Allowed OCR / typography character confusion pairs (bidirectional)
ALLOWED_CONFUSIONS: set[frozenset[str]] = {
    frozenset({"0", "O"}),
    frozenset({"0", "o"}),
    frozenset({"1", "l"}),
    frozenset({"1", "I"}),
    frozenset({"1", "i"}),
    frozenset({"5", "S"}),
    frozenset({"5", "s"}),
    frozenset({"8", "B"}),
    frozenset({"2", "Z"}),
    frozenset({"2", "z"}),
}


def is_allowed_substitution(c1: str, c2: str) -> bool:
    """Check if substitution between c1 and c2 is in the allowed confusion set."""
    if c1 == c2 or c1.lower() == c2.lower():
        return True
    return frozenset({c1, c2}) in ALLOWED_CONFUSIONS or frozenset({c1.upper(), c2.upper()}) in ALLOWED_CONFUSIONS


def bounded_levenshtein(s1: str, s2: str, max_dist: int = 2) -> tuple[int, bool]:
    """Compute Levenshtein distance bounded by max_dist.
    
    Returns (distance, is_valid_substitution_only).
    If distance > max_dist or an unallowed substitution is used, is_valid is False.
    """
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > max_dist:
        return 999, False

    # DP matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    dist = dp[len1][len2]
    if dist > max_dist:
        return dist, False

    # Traceback to verify substitutions against allowed confusion set
    i, j = len1, len2
    valid_confusions = True
    while i > 0 and j > 0:
        c1, c2 = s1[i - 1], s2[j - 1]
        if c1 == c2:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution occurred
            if not is_allowed_substitution(c1, c2):
                valid_confusions = False
                break
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            # Deletion (e.g. extra punctuation)
            if c1 not in " ,.;:-_/'\"":
                valid_confusions = False
                break
            i -= 1
        else:
            # Insertion
            if c2 not in " ,.;:-_/'\"":
                valid_confusions = False
                break
            j -= 1

    return dist, valid_confusions


def match_fuzzy_stage2(
    target_value: Any,
    page_obj: DocumentPage,
    max_edit_distance: int = 2,
) -> tuple[BBox | None, str | None]:
    """Execute Stage 2 character-level bounded fuzzy matching.
    
    Returns (bbox, matched_text) if a valid match is found.
    """
    if target_value is None:
        return None, None

    val_str = str(target_value).strip()
    if len(val_str) < 3:
        # Bounded fuzzy match requires length >= 3 to prevent false positives
        return None, None

    best_match: tuple[int, BBox, str] | None = None

    # 1. Check line tokens
    for line in page_obj.lines:
        ltoks = line.tokens
        # Check single tokens
        for tok in ltoks:
            t_txt = tok.text.strip(" ,.;:'\"()[]{}")
            if abs(len(t_txt) - len(val_str)) <= max_edit_distance:
                dist, valid = bounded_levenshtein(val_str, t_txt, max_dist=max_edit_distance)
                if valid and dist <= max_edit_distance:
                    if best_match is None or dist < best_match[0]:
                        best_match = (dist, tok.bbox, t_txt)
                        if dist == 1:
                            return tok.bbox, t_txt

        # Check sliding window of tokens for multi-token targets
        words_target = val_str.split()
        if len(words_target) > 1 and len(ltoks) >= len(words_target):
            k = len(words_target)
            for start in range(len(ltoks) - k + 1):
                sub_toks = ltoks[start : start + k]
                candidate_str = " ".join(t.text.strip(" ,.;:'\"()[]{}") for t in sub_toks)
                dist, valid = bounded_levenshtein(val_str, candidate_str, max_dist=max_edit_distance)
                if valid and dist <= max_edit_distance:
                    min_x = min(t.bbox.x for t in sub_toks)
                    min_y = min(t.bbox.y for t in sub_toks)
                    max_x = max(t.bbox.x + t.bbox.width for t in sub_toks)
                    max_y = max(t.bbox.y + t.bbox.height for t in sub_toks)
                    ubox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y, page=page_obj.page_number)
                    if best_match is None or dist < best_match[0]:
                        best_match = (dist, ubox, candidate_str)

    if best_match:
        return best_match[1], best_match[2]

    return None, None
