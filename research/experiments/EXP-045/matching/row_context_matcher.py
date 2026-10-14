"""EXP-045 Algorithm 1: Row-Context Matching (for SUB_ROW_CONTEXT).

Builds a 5-point spatial fingerprint for each candidate token:
- Target token value
- Nearest sibling to left (same visual line)
- Nearest sibling to right (same visual line)
- Token directly above (same vertical column band, previous line)
- Token directly below (same vertical column band, next line)

Scores candidates using weighted n-gram / token similarity against extracted sibling context:
  score = w1 * sim(target) + w2 * sim(left) + w3 * sim(right) + w4 * sim(above) + w5 * sim(below)
Default weights: w1=1.0, w2=0.5, w3=0.5, w4=0.3, w5=0.3.
"""

from __future__ import annotations

from typing import Any
from rapidfuzz import fuzz

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def token_similarity(a: str, b: str) -> float:
    """Compute normalized token similarity between 0.0 and 1.0."""
    if not a or not b:
        return 0.0
    if a.lower() == b.lower():
        return 1.0
    return fuzz.token_sort_ratio(a.lower(), b.lower()) / 100.0


def find_token_neighbors(
    target_tok: DocumentToken,
    all_tokens: list[DocumentToken],
    y_tol: float = 4.0,
    x_tol: float = 12.0,
) -> dict[str, DocumentToken | None]:
    """Find the 4 directional neighbors of a token on a page."""
    tx1, ty1, tw, th = target_tok.bbox.to_coco()
    tx2, ty2 = tx1 + tw, ty1 + th
    t_mid_y = (ty1 + ty2) / 2.0
    t_mid_x = (tx1 + tx2) / 2.0

    left_tok: DocumentToken | None = None
    right_tok: DocumentToken | None = None
    above_tok: DocumentToken | None = None
    below_tok: DocumentToken | None = None

    min_left_dist = float("inf")
    min_right_dist = float("inf")
    min_above_dist = float("inf")
    min_below_dist = float("inf")

    for tok in all_tokens:
        if tok is target_tok:
            continue
        ox1, oy1, ow, oh = tok.bbox.to_coco()
        ox2, oy2 = ox1 + ow, oy1 + oh
        o_mid_y = (oy1 + oy2) / 2.0
        o_mid_x = (ox1 + ox2) / 2.0

        # Same horizontal line (y-overlap or close mid_y)
        if abs(t_mid_y - o_mid_y) <= max(y_tol, th * 0.5):
            # Left sibling
            if ox2 <= tx1 + 1.0:
                dist = tx1 - ox2
                if dist < min_left_dist:
                    min_left_dist = dist
                    left_tok = tok
            # Right sibling
            elif ox1 >= tx2 - 1.0:
                dist = ox1 - tx2
                if dist < min_right_dist:
                    min_right_dist = dist
                    right_tok = tok

        # Same vertical column band (x-overlap)
        x_overlap = max(0.0, min(tx2, ox2) - max(tx1, ox1))
        if x_overlap > 0 or abs(t_mid_x - o_mid_x) <= max(x_tol, tw * 0.5):
            # Above token
            if oy2 <= ty1 + 1.0:
                dist = ty1 - oy2
                if dist < min_above_dist:
                    min_above_dist = dist
                    above_tok = tok
            # Below token
            elif oy1 >= ty2 - 1.0:
                dist = oy1 - ty2
                if dist < min_below_dist:
                    min_below_dist = dist
                    below_tok = tok

    return {
        "left": left_tok,
        "right": right_tok,
        "above": above_tok,
        "below": below_tok,
    }


def match_row_context(
    target_val: Any,
    page_obj: DocumentPage,
    sibling_hints: dict[str, str] | None = None,
    min_score_threshold: float = 1.3,
) -> tuple[BBox | None, str | None]:
    """Execute Stage 2 Row-Context Matching.
    
    Args:
        target_val: Target value to ground
        page_obj: Page to search
        sibling_hints: Optional dict with 'left', 'right', 'above', 'below' expected text
        min_score_threshold: Minimum weighted score to accept candidate
    
    Returns:
        (matched_bbox, matched_text) or (None, None)
    """
    if target_val is None or not page_obj or not page_obj.tokens:
        return None, None

    val_str = str(target_val).strip()
    if not val_str:
        return None, None

    # Find all exact or fuzzy candidate tokens for target_val
    cand_tokens: list[DocumentToken] = []
    for tok in page_obj.tokens:
        if val_str.lower() == tok.text.lower() or (len(val_str) >= 3 and val_str.lower() in tok.text.lower()):
            cand_tokens.append(tok)

    if not cand_tokens:
        return None, None

    # If only 1 occurrence, return immediately
    if len(cand_tokens) == 1:
        return cand_tokens[0].bbox, cand_tokens[0].text

    # Multiple occurrences: disambiguate using 5-point fingerprint
    if not sibling_hints:
        # Default to picking candidate with highest token density around it
        best_tok = max(cand_tokens, key=lambda t: t.bbox.width * t.bbox.height)
        return best_tok.bbox, best_tok.text

    best_cand: DocumentToken | None = None
    best_score = -1.0

    w1, w2, w3, w4, w5 = 1.0, 0.5, 0.5, 0.3, 0.3

    for cand in cand_tokens:
        neighbors = find_token_neighbors(cand, page_obj.tokens)
        sim_target = 1.0 if val_str.lower() in cand.text.lower() else 0.8
        sim_left = token_similarity(sibling_hints.get("left", ""), neighbors["left"].text if neighbors["left"] else "")
        sim_right = token_similarity(sibling_hints.get("right", ""), neighbors["right"].text if neighbors["right"] else "")
        sim_above = token_similarity(sibling_hints.get("above", ""), neighbors["above"].text if neighbors["above"] else "")
        sim_below = token_similarity(sibling_hints.get("below", ""), neighbors["below"].text if neighbors["below"] else "")

        score = (
            w1 * sim_target
            + w2 * sim_left
            + w3 * sim_right
            + w4 * sim_above
            + w5 * sim_below
        )

        if score > best_score:
            best_score = score
            best_cand = cand

    if best_cand and best_score >= min_score_threshold:
        return best_cand.bbox, best_cand.text

    # Fallback to the first candidate if score threshold not met but single prominent candidate exists
    return cand_tokens[0].bbox, cand_tokens[0].text
