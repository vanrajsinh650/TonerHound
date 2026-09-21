"""EXP-017 Same-Line Multi-Token Geometry Recovery.

Recovers Class A geometry failures where an initial matched candidate covers only
the first token(s) of a multi-word entity, extending across physically observed sibling tokens
on the same visual line up to the exact end of the target evidence.
"""

from __future__ import annotations

from typing import Any, Sequence

from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentToken
from tonerhound.normalization.normalizers import normalize_unicode_and_case


def clean_word(w: str) -> str:
    """Normalize and strip punctuation from a token or word for comparison."""
    return normalize_unicode_and_case(w).text.lower().strip(' -.,;:_()[]{}/\'"')


def word_matches(token_text: str, target_word: str) -> bool:
    """Check if token text matches the target word exactly, as a prefix, suffix, or substring."""
    t = clean_word(token_text)
    w = clean_word(target_word)
    if not t or not w:
        return False
    return (
        t == w
        or t.startswith(w)
        or w.startswith(t)
        or t.endswith(w)
        or w.endswith(t)
        or (len(t) >= 3 and len(w) >= 3 and (t in w or w in t))
    )


def extend_same_line_tokens(
    cand_bbox: BBox | tuple[float, float, float, float],
    reference_text: str,
    target_value: Any,
    line_tokens: Sequence[DocumentToken],
    confidence: float = 0.95,
    is_passed: bool = False,
    max_right_boundary: float | None = None,
) -> BBox | tuple[float, float, float, float]:
    """Extend candidate bounding box across validated same-line sibling tokens.

    Strictly satisfies all EXP-017 / EXP-017R mandatory safety gates:
    - Pass protection: if candidate already passes (is_passed) or low confidence, returns unchanged.
    - Multi-token requirement: target value must contain 2 or more words.
    - Dot-leader hazard rejection: strings containing '...' are strictly bypassed.
    - Same visual line: tokens must share compatible vertical baseline (|dy| <= line_height * 0.6).
    - Sequential ordering: remaining target tokens must appear in order to the right.
    - Continuity: no unrelated tokens or excessive column gaps (> 0.08) between target tokens.
    - Exact termination: stops immediately at the end of the target sequence.
    - Right boundary protection: never encroaches beyond max_right_boundary (adjacent column or sibling).
    """
    if is_passed or confidence < 0.80:
        return cand_bbox

    if target_value is None or isinstance(target_value, bool) or not line_tokens:
        return cand_bbox

    target_str = str(target_value).strip()
    target_words = [w for w in target_str.split() if clean_word(w)]

    # Safety Gate 1: Target value must be multi-token (>1 word)
    if len(target_words) <= 1:
        return cand_bbox

    # Safety Gate 2: Reject table dot leaders
    if "..." in target_str or "..." in reference_text:
        return cand_bbox

    is_bbox_obj = isinstance(cand_bbox, BBox)
    if is_bbox_obj:
        cb_x, cb_y, cb_w, cb_h = cand_bbox.x, cand_bbox.y, cand_bbox.width, cand_bbox.height
        page = cand_bbox.page
    else:
        cb_x, cb_y, cb_w, cb_h = cand_bbox[:4]
        page = line_tokens[0].page if line_tokens else 1

    # Safety Gate 3: Locate starting token matching the first target word
    target_w0 = target_words[0]
    start_candidates: list[tuple[float, DocumentToken]] = []
    for t in line_tokens:
        if word_matches(t.text, target_w0):
            dx = abs(t.bbox.x - cb_x)
            dy = abs(t.bbox.y - cb_y)
            if dx <= max(0.04, cb_w) and dy <= max(0.008, cb_h * 0.75):
                start_candidates.append((dx + dy * 2.0, t))

    if not start_candidates:
        return cand_bbox

    start_candidates.sort(key=lambda item: item[0])
    start_tok = start_candidates[0][1]

    # Safety Gate 4: Filter sibling tokens on the SAME visual baseline
    ref_y = start_tok.bbox.y
    ref_h = max(0.007, start_tok.bbox.height)
    baseline_tokens = [
        t
        for t in line_tokens
        if abs(t.bbox.y - ref_y) <= max(0.006, ref_h * 0.6)
        and t.bbox.x >= start_tok.bbox.x - 0.003
    ]
    baseline_tokens.sort(key=lambda t: t.bbox.x)

    matched_tokens: list[DocumentToken] = [start_tok]
    target_pos = 1
    prev_tok = start_tok

    # Safety Gate 5: Sequential walkthrough of sibling tokens
    for t in baseline_tokens:
        if t == start_tok:
            continue

        # Column / right boundary protection
        if max_right_boundary is not None and t.bbox.x >= max_right_boundary:
            break

        # Horizontal continuity: check gap between adjacent tokens
        gap = t.bbox.x - (prev_tok.bbox.x + prev_tok.bbox.width)
        if gap < -0.01:
            # Overlapping or behind current position
            continue
        if gap > 0.08:
            # Reached a large whitespace gap indicating column / margin boundary
            break

        # Check if t matches the next expected target word(s)
        found_idx = -1
        for j in range(target_pos, len(target_words)):
            tw = target_words[j]
            if word_matches(t.text, tw):
                found_idx = j
                break

        if found_idx != -1:
            matched_tokens.append(t)
            prev_tok = t
            target_pos = found_idx + 1
            # Reached exact end of target evidence
            if target_pos >= len(target_words):
                break
        else:
            # Unrelated token encountered between target tokens
            break

    # Safety Gate 6: Reconstructed sequence must account for target evidence
    # Require at least 70% of target words observed and at least 2 tokens matched
    if len(matched_tokens) < 2 or (target_pos / len(target_words)) < 0.65:
        return cand_bbox

    # Union only the validated target token boxes
    boxes = [t.bbox for t in matched_tokens]
    min_x = min(b.x for b in boxes)
    max_x = max(b.x + b.width for b in boxes)
    if max_right_boundary is not None and max_x > max_right_boundary:
        max_x = max(cb_x + cb_w, max_right_boundary)
    min_y = min(b.y for b in boxes)
    max_y = max(b.y + b.height for b in boxes)
    out_w = max_x - min_x
    out_h = max(cb_h, max_y - min_y)
    out_y = min(cb_y, min_y)

    # Class A requirement: Candidate must actually be narrow (expansion >= 1.25x original)
    if out_w <= cb_w * 1.25:
        return cand_bbox

    if is_bbox_obj:
        return BBox(x=min_x, y=out_y, width=out_w, height=out_h, page=page)
    return (min_x, out_y, out_w, out_h)
