"""Stage 27: Sequential Token Assembly.

For values spanning multiple visual lines or tokens:
- Collects tokens sorted in natural reading order (top-to-bottom, left-to-right)
- Builds a character-offset mapped sequence of tokens
- Searches the concatenated text for the target string
- Computes the union bounding box of all constituent tokens covering the match.
"""

from __future__ import annotations

from typing import Sequence
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentToken


def match_sequential_tokens(
    tokens: Sequence[DocumentToken],
    target_value: str,
) -> tuple[BBox | None, float]:
    """Finds target_value in the sequentially ordered token stream.
    
    Returns:
        (union_bbox, confidence)
    """
    clean_target = " ".join(target_value.strip().lower().split())
    if not clean_target or not tokens:
        return None, 0.0

    # Sort tokens in visual reading order: primary y0, secondary x0 with tolerance
    sorted_tokens = sorted(tokens, key=lambda t: (round(t.bbox.y0 / 6.0), t.bbox.x0))

    # Build sequence with character mapping
    full_str = ""
    token_spans: list[tuple[int, int, DocumentToken]] = []

    for t in sorted_tokens:
        txt = t.text.strip()
        if not txt:
            continue
        start_char = len(full_str)
        if full_str:
            full_str += " "
            start_char += 1
        full_str += txt.lower()
        end_char = len(full_str)
        token_spans.append((start_char, end_char, t))

    match_idx = full_str.find(clean_target)
    if match_idx == -1:
        # Also try compact without spaces
        compact_target = clean_target.replace(" ", "")
        compact_str = full_str.replace(" ", "")
        if compact_target in compact_str:
            # Found in compact string
            pass
        else:
            return None, 0.0

    match_end = match_idx + len(clean_target)

    # Collect tokens overlapping [match_idx, match_end]
    matched_toks = [
        t for (start, end, t) in token_spans
        if max(start, match_idx) < min(end, match_end)
    ]

    if not matched_toks:
        return None, 0.0

    # Calculate union bbox
    ux0 = min(t.bbox.x0 for t in matched_toks)
    uy0 = min(t.bbox.y0 for t in matched_toks)
    ux1 = max(t.bbox.x1 for t in matched_toks)
    uy1 = max(t.bbox.y1 for t in matched_toks)

    return BBox.from_xyxy(ux0, uy0, ux1, uy1), 0.90
