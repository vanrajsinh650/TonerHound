"""Stage 20: Character-Span Sub-Slicer.

When a target value is a substring of a larger token (e.g. "1,250" in "$1,250.00"):
- Finds the substring span (start_idx, end_idx) inside the full token string
- Performs proportional linear interpolation along the horizontal axis
- Slices the token bbox to match the character span.
"""

from __future__ import annotations

import re
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentToken


def slice_token_by_char_span(
    token_text: str,
    token_bbox: BBox | tuple[float, float, float, float] | list[float],
    target_substring: str,
) -> BBox | None:
    """Proportionally slice token_bbox to cover only target_substring."""
    if not token_text or not target_substring:
        return None

    if isinstance(token_bbox, (list, tuple)):
        tb = BBox(token_bbox[0], token_bbox[1], token_bbox[2], token_bbox[3])
    else:
        tb = token_bbox

    # Clean/normalize for search
    idx = token_text.find(target_substring)
    if idx == -1:
        # Try case-insensitive search
        idx = token_text.lower().find(target_substring.lower())

    if idx == -1:
        return None

    n_chars = len(token_text)
    if n_chars == 0:
        return None

    span_len = len(target_substring)
    start_ratio = idx / n_chars
    end_ratio = (idx + span_len) / n_chars

    width = tb.width
    sliced_x0 = tb.x0 + width * start_ratio
    sliced_x1 = tb.x0 + width * end_ratio

    # Keep y0 and y1 unchanged
    return BBox.from_xyxy(sliced_x0, tb.y0, sliced_x1, tb.y1)


def slice_candidate_tokens(
    tokens: list[DocumentToken],
    target_value: str,
) -> BBox | None:
    """Check each token; if target is a substring, slice its bbox proportionally."""
    clean_target = target_value.strip()
    if not clean_target:
        return None

    for tok in tokens:
        txt = tok.text.strip()
        if clean_target in txt and len(clean_target) < len(txt):
            sliced = slice_token_by_char_span(txt, tok.bbox, clean_target)
            if sliced is not None:
                return sliced
    return None
