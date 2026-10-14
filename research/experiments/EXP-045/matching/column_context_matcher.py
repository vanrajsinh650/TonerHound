"""EXP-045 Algorithm 2: Column-Context Matching (for SUB_COLUMN_CONTEXT).

Detects column header cells directly above candidate tokens (same vertical column band)
and scores the header against the schema field name using token-set similarity.
Rejects matches where header similarity is below 0.50.
"""

from __future__ import annotations

import re
from typing import Any
from rapidfuzz import fuzz

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def normalize_field_name(field_path: str) -> str:
    """Extract and normalize the terminal property name from a field path."""
    prop = field_path.split(".")[-1]
    # Remove array indices e.g. [0]
    prop = re.sub(r"\[\d+\]", "", prop)
    # Split camelCase or snake_case into space-separated words
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", prop).replace("_", " ").strip()
    return words.lower()


def find_column_header(
    cand_tok: DocumentToken,
    all_tokens: list[DocumentToken],
    max_vertical_lookback: float = 250.0,
) -> DocumentToken | None:
    """Find the column header token directly above a candidate token."""
    cx1, cy1, cw, ch = cand_tok.bbox.to_coco()
    cx2 = cx1 + cw
    c_mid_x = (cx1 + cx2) / 2.0

    best_header: DocumentToken | None = None
    min_dist = float("inf")

    for tok in all_tokens:
        if tok is cand_tok:
            continue
        tx1, ty1, tw, th = tok.bbox.to_coco()
        tx2 = tx1 + tw
        # Vertical alignment (x overlap)
        x_overlap = max(0.0, min(cx2, tx2) - max(cx1, tx1))
        if x_overlap > 0 or abs(c_mid_x - (tx1 + tx2) / 2.0) <= max(15.0, cw * 0.5):
            # Must be above candidate within max_vertical_lookback
            if ty1 < cy1 and (cy1 - ty1) <= max_vertical_lookback:
                dist = cy1 - ty1
                if dist < min_dist:
                    min_dist = dist
                    best_header = tok

    return best_header


def match_column_context(
    target_val: Any,
    page_obj: DocumentPage,
    field_path: str,
    min_header_similarity: float = 0.50,
) -> tuple[BBox | None, str | None]:
    """Execute Stage 1 Column-Context Matching.
    
    Args:
        target_val: Value to ground
        page_obj: Page to search
        field_path: Schema field path (e.g., holdings[0].security_name)
        min_header_similarity: Threshold for column header similarity (default: 0.50)
    
    Returns:
        (matched_bbox, matched_text) or (None, None)
    """
    if target_val is None or not page_obj or not page_obj.tokens:
        return None, None

    val_str = str(target_val).strip()
    if not val_str:
        return None, None

    # Search candidates matching val_str
    cand_tokens: list[DocumentToken] = []
    for tok in page_obj.tokens:
        if val_str.lower() == tok.text.lower() or (len(val_str) >= 3 and val_str.lower() in tok.text.lower()):
            cand_tokens.append(tok)

    if not cand_tokens:
        return None, None

    if len(cand_tokens) == 1:
        return cand_tokens[0].bbox, cand_tokens[0].text

    norm_field = normalize_field_name(field_path)
    best_cand: PageToken | None = None
    best_sim = -1.0

    for cand in cand_tokens:
        header_tok = find_column_header(cand, page_obj.tokens)
        if not header_tok:
            continue
        sim = fuzz.token_set_ratio(norm_field, header_tok.text.lower()) / 100.0
        if sim > best_sim:
            best_sim = sim
            best_cand = cand

    if best_cand and best_sim >= min_header_similarity:
        return best_cand.bbox, best_cand.text

    return None, None
