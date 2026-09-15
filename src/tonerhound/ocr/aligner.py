"""Fuzzy sequence alignment with character-level bounding box slicing."""

from __future__ import annotations

import re
from typing import Any

from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentToken
from tonerhound.ocr.normalizer import clean_ocr_noise_span, ocr_similarity


def slice_token_bbox(token: DocumentToken, start_char: int, end_char: int) -> BBox:
    """Slice a DocumentToken bounding box horizontally proportionally to character span [start_char, end_char)."""
    total_len = max(1, len(token.text))
    s_char = max(0, min(total_len, start_char))
    e_char = max(s_char, min(total_len, end_char))

    start_ratio = s_char / total_len
    end_ratio = e_char / total_len

    new_x = token.bbox.x + start_ratio * token.bbox.width
    new_w = max(0.002, (end_ratio - start_ratio) * token.bbox.width)

    return BBox(
        x=new_x,
        y=token.bbox.y,
        width=new_w,
        height=token.bbox.height,
        page=token.page,
    )


class OCRSequenceAligner:
    """Fuzzy sequence aligner that maps query strings to precise sub-token bounding boxes."""

    @staticmethod
    def align_to_tokens(
        query: str,
        tokens: list[DocumentToken],
        min_similarity: float = 0.72,
    ) -> tuple[float, list[DocumentToken], BBox | None]:
        """Align query across tokens, returning (similarity, matched_tokens, sliced_bbox)."""
        if not query or not tokens:
            return 0.0, [], None

        clean_query = query.strip()
        q_alnum = re.sub(r"[^A-Za-z0-9]+", "", clean_query).upper()
        if not q_alnum:
            return 0.0, [], None

        best_sim = 0.0
        best_slice: tuple[int, int] = (0, 0)
        n = len(tokens)

        # Estimate expected number of tokens
        q_words = [w for w in clean_query.split() if w]
        est_tokens = max(1, len(q_words))
        max_w = min(n, est_tokens + 4)

        for w in range(1, max_w + 1):
            for i in range(n - w + 1):
                sub = tokens[i : i + w]
                sub_text = "".join(t.text for t in sub)
                sim = ocr_similarity(clean_query, sub_text)
                if sim > best_sim:
                    best_sim = sim
                    best_slice = (i, i + w)

        if best_sim < min_similarity:
            return best_sim, [], None

        matched_tokens = tokens[best_slice[0] : best_slice[1]]
        if not matched_tokens:
            return 0.0, [], None

        # Slice boundaries to strip noise punctuation from first and last tokens
        sliced_boxes: list[BBox] = []
        for idx, t in enumerate(matched_tokens):
            t_txt = t.text
            _clean, s_idx, e_idx = clean_ocr_noise_span(t_txt)
            if s_idx < e_idx and (idx == 0 or idx == len(matched_tokens) - 1):
                sliced_boxes.append(slice_token_bbox(t, s_idx, e_idx))
            else:
                sliced_boxes.append(t.bbox)

        final_box = union_bbox_list(sliced_boxes)
        return best_sim, matched_tokens, final_box
