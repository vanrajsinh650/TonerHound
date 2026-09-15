"""OCR-aware candidate retrieval and row-level matching engine."""

from __future__ import annotations

import re
from typing import Any

from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentToken
from tonerhound.normalization.normalizers import (
    clean_currency_and_numbers,
    detect_checkbox_state,
    is_number_equal,
    parse_numeric_value,
)
from tonerhound.ocr.aligner import OCRSequenceAligner, slice_token_bbox
from tonerhound.ocr.normalizer import (
    clean_ocr_noise_span,
    ocr_canonical_numeric,
    ocr_similarity,
)


class OCRMatcher:
    """Specialized matcher for corrupted OCR text, numeric fields, and tabular records."""

    @staticmethod
    def match_row_field(
        field_path: str,
        value: Any,
        row_tokens: list[DocumentToken],
        field_context: str | None = None,
        min_similarity: float = 0.72,
    ) -> tuple[BBox, str, float] | None:
        """Resolve physical evidence for an extracted field among visual tokens of a single table row."""
        if value is None or not row_tokens:
            return None

        val_str = str(value).strip()
        if not val_str:
            return None

        is_bool = isinstance(value, bool) or (
            val_str.lower() in ("true", "false", "yes", "no")
            and any(k in field_path.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no", "contingent", "unliquidated", "disputed", "subject_to_offset"))
        )
        is_num = isinstance(value, (int, float)) and not isinstance(value, bool)

        # 1. Boolean checkbox resolution in row
        if is_bool:
            target_bool = bool(value) if isinstance(value, bool) else (val_str.lower() in ("true", "yes", "1"))
            for t in row_tokens:
                cb_state = detect_checkbox_state(t.text)
                if cb_state is not None and cb_state == target_bool:
                    return t.bbox, t.text, 0.95
                # Also check common OCR glyphs for checkboxes on scanned forms
                if not target_bool and t.text.strip().lower() in ("no", "o", "q", "d", "[]", "[ ]", "0"):
                    return t.bbox, t.text, 0.90
                if target_bool and t.text.strip().lower() in ("yes", "x", "[x]", "1"):
                    return t.bbox, t.text, 0.90

        # 2. Numeric / Currency / Identifier matching in row
        target_num = parse_numeric_value(value)
        if target_num is not None and not is_bool:
            # Pass 2a: Exact float match on token
            for t in row_tokens:
                tn = parse_numeric_value(t.text)
                if tn is not None and is_number_equal(tn, target_num):
                    # Clean punctuation from number bbox
                    _clean, s_idx, e_idx = clean_ocr_noise_span(t.text)
                    box = slice_token_bbox(t, s_idx, e_idx) if s_idx < e_idx else t.bbox
                    return box, t.text, 0.95

            # Pass 2b: Raw digits match (handling lost decimal periods, e.g. 2.1 vs 24 or 21, 3.11 vs 311., 121.50 vs 1215)
            digits_target = re.sub(r"\D", "", val_str)
            if len(digits_target) >= 2:
                for t in row_tokens:
                    t_digits = re.sub(r"\D", "", t.text)
                    if t_digits == digits_target:
                        _clean, s_idx, e_idx = clean_ocr_noise_span(t.text)
                        box = slice_token_bbox(t, s_idx, e_idx) if s_idx < e_idx else t.bbox
                        return box, t.text, 0.90

            # Pass 2c: Adjacent token pairing for currency (e.g. "$" and "6,867.62")
            for i in range(len(row_tokens) - 1):
                t1, t2 = row_tokens[i], row_tokens[i + 1]
                combo = t1.text + t2.text
                c_num = parse_numeric_value(combo)
                if c_num is not None and is_number_equal(c_num, target_num):
                    ub = union_bbox_list([t1.bbox, t2.bbox])
                    if ub:
                        return ub, combo, 0.95

        # 3. Alphanumeric / String sequence alignment in row
        if not is_bool and not is_num:
            sim, matched_tokens, box = OCRSequenceAligner.align_to_tokens(
                query=val_str,
                tokens=row_tokens,
                min_similarity=min_similarity,
            )
            if sim >= min_similarity and box is not None and matched_tokens:
                return box, " ".join(t.text for t in matched_tokens), sim

        return None
