"""Matching algorithms for TonerHound: Exact, Normalized, and Fuzzy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentToken, VisualLine
from tonerhound.normalization.normalizers import (
    clean_currency_and_numbers,
    is_date_equal,
    is_number_equal,
    normalize_unicode_and_case,
    parse_date_value,
    parse_numeric_value,
)


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    """A located candidate occurrence of the evidence in the document."""

    page: int
    bbox: BBox
    tokens: tuple[DocumentToken, ...]
    matched_text: str
    match_type: str  # "exact", "normalized_number", "normalized_date", "fuzzy"
    raw_similarity: float  # 0.0 to 1.0
    line_index: int = 0

    @property
    def is_multi_line(self) -> bool:
        if not self.tokens:
            return False
        return len({t.line_index for t in self.tokens}) > 1


class EvidenceMatcher:
    """Multi-tiered evidence matcher across document index."""

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index

    def find_exact_candidates(
        self,
        query: str,
        page_hint: int | None = None,
    ) -> list[MatchCandidate]:
        """Tier 1: Search exact string / phrase in document."""
        if not query or not query.strip():
            return []

        norm_query = normalize_unicode_and_case(query).text.strip()
        candidates: list[MatchCandidate] = []

        pages = [page_hint] if (page_hint and self.index.get_page(page_hint)) else [p.page_number for p in self.index.pages]

        for p_num in pages:
            page = self.index.get_page(p_num)
            if not page:
                continue

            # Check exact matches on visual lines first
            for line in page.lines:
                norm_line = normalize_unicode_and_case(line.text).text.strip()
                if norm_query == norm_line:
                    candidates.append(
                        MatchCandidate(
                            page=p_num,
                            bbox=line.bbox,
                            tokens=tuple(line.tokens),
                            matched_text=line.text,
                            match_type="exact",
                            raw_similarity=1.0,
                            line_index=line.line_index,
                        )
                    )
                elif norm_query in norm_line:
                    # Find matching subset of tokens in line
                    matched = self._find_token_subsequence(line.tokens, norm_query)
                    if matched:
                        boxes = [t.bbox for t in matched]
                        ub = union_bbox_list(boxes)
                        if ub:
                            candidates.append(
                                MatchCandidate(
                                    page=p_num,
                                    bbox=ub,
                                    tokens=tuple(matched),
                                    matched_text=" ".join(t.text for t in matched),
                                    match_type="exact",
                                    raw_similarity=1.0,
                                    line_index=line.line_index,
                                )
                            )

            # Fallback to page-wide search if not found in single lines
            if not candidates:
                page_matches = self.index.search_exact(query, page=p_num)
                for box, text in page_matches:
                    candidates.append(
                        MatchCandidate(
                            page=p_num,
                            bbox=box,
                            tokens=(),
                            matched_text=text,
                            match_type="exact",
                            raw_similarity=1.0,
                        )
                    )

        return candidates

    def find_normalized_numeric_candidates(
        self,
        value: Any,
        page_hint: int | None = None,
    ) -> list[MatchCandidate]:
        """Tier 2a: Locate numeric values regardless of currency or punctuation formatting."""
        target_num = parse_numeric_value(value)
        if target_num is None:
            return []

        candidates: list[MatchCandidate] = []
        pages = [page_hint] if (page_hint and self.index.get_page(page_hint)) else [p.page_number for p in self.index.pages]

        for p_num in pages:
            page = self.index.get_page(p_num)
            if not page:
                continue

            for line in page.lines:
                for token in line.tokens:
                    token_num = parse_numeric_value(token.text)
                    if token_num is not None and is_number_equal(target_num, token_num):
                        candidates.append(
                            MatchCandidate(
                                page=p_num,
                                bbox=token.bbox,
                                tokens=(token,),
                                matched_text=token.text,
                                match_type="normalized_number",
                                raw_similarity=1.0,
                                line_index=line.line_index,
                            )
                        )

        return candidates

    def find_normalized_date_candidates(
        self,
        value: Any,
        page_hint: int | None = None,
    ) -> list[MatchCandidate]:
        """Tier 2b: Locate date values across varied surface calendar formats."""
        target_date = parse_date_value(value)
        if target_date is None:
            return []

        candidates: list[MatchCandidate] = []
        pages = [page_hint] if (page_hint and self.index.get_page(page_hint)) else [p.page_number for p in self.index.pages]

        for p_num in pages:
            page = self.index.get_page(p_num)
            if not page:
                continue

            for line in page.lines:
                # Check single tokens or sliding windows of 2-4 tokens
                line_tokens = line.tokens
                n_tok = len(line_tokens)
                for window_size in range(1, min(5, n_tok + 1)):
                    for start_i in range(n_tok - window_size + 1):
                        window = line_tokens[start_i : start_i + window_size]
                        combined_text = " ".join(t.text for t in window)
                        if is_date_equal(target_date, combined_text):
                            ub = union_bbox_list([t.bbox for t in window])
                            if ub:
                                candidates.append(
                                    MatchCandidate(
                                        page=p_num,
                                        bbox=ub,
                                        tokens=tuple(window),
                                        matched_text=combined_text,
                                        match_type="normalized_date",
                                        raw_similarity=1.0,
                                        line_index=line.line_index,
                                    )
                                )

        return candidates

    def find_fuzzy_candidates(
        self,
        query: str,
        threshold: float = 0.80,
        page_hint: int | None = None,
    ) -> list[MatchCandidate]:
        """Tier 3: Fuzzy sequence alignment for OCR errors or minor textual drift."""
        if not query or len(query.strip()) < 3:
            return []

        norm_query = normalize_unicode_and_case(query).text.strip()
        candidates: list[MatchCandidate] = []
        pages = [page_hint] if (page_hint and self.index.get_page(page_hint)) else [p.page_number for p in self.index.pages]

        for p_num in pages:
            page = self.index.get_page(p_num)
            if not page:
                continue

            for line in page.lines:
                norm_line = normalize_unicode_and_case(line.text).text.strip()
                sim = fuzz.partial_ratio(norm_query, norm_line) / 100.0
                if sim >= threshold:
                    # Approximate token subset in line
                    candidates.append(
                        MatchCandidate(
                            page=p_num,
                            bbox=line.bbox,
                            tokens=tuple(line.tokens),
                            matched_text=line.text,
                            match_type="fuzzy",
                            raw_similarity=sim,
                            line_index=line.line_index,
                        )
                    )

        return candidates

    def _find_token_subsequence(
        self, tokens: list[DocumentToken], norm_target: str
    ) -> list[DocumentToken]:
        """Locate contiguous sublist of tokens whose concatenation matches norm_target."""
        for window_size in range(1, len(tokens) + 1):
            for start_i in range(len(tokens) - window_size + 1):
                window = tokens[start_i : start_i + window_size]
                text = normalize_unicode_and_case(" ".join(t.text for t in window)).text.strip()
                if text == norm_target:
                    return window
        return []
