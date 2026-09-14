"""Matching algorithms for TonerHound: Exact, Normalized, and Fuzzy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentToken
from tonerhound.normalization.normalizers import (
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
        self._numeric_cache: dict[int, list[tuple[DocumentToken, float, int]]] = {}

    def _get_page_numeric_tokens(self, page_num: int) -> list[tuple[DocumentToken, float, int]]:
        if page_num not in self._numeric_cache:
            page = self.index.get_page(page_num)
            cached: list[tuple[DocumentToken, float, int]] = []
            if page:
                for line in page.lines:
                    for token in line.tokens:
                        num = parse_numeric_value(token.text)
                        if num is not None:
                            cached.append((token, num, line.line_index))
            self._numeric_cache[page_num] = cached
        return self._numeric_cache[page_num]

    def find_exact_candidates(
        self,
        query: str,
        page_hint: int | None = None,
    ) -> list[MatchCandidate]:
        """Tier 1: Search exact string / phrase in document."""
        if not query or not query.strip():
            return []

        norm_query = normalize_unicode_and_case(query).text.strip()
        if not norm_query:
            return []

        # Prevent scanning whole document for short 1-2 char queries without page hint
        if len(norm_query) <= 2 and page_hint is None and len(self.index.pages) > 1:
            return []

        candidates: list[MatchCandidate] = []
        pages = [page_hint] if (page_hint and self.index.get_page(page_hint)) else [p.page_number for p in self.index.pages]

        for p_num in pages:
            page = self.index.get_page(p_num)
            if not page:
                continue

            # Check exact matches on visual lines first
            for line in page.lines:
                norm_line = line.norm_text
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
                else:
                    clean_query = norm_query.strip(" -.,;:_")
                    if clean_query and (norm_query in norm_line or clean_query in norm_line):
                        # Find matching subset of tokens in line
                        matched = self._find_token_subsequence(line.tokens, clean_query)
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
                if len(norm_query) <= 2:
                    matching_tokens = [t for t in self.index._token_index.get(norm_query, []) if t.page == p_num]
                    for t in matching_tokens:
                        candidates.append(
                            MatchCandidate(
                                page=p_num,
                                bbox=t.bbox,
                                tokens=(t,),
                                matched_text=t.text,
                                match_type="exact",
                                raw_similarity=1.0,
                                line_index=t.line_index,
                            )
                        )
                else:
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

        # Prevent full document scan for numbers without page hint on large docs
        if page_hint is None and len(self.index.pages) > 10:
            return []

        candidates: list[MatchCandidate] = []
        pages = [page_hint] if (page_hint and self.index.get_page(page_hint)) else [p.page_number for p in self.index.pages]

        val_str = str(value).strip()
        for p_num in pages:
            for token, token_num, line_idx in self._get_page_numeric_tokens(p_num):
                if is_number_equal(target_num, token_num):
                    tok_bbox = token.bbox
                    if val_str in token.text and len(val_str) < len(token.text):
                        idx = token.text.find(val_str)
                        tok_bbox = tok_bbox.sub_bbox(idx, idx + len(val_str), len(token.text))

                    candidates.append(
                        MatchCandidate(
                            page=p_num,
                            bbox=tok_bbox,
                            tokens=(token,),
                            matched_text=token.text,
                            match_type="normalized_number",
                            raw_similarity=1.0,
                            line_index=line_idx,
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

        # Prevent full document scan for dates without page hint on large docs
        if page_hint is None and len(self.index.pages) > 10:
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
                            # Shrink window to minimal sub-window that still matches the date
                            sub = list(window)
                            while len(sub) > 1 and is_date_equal(target_date, " ".join(t.text for t in sub[1:])):
                                sub.pop(0)
                            while len(sub) > 1 and is_date_equal(target_date, " ".join(t.text for t in sub[:-1])):
                                sub.pop()

                            min_text = " ".join(t.text for t in sub)
                            ub = union_bbox_list([t.bbox for t in sub])
                            if ub and not any(c.page == p_num and c.bbox.iou(ub) >= 0.95 for c in candidates):
                                candidates.append(
                                    MatchCandidate(
                                        page=p_num,
                                        bbox=ub,
                                        tokens=tuple(sub),
                                        matched_text=min_text,
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
        if not query or len(query.strip()) < 4:
            return []

        # Prevent catastrophic O(N_fields * N_lines) scans on large docs without a page hint
        if page_hint is None and len(self.index.pages) > 10:
            return []

        norm_query = normalize_unicode_and_case(query).text.strip()
        candidates: list[MatchCandidate] = []
        pages = [page_hint] if (page_hint and self.index.get_page(page_hint)) else [p.page_number for p in self.index.pages]

        for p_num in pages:
            page = self.index.get_page(p_num)
            if not page:
                continue

            for line in page.lines:
                norm_line = line.norm_text
                sim = fuzz.partial_ratio(norm_query, norm_line) / 100.0
                if sim >= threshold:
                    sub_toks, sub_sim = self._find_best_fuzzy_subsequence(line.tokens, norm_query)
                    if sub_toks and sub_sim >= threshold * 0.85:
                        ub = union_bbox_list([t.bbox for t in sub_toks])
                        if ub:
                            candidates.append(
                                MatchCandidate(
                                    page=p_num,
                                    bbox=ub,
                                    tokens=tuple(sub_toks),
                                    matched_text=" ".join(t.text for t in sub_toks),
                                    match_type="fuzzy",
                                    raw_similarity=max(sim, sub_sim),
                                    line_index=line.line_index,
                                )
                            )
                            continue

                    # Approximate token subset in line fallback
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

    def _find_best_fuzzy_subsequence(
        self, tokens: list[DocumentToken], norm_target: str
    ) -> tuple[list[DocumentToken], float]:
        """Find sub-sequence of tokens with highest fuzzy similarity to norm_target."""
        best_sub: list[DocumentToken] = []
        best_sim = 0.0
        n_tok = len(tokens)
        target_words = norm_target.split()
        target_word_count = max(1, len(target_words))

        min_win = max(1, target_word_count - 1)
        max_win = min(n_tok, target_word_count + 2)

        for win_size in range(min_win, max_win + 1):
            for start_i in range(n_tok - win_size + 1):
                sub = tokens[start_i : start_i + win_size]
                sub_text = normalize_unicode_and_case(" ".join(t.text for t in sub)).text.strip()
                sim = fuzz.ratio(norm_target, sub_text) / 100.0
                if sim > best_sim:
                    best_sim = sim
                    best_sub = sub

        return best_sub, best_sim

    def _find_token_subsequence(
        self, tokens: list[DocumentToken], norm_target: str
    ) -> list[DocumentToken]:
        """Locate contiguous sublist of tokens whose concatenation matches norm_target."""
        clean_target = norm_target.strip(" -.,;:_")
        for window_size in range(1, len(tokens) + 1):
            for start_i in range(len(tokens) - window_size + 1):
                window = tokens[start_i : start_i + window_size]
                text = normalize_unicode_and_case(" ".join(t.text for t in window)).text.strip()
                if text == norm_target or text.strip(" -.,;:_") == clean_target:
                    return window
        return []
