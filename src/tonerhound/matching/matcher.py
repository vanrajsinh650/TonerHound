"""Matching algorithms for TonerHound: Exact, Normalized, and Fuzzy."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentToken
from tonerhound.normalization.normalizers import (
    detect_checkbox_state,
    is_date_equal,
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
        """Tier 1: Search exact string / phrase in document using sublinear inverted line/word index."""
        if not query or not query.strip():
            return []

        norm_query = normalize_unicode_and_case(query).text.strip()
        if not norm_query:
            return []

        # Prevent scanning whole document for short 1-2 char queries without page hint
        if len(norm_query) <= 2 and page_hint is None and len(self.index.pages) > 1:
            return []

        clean_query = norm_query.strip(" -.,;:_()[]{}/'\"")
        candidates: list[MatchCandidate] = []
        seen_lines: set[tuple[int, int]] = set()

        # Step 1: Identify candidate visual lines to inspect
        lines_to_check: list[tuple[int, int]] = []
        if page_hint is not None:
            page = self.index.get_page(page_hint)
            if page:
                lines_to_check = [(page_hint, l.line_index) for l in page.lines]
        else:
            q_words = [
                w.strip(" -.,;:_()[]{}/'\"")
                for w in norm_query.split()
                if len(w.strip(" -.,;:_()[]{}/'\"")) >= 2
            ]
            if q_words:
                postings = [self.index._lines_by_token.get(w, []) for w in q_words]
                non_empty = [p for p in postings if p]
                if non_empty:
                    # Pick the rarest word's posting list
                    lines_to_check = list(min(non_empty, key=len))
            elif len(self.index.pages) <= 10:
                for p in self.index.pages:
                    lines_to_check.extend([(p.page_number, l.line_index) for l in p.lines])

        # Step 2: Check candidate lines
        for p_num, l_idx in lines_to_check:
            page = self.index.get_page(p_num)
            if not page or l_idx >= len(page.lines):
                continue
            line = page.lines[l_idx]
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
                seen_lines.add((p_num, l_idx))
            elif clean_query and (norm_query in norm_line or clean_query in norm_line):
                matched = self._find_token_subsequence(line.tokens, clean_query)
                if matched:
                    ub = union_bbox_list([t.bbox for t in matched])
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
                        seen_lines.add((p_num, l_idx))

        # Step 2b: Multi-line token subsequence check across consecutive lines
        if not candidates and len(norm_query.split()) >= 2:
            multi_cands = self._find_multiline_candidates(norm_query, clean_query, lines_to_check, page_hint=page_hint)
            candidates.extend(multi_cands)

        # Step 3: Fallback to token indexes if not found in candidate lines
        if not candidates:
            matching_tokens = self.index._token_index.get(norm_query, [])
            if not matching_tokens and clean_query:
                matching_tokens = self.index._stem_token_index.get(clean_query, [])

            if matching_tokens:
                for t in matching_tokens:
                    if page_hint is not None and t.page != page_hint:
                        continue
                    candidates.append(
                        MatchCandidate(
                            page=t.page,
                            bbox=t.bbox,
                            tokens=(t,),
                            matched_text=t.text,
                            match_type="exact",
                            raw_similarity=1.0,
                            line_index=t.line_index,
                        )
                    )

        # Step 4: Page-wide substring search fallback for small documents or hint page
        if not candidates and (page_hint is not None or len(self.index.pages) <= 10):
            target_pages = [page_hint] if page_hint is not None else [p.page_number for p in self.index.pages]
            for p_num in target_pages:
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
        """Tier 2a: Locate numeric values via sublinear O(1) inverted numeric index."""
        target_num = parse_numeric_value(value)
        if target_num is None:
            return []

        key = round(target_num, 6)
        raw_matches = list(self.index._numeric_index.get(key, []))

        # Check integer equivalence (e.g. 1420.0 vs 1420)
        if not raw_matches and abs(target_num) > 1e-4:
            int_key = round(float(round(target_num)), 6)
            if int_key != key:
                raw_matches.extend(self.index._numeric_index.get(int_key, []))

        # Tolerance fallback if still empty and small index
        if not raw_matches and len(self.index._numeric_index) <= 20000:
            for k, entries in self.index._numeric_index.items():
                if abs(k - key) <= max(1e-5, abs(key) * 1e-5):
                    raw_matches.extend(entries)

        if not raw_matches:
            return []

        candidates: list[MatchCandidate] = []
        val_str = str(value).strip()

        for token, p_num, line_idx in raw_matches:
            if page_hint is not None and p_num != page_hint:
                continue

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
        """Tier 2b: Locate date values via sublinear inverted date index."""
        target_date = parse_date_value(value)
        if target_date is None:
            return []

        date_key = target_date.strftime("%Y-%m-%d")
        raw_matches = list(self.index._date_index.get(date_key, []))

        # If not indexed yet, check visual lines on target pages for small docs
        if not raw_matches:
            pages = [page_hint] if (page_hint and self.index.get_page(page_hint)) else [p.page_number for p in self.index.pages]
            if len(pages) <= 10 or page_hint is not None:
                for p_num in pages:
                    page = self.index.get_page(p_num)
                    if not page:
                        continue
                    for line in page.lines:
                        line_tokens = line.tokens
                        n_tok = len(line_tokens)
                        for window_size in range(1, min(5, n_tok + 1)):
                            for start_i in range(n_tok - window_size + 1):
                                window = line_tokens[start_i : start_i + window_size]
                                combined_text = " ".join(t.text for t in window)
                                if is_date_equal(target_date, combined_text):
                                    raw_matches.append((p_num, line.line_index, tuple(window), combined_text))

        candidates: list[MatchCandidate] = []
        for p_num, line_idx, sub_tokens, min_text in raw_matches:
            if page_hint is not None and p_num != page_hint:
                continue

            ub = union_bbox_list([t.bbox for t in sub_tokens])
            if ub and not any(c.page == p_num and c.bbox.iou(ub) >= 0.50 for c in candidates):
                candidates.append(
                    MatchCandidate(
                        page=p_num,
                        bbox=ub,
                        tokens=tuple(sub_tokens),
                        matched_text=min_text,
                        match_type="normalized_date",
                        raw_similarity=1.0,
                        line_index=line_idx,
                    )
                )

        return candidates

    def find_fuzzy_candidates(
        self,
        query: str,
        threshold: float = 0.80,
        page_hint: int | None = None,
    ) -> list[MatchCandidate]:
        """Tier 3: Fuzzy sequence alignment using n-gram inverted index pruning."""
        if not query or len(query.strip()) < 4:
            return []

        norm_query = normalize_unicode_and_case(query).text.strip()
        if not norm_query:
            return []

        line_overlap_counts: dict[tuple[int, int], int] = defaultdict(int)
        clean_q = norm_query.strip(" -.,;:_()[]{}/'\"")
        q_ngrams = [clean_q[i : i + 3] for i in range(len(clean_q) - 2)]

        for ng in q_ngrams:
            for p_num, l_idx in self.index._ngram_to_lines.get(ng, ()):
                if page_hint is not None and p_num != page_hint:
                    continue
                line_overlap_counts[(p_num, l_idx)] += 1

        # Also check stem words
        q_words = [
            w.strip(" -.,;:_()[]{}/'\"")
            for w in norm_query.split()
            if len(w.strip(" -.,;:_()[]{}/'\"")) >= 3
        ]
        for w in q_words:
            for p_num, l_idx in self.index._lines_by_token.get(w, ()):
                if page_hint is not None and p_num != page_hint:
                    continue
                line_overlap_counts[(p_num, l_idx)] += 3

        if not line_overlap_counts:
            # Fallback if query has no matching n-grams: only scan lines on page_hint or small documents (<= 10 pages)
            if page_hint is not None or len(self.index.pages) <= 10:
                target_pages = [page_hint] if page_hint else [p.page_number for p in self.index.pages]
                for p_num in target_pages:
                    p = self.index.get_page(p_num)
                    if p:
                        for line in p.lines:
                            line_overlap_counts[(p_num, line.line_index)] = 1
            else:
                return []

        # Select top-50 candidate lines by overlap
        top_lines = sorted(line_overlap_counts.items(), key=lambda item: item[1], reverse=True)[:50]

        candidates: list[MatchCandidate] = []
        for (p_num, line_idx), _score in top_lines:
            page = self.index.get_page(p_num)
            if not page or line_idx >= len(page.lines):
                continue
            line = page.lines[line_idx]
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
        if n_tok == 0:
            return [], 0.0

        norm_tokens = [normalize_unicode_and_case(t.text).text.strip() for t in tokens]
        target_words = norm_target.split()
        target_word_count = max(1, len(target_words))

        min_win = max(1, target_word_count - 1)
        max_win = min(n_tok, target_word_count + 2)

        for win_size in range(min_win, max_win + 1):
            for start_i in range(n_tok - win_size + 1):
                sub_text = " ".join(norm_tokens[start_i : start_i + win_size])
                sim = fuzz.ratio(norm_target, sub_text) / 100.0
                if sim > best_sim:
                    best_sim = sim
                    best_sub = tokens[start_i : start_i + win_size]

        return best_sub, best_sim

    def _find_token_subsequence(
        self, tokens: list[DocumentToken], norm_target: str
    ) -> list[DocumentToken]:
        """Locate contiguous sublist of tokens whose concatenation matches norm_target."""
        n_tok = len(tokens)
        if n_tok == 0:
            return []

        def _clean_p(s: str) -> str:
            return re.sub(r"[\s,.;:\-_()]+", " ", s).strip()

        clean_target = _clean_p(norm_target)
        norm_tokens = [normalize_unicode_and_case(t.text).text.strip() for t in tokens]
        target_words = norm_target.split()
        target_word_count = max(1, len(target_words))
        target_digits = re.sub(r"\D+", "", clean_target)

        # Prioritize windows matching target word count +/- 2
        min_win = max(1, target_word_count - 1)
        max_win = min(n_tok, target_word_count + 3)

        for window_size in range(min_win, max_win + 1):
            for start_i in range(n_tok - window_size + 1):
                sub_norm = " ".join(norm_tokens[start_i : start_i + window_size])
                if sub_norm == norm_target or _clean_p(sub_norm) == clean_target:
                    return tokens[start_i : start_i + window_size]
                if len(target_digits) >= 5 and re.sub(r"\D+", "", sub_norm) == target_digits:
                    return tokens[start_i : start_i + window_size]

        # Secondary search for edge cases
        for window_size in range(1, min_win):
            for start_i in range(n_tok - window_size + 1):
                sub_norm = " ".join(norm_tokens[start_i : start_i + window_size])
                if sub_norm == norm_target or _clean_p(sub_norm) == clean_target:
                    return tokens[start_i : start_i + window_size]
                if len(target_digits) >= 5 and re.sub(r"\D+", "", sub_norm) == target_digits:
                    return tokens[start_i : start_i + window_size]

        # Fallback pass: space-elided and merged-token comparison (e.g. TARGETEXTERMINATING, POBOX5300)
        clean_tgt_alnum = re.sub(r"[^a-z0-9]+", "", norm_target.lower())
        if len(clean_tgt_alnum) >= 3:
            for window_size in range(1, min(n_tok, target_word_count + 4) + 1):
                for start_i in range(n_tok - window_size + 1):
                    sub = tokens[start_i : start_i + window_size]
                    sub_alnum = re.sub(r"[^a-z0-9]+", "", "".join(t.text for t in sub).lower())
                    if sub_alnum == clean_tgt_alnum:
                        return sub
                    if sub_alnum and clean_tgt_alnum:
                        cov = min(len(sub_alnum), len(clean_tgt_alnum)) / max(len(sub_alnum), len(clean_tgt_alnum))
                        if (clean_tgt_alnum in sub_alnum or sub_alnum in clean_tgt_alnum) and cov >= 0.70:
                            return sub

        return []

    def _find_multiline_candidates(
        self,
        norm_query: str,
        clean_query: str,
        lines_to_check: list[tuple[int, int]],
        page_hint: int | None = None,
    ) -> list[MatchCandidate]:
        """Find multi-line exact matches across consecutive visual lines."""
        q_words = norm_query.split()
        if len(q_words) < 2:
            return []

        pages_to_check: set[int] = set()
        if page_hint is not None:
            pages_to_check.add(page_hint)
        elif lines_to_check:
            pages_to_check.update(p for p, _ in lines_to_check)
        elif len(self.index.pages) <= 10:
            pages_to_check.update(p.page_number for p in self.index.pages)

        candidates: list[MatchCandidate] = []
        for p_num in pages_to_check:
            page = self.index.get_page(p_num)
            if not page or len(page.lines) < 2:
                continue

            n_lines = len(page.lines)
            for i in range(n_lines - 1):
                for span_len in (2, 3):
                    if i + span_len > n_lines:
                        continue
                    span_lines = page.lines[i : i + span_len]
                    
                    # Find candidate start tokens in first line
                    start_tokens = [t for t in span_lines[0].tokens if q_words[0] in t.text.lower()]
                    token_sets_to_test = []
                    
                    # If column start token found, construct column-bounded token set
                    for st in start_tokens:
                        col_tokens: list[DocumentToken] = []
                        for l in span_lines:
                            col_tokens.extend(t for t in l.tokens if abs(t.bbox.x - st.bbox.x) <= 0.40)
                        token_sets_to_test.append(col_tokens)
                    
                    # Also test full span tokens
                    all_span_tokens: list[DocumentToken] = []
                    for l in span_lines:
                        all_span_tokens.extend(l.tokens)
                    token_sets_to_test.append(all_span_tokens)

                    for test_tokens in token_sets_to_test:
                        comb_text = " ".join(t.text for t in test_tokens)
                        norm_comb = normalize_unicode_and_case(comb_text).text.strip()
                        if q_words[0] in norm_comb and q_words[-1] in norm_comb:
                            matched = self._find_token_subsequence(test_tokens, clean_query or norm_query)
                            if matched:
                                ub = union_bbox_list([t.bbox for t in matched])
                                if ub:
                                    candidates.append(
                                        MatchCandidate(
                                            page=p_num,
                                            bbox=ub,
                                            tokens=tuple(matched),
                                            matched_text=" ".join(t.text for t in matched),
                                            match_type="exact",
                                            raw_similarity=1.0,
                                            line_index=span_lines[0].line_index,
                                        )
                                    )
                                    break
                    if candidates:
                        break

        return candidates

    def find_boolean_candidates(
        self,
        value: Any,
        field_name: str | None = None,
        page_hint: int | None = None,
        field_context: str | None = None,
    ) -> list[MatchCandidate]:
        """Locate checkbox glyph or token for a boolean field."""
        if value is None:
            return []
        target_bool = bool(value) if isinstance(value, bool) else (str(value).lower() in ("true", "yes", "1"))

        pages = (
            [page_hint]
            if (page_hint and self.index.get_page(page_hint))
            else [p.page_number for p in self.index.pages]
        )

        context_words: set[str] = set()
        if field_name:
            clean_name = field_name.split(".")[-1].split("[")[0]
            context_words.update(clean_name.lower().replace("_", " ").split())
        if field_context:
            context_words.update(field_context.lower().split())

        stop_words = {
            "true", "false", "yes", "no", "the", "and", "for", "box", "item", "check",
            "is", "has", "flag", "part", "sec", "to", "of", "in", "a", "an",
        }
        context_words = {w for w in context_words if len(w) >= 2 and w not in stop_words}

        candidates: list[MatchCandidate] = []
        for p_num in pages:
            page = self.index.get_page(p_num)
            if not page:
                continue

            kw_tokens: list[tuple[DocumentToken, float]] = []
            for t in page.tokens:
                norm_t = t.text.lower().strip(" -.,;:_()[]{}/'\"")
                if norm_t in context_words:
                    kw_tokens.append((t, 2.0))
                elif any(kw in norm_t for kw in context_words if len(kw) >= 4):
                    kw_tokens.append((t, 1.0))

            if not kw_tokens:
                for line in page.lines:
                    line_words = set(line.norm_text.lower().split())
                    if context_words & line_words:
                        if line.tokens:
                            kw_tokens.append((line.tokens[0], 0.5))

            if not kw_tokens and len(pages) <= 2:
                kw_tokens = [(t, 0.1) for t in page.tokens[:20]]

            for kw_t, kw_score in kw_tokens:
                kw_cy = kw_t.bbox.y + kw_t.bbox.height / 2.0
                for t in page.tokens:
                    t_cy = t.bbox.y + t.bbox.height / 2.0
                    if abs(t_cy - kw_cy) <= max(0.012, kw_t.bbox.height * 1.5):
                        cb_state = detect_checkbox_state(t.text)
                        if cb_state is not None and cb_state == target_bool:
                            dist_x = abs(t.bbox.x - kw_t.bbox.x)
                            if dist_x <= 0.35:
                                box = t.bbox
                                if box.width < 0.018:
                                    pad_w = (0.022 - box.width) / 2.0
                                    box = BBox(
                                        x=max(0.0, box.x - pad_w),
                                        y=box.y,
                                        width=0.022,
                                        height=box.height,
                                        page=box.page,
                                    )
                                box = box.align_to_line_height(target_height=0.015)

                                sim = kw_score * 5.0 - dist_x * 10.0
                                candidates.append(
                                    MatchCandidate(
                                        page=p_num,
                                        bbox=box,
                                        tokens=(t,),
                                        matched_text=t.text,
                                        match_type="boolean",
                                        raw_similarity=sim,
                                        line_index=t.line_index,
                                    )
                                )

        unique_cands: list[MatchCandidate] = []
        for c in sorted(candidates, key=lambda x: x.raw_similarity, reverse=True):
            if not any(c.page == u.page and c.bbox.iou(u.bbox) >= 0.7 for u in unique_cands):
                unique_cands.append(c)

        return unique_cands[:10]

