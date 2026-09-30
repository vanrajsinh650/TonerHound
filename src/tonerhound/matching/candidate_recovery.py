"""EXP-013 Candidate Generation Recovery Engine.

Recovers evidence present in document geometry but absent or fragmented
in standard candidate generation.

Priority failure classes addressed:
1. Spaced-token numeric recovery (e.g. "3 3 . 3 3 3 3 %" -> 33.3333%)
2. OCR-fragmented numeric values (reconstruct split tokens/glyph fragments)
3. Split currency / percentage / date tokens (e.g. ["33.3333333", "%"])
4. Fragmented multi-token field values (interleaved rows / non-contiguous tokens)
5. Candidates skipped because one expected anchor token is missing (bounded span matching)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Any

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import DocumentToken, VisualLine

# Feature flag for EXP-028F Phase 1.2: Token-gated global fallback for long documents
ENABLE_GLOBAL_FALLBACK: bool = True
from tonerhound.normalization.normalizers import (
    clean_currency_and_numbers,
    normalize_unicode_and_case,
    parse_numeric_value,
)


@dataclass(frozen=True, slots=True)
class RecoveredDiagnostic:
    """Critical diagnostic record for every recovered candidate."""

    original_value: Any
    raw_document_tokens: tuple[str, ...]
    reconstructed_candidate: str
    candidate_bbox: tuple[float, float, float, float]
    normalization_path: str
    why_previously_missed: str
    page: int
    mechanism: str


class CandidateRecoveryEngine:
    """Modular candidate recovery engine for TonerHound."""

    def __init__(
        self,
        index: DocumentIndex,
        enable_spaced_numeric: bool = True,
        enable_fragmented_numeric: bool = True,
        enable_split_symbol: bool = True,
        enable_interleaved_tokens: bool = True,
        enable_bounded_spans: bool = True,
        max_candidate_per_field: int = 500,
        enable_global_fallback: bool | None = None,
    ) -> None:
        self.index = index
        self.enable_spaced_numeric = enable_spaced_numeric
        self.enable_fragmented_numeric = enable_fragmented_numeric
        self.enable_split_symbol = enable_split_symbol
        self.enable_interleaved_tokens = enable_interleaved_tokens
        self.enable_bounded_spans = enable_bounded_spans
        self.max_candidate_per_field = max_candidate_per_field
        self.enable_global_fallback = enable_global_fallback
        self.diagnostics: list[RecoveredDiagnostic] = []

    def _resolve_global_fallback(self) -> bool:
        """Resolve effective global fallback setting, respecting instance/module overrides."""
        if self.enable_global_fallback is not None:
            return self.enable_global_fallback
        return ENABLE_GLOBAL_FALLBACK

    def _find_candidate_pages(
        self,
        value: Any,
        evidence_text: str | None = None,
        field_name: str | None = None,
    ) -> set[int]:
        """Inverted-index pre-check (Safeguard 1): O(1) identification of candidate pages containing target tokens."""
        pages: set[int] = set()
        is_bool = isinstance(value, bool) or (
            isinstance(value, str)
            and value.strip().lower() in ("true", "false", "yes", "no")
            and any(k in (field_name or "").lower() for k in ("_box", "checkbox", "is_", "has_", "flag"))
        )
        target_num = parse_numeric_value(value)

        # 1. Numeric check against inverted indexes
        if target_num is not None:
            # Safeguard: Do not trigger global fallback for zero or single-digit integers (0..9)
            # which appear ubiquitously across all pages and create massive false-grounding risk.
            if abs(target_num) < 1e-4 or (target_num == int(target_num) and 0 <= abs(target_num) < 10):
                return set()

            key = round(target_num, 6)
            for tok, p_num, _ in self.index._numeric_index.get(key, ()):
                pages.add(p_num)

            # Integer equivalent
            if abs(target_num) > 1e-4:
                int_key = round(float(round(target_num)), 6)
                if int_key != key:
                    for tok, p_num, _ in self.index._numeric_index.get(int_key, ()):
                        pages.add(p_num)

            # String token lookup for numeric digits
            val_str = str(value).strip("$€£¥%,() ")
            if val_str:
                for tok in self.index._token_index.get(val_str, ()):
                    pages.add(tok.page)
                for tok in self.index._stem_token_index.get(val_str, ()):
                    pages.add(tok.page)

            # Check significant integer digits (e.g. 1420 in 1420.50)
            if abs(target_num) < 1e12:
                int_str = str(abs(int(round(target_num))))
                if len(int_str) >= 2:
                    for tok in self.index._token_index.get(int_str, ()):
                        pages.add(tok.page)
                    for tok in self.index._stem_token_index.get(int_str, ()):
                        pages.add(tok.page)
                    for p_num, _ in self.index._lines_by_token.get(int_str, ()):
                        pages.add(p_num)

            # Spaced/fragmented numeric fallback: check if unspaced digit sequence appears in normalized page text
            if not pages and abs(target_num) < 1e12:
                int_str = str(abs(int(round(target_num))))
                if len(int_str) >= 2:
                    for p_num, (norm_page_text, _) in self.index._page_normalized_text.items():
                        if int_str in norm_page_text.text.replace(" ", ""):
                            pages.add(p_num)

        # 2. String / Text token lookup
        query_str = evidence_text if evidence_text else (str(value) if isinstance(value, str) else None)
        if query_str and not is_bool:
            generic_stopwords = {
                "sole", "none", "defined", "other", "common", "stock", "com", "inc", "corp",
                "class", "shs", "shares", "the", "and", "of", "in", "to", "for", "with",
                "on", "at", "from", "by", "an", "as", "is", "are", "or", "that", "this", "it"
            }
            norm_q = normalize_unicode_and_case(query_str).text.strip()
            # If string is purely generic stopword, skip global fallback on long documents
            if norm_q.lower() in generic_stopwords and len(self.index.pages) > 10:
                return set()

            q_words = [
                w.strip(" ,.:;-/_'\"()[]{}*&#$€£¥")
                for w in norm_q.split()
                if len(w.strip(" ,.:;-/_'\"()[]{}*&#$€£¥")) >= 2
            ]
            distinctive_words = [
                w for w in q_words
                if len(w) >= 3 and w.lower() not in generic_stopwords
            ]
            lookup_words = distinctive_words if distinctive_words else q_words

            for w in lookup_words:
                for tok in self.index._token_index.get(w, ()):
                    pages.add(tok.page)
                for tok in self.index._stem_token_index.get(w, ()):
                    pages.add(tok.page)
                for p_num, _ in self.index._lines_by_token.get(w, ()):
                    pages.add(p_num)

            # Spaced string fallback: check if unspaced query appears in normalized page text
            if not pages and len(norm_q) >= 4 and norm_q.lower() not in generic_stopwords:
                q_no_space = "".join(norm_q.split())
                if len(q_no_space) >= 4:
                    for p_num, (norm_page_text, _) in self.index._page_normalized_text.items():
                        if q_no_space in norm_page_text.text.replace(" ", ""):
                            pages.add(p_num)

        return pages

    def _execute_recovery_passes(
        self,
        target_pages: list[int],
        value: Any,
        query_str: str | None,
        is_num: bool,
        is_bool: bool,
        has_exact_match: bool,
        start_time: float,
        timeout_sec: float,
    ) -> list[MatchCandidate]:
        """Execute recovery passes across designated pages with timeout (Safeguard 3) and cap (Safeguard 2)."""
        recovered: list[MatchCandidate] = []
        target_num = parse_numeric_value(value) if (is_num or (isinstance(value, str) and not is_bool and parse_numeric_value(value) is not None)) else None
        q_words = [w for w in query_str.strip().split() if len(w.strip(" ,.;:-_()[]{}'\"")) >= 2] if (query_str and not is_bool) else []

        for p_num in target_pages:
            # Safeguard 3: Hard timeout check per field
            if time.perf_counter() - start_time > timeout_sec:
                break
            # Safeguard 2: Candidate cap check
            if len(recovered) >= 200:
                break

            page = self.index.get_page(p_num)
            if not page:
                continue

            for line in page.lines:
                if time.perf_counter() - start_time > timeout_sec:
                    break

                # 1. Numeric passes
                if target_num is not None:
                    if self.enable_spaced_numeric:
                        recovered.extend(self._recover_spaced_numeric(line, target_num, value, p_num))
                    if self.enable_fragmented_numeric:
                        recovered.extend(self._recover_fragmented_numeric(line, target_num, value, p_num))
                    if self.enable_split_symbol:
                        recovered.extend(self._recover_split_symbol(line, target_num, value, p_num))

                # 2. String / multi-token passes
                if q_words:
                    if self.enable_interleaved_tokens and len(q_words) >= 2:
                        recovered.extend(self._recover_interleaved_tokens(line, q_words, query_str.strip(), value, p_num))
                    if self.enable_bounded_spans and len(q_words) >= 3 and not has_exact_match:
                        recovered.extend(self._recover_bounded_spans(line, q_words, query_str.strip(), value, p_num))

            # Safeguard 6: Early termination on high-confidence match
            if any(c.raw_similarity >= 0.95 for c in recovered):
                break

        return recovered

    def _deduplicate_and_rank(
        self,
        candidates: list[MatchCandidate],
        page_hint: int | None,
        is_global_fallback: bool,
    ) -> list[MatchCandidate]:
        """Deduplicate, penalize global candidates (Safeguard 5), and prioritize pages (Safeguard 4)."""
        unique: list[MatchCandidate] = []
        for c in sorted(
            candidates,
            key=lambda x: (x.raw_similarity, len(x.tokens), len(x.matched_text)),
            reverse=True,
        ):
            if not any(u.page == c.page and u.bbox.iou(c.bbox) >= 0.85 for u in unique):
                # Safeguard 5: Conservative 0.85 similarity multiplier for global fallback candidates
                if is_global_fallback and page_hint is not None and abs(c.page - page_hint) > 1:
                    c = MatchCandidate(
                        page=c.page,
                        bbox=c.bbox,
                        tokens=c.tokens,
                        matched_text=c.matched_text,
                        match_type=c.match_type,
                        raw_similarity=c.raw_similarity * 0.85,
                        line_index=c.line_index,
                    )
                unique.append(c)

        # Safeguard 4: Page priority ranking (page_hint > page_hint +/- 1 > global)
        if page_hint is not None and len(unique) > 1:
            def _priority_key(cand: MatchCandidate) -> tuple[int, float]:
                if cand.page == page_hint:
                    prio = 0
                elif abs(cand.page - page_hint) == 1:
                    prio = 1
                else:
                    prio = 2
                return (prio, -cand.raw_similarity)

            unique.sort(key=_priority_key)

        # Safeguard 2: Candidate cap (max 200 candidates)
        return unique[: min(200, self.max_candidate_per_field)]

    def recover(
        self,
        value: Any,
        page_hint: int | None = None,
        field_name: str | None = None,
        field_context: str | None = None,
        evidence_text: str | None = None,
        has_exact_match: bool = False,
        is_global_fallback: bool = False,
        original_page_hint: int | None = None,
        timeout_sec: float = 0.500,
    ) -> list[MatchCandidate]:
        """Run modular candidate recovery passes for a given extraction with global fallback."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return []

        start_time = time.perf_counter()
        is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
        is_bool = isinstance(value, bool) or (
            isinstance(value, str)
            and value.strip().lower() in ("true", "false", "yes", "no")
            and any(k in (field_name or "").lower() for k in ("_box", "checkbox", "is_", "has_", "flag"))
        )
        query_str = evidence_text if evidence_text else (str(value) if isinstance(value, str) else None)
        global_enabled = self._resolve_global_fallback()

        # Step 1 & 2: Local search (page_hint and page_hint +/- 1)
        local_candidates: list[MatchCandidate] = []
        if page_hint is not None and self.index.get_page(page_hint):
            local_pages = [page_hint]
            if page_hint - 1 >= 1 and self.index.get_page(page_hint - 1):
                local_pages.append(page_hint - 1)
            if page_hint + 1 <= len(self.index.pages) and self.index.get_page(page_hint + 1):
                local_pages.append(page_hint + 1)

            local_candidates = self._execute_recovery_passes(
                target_pages=local_pages,
                value=value,
                query_str=query_str,
                is_num=is_num,
                is_bool=is_bool,
                has_exact_match=has_exact_match,
                start_time=start_time,
                timeout_sec=timeout_sec,
            )

        # Safeguard 6: Early termination if local search found candidates
        if local_candidates:
            return self._deduplicate_and_rank(
                candidates=local_candidates,
                page_hint=page_hint,
                is_global_fallback=False,
            )

        # Step 3: Global fallback (if local search produced no candidates or page_hint was None)
        allow_global = global_enabled if page_hint is not None else True
        if not allow_global and len(self.index.pages) > 10:
            return []

        if global_enabled:
            # Safeguard 1: Inverted-index pre-check
            candidate_pages = self._find_candidate_pages(
                value=value,
                evidence_text=evidence_text,
                field_name=field_name,
            )
            if not candidate_pages:
                return []
            if page_hint is not None:
                local_set = {p for p in (page_hint, page_hint - 1, page_hint + 1) if p >= 1}
                global_pages = sorted(candidate_pages - local_set)
            else:
                global_pages = sorted(candidate_pages)
            if not global_pages:
                return []
            target_pages = global_pages
        else:
            # Legacy fallback for documents with <= 10 pages when global fallback is disabled
            target_pages = [p.page_number for p in self.index.pages]

        global_candidates = self._execute_recovery_passes(
            target_pages=target_pages,
            value=value,
            query_str=query_str,
            is_num=is_num,
            is_bool=is_bool,
            has_exact_match=has_exact_match,
            start_time=start_time,
            timeout_sec=timeout_sec,
        )

        eff_hint = original_page_hint if original_page_hint is not None else page_hint
        is_global = is_global_fallback or (eff_hint is not None)
        return self._deduplicate_and_rank(
            candidates=global_candidates,
            page_hint=eff_hint,
            is_global_fallback=is_global,
        )

    def _recover_spaced_numeric(
        self,
        line: VisualLine,
        target_num: float,
        orig_val: Any,
        page_num: int,
    ) -> list[MatchCandidate]:
        """Mechanism 1: Recover numbers with wide character tracking or spaced tokens.

        Example: ['3', '3', '.', '3', '3', '3', '3', '%'] -> 33.3333%
        """
        tokens = line.tokens
        n = len(tokens)
        if n < 2 or not any(ch.isdigit() for ch in line.text):
            return []

        candidates: list[MatchCandidate] = []
        # Check window sizes 2 to 16
        for w in range(2, min(n + 1, 17)):
            for i in range(n - w + 1):
                sub = tokens[i : i + w]
                # Check horizontal spacing: spaced tokens shouldn't have giant gaps
                max_gap = 0.040
                gap_ok = True
                for k in range(len(sub) - 1):
                    gap = sub[k + 1].bbox.x - (sub[k].bbox.x + sub[k].bbox.width)
                    if gap > max_gap:
                        gap_ok = False
                        break
                if not gap_ok:
                    continue

                sub_text = "".join(t.text for t in sub)
                # Parse numeric value
                parsed = parse_numeric_value(sub_text)
                if parsed is not None and self._is_num_close(parsed, target_num):
                    # Check if next token is % and can be included for higher bounding box fidelity
                    matched_tokens = list(sub)
                    if i + w < n and tokens[i + w].text.strip() == "%":
                        gap_pct = tokens[i + w].bbox.x - (sub[-1].bbox.x + sub[-1].bbox.width)
                        if gap_pct <= 0.035:
                            matched_tokens.append(tokens[i + w])
                            sub_text = "".join(t.text for t in matched_tokens)

                    ub = union_bbox_list([t.bbox for t in matched_tokens])
                    if ub:
                        cand = MatchCandidate(
                            page=page_num,
                            bbox=ub,
                            tokens=tuple(matched_tokens),
                            matched_text=sub_text,
                            match_type="recovered_spaced_numeric",
                            raw_similarity=0.98,
                            line_index=line.line_index,
                        )
                        candidates.append(cand)
                        self._record_diag(
                            orig_val=orig_val,
                            raw_toks=[t.text for t in matched_tokens],
                            recon=sub_text,
                            cand=cand,
                            norm_path="whitespace_elision_parse_numeric",
                            why="Document rendered characters as separated tokens; failed single-token numeric parsing",
                            mech="spaced_numeric",
                        )
        return candidates

    def _recover_fragmented_numeric(
        self,
        line: VisualLine,
        target_num: float,
        orig_val: Any,
        page_num: int,
    ) -> list[MatchCandidate]:
        """Mechanism 2: Reconstruct numeric values split across multiple OCR tokens.

        Example: ['-3', '186'] -> -3186 or ['21,', '693.'] -> 21693
        """
        tokens = line.tokens
        n = len(tokens)
        if n < 2 or not any(ch.isdigit() for ch in line.text):
            return []

        candidates: list[MatchCandidate] = []
        for w in range(2, min(n + 1, 8)):
            for i in range(n - w + 1):
                sub = tokens[i : i + w]
                # Horizontal adjacency check
                is_adj = True
                for k in range(len(sub) - 1):
                    gap = sub[k + 1].bbox.x - (sub[k].bbox.x + sub[k].bbox.width)
                    if gap > 0.035:
                        is_adj = False
                        break
                if not is_adj:
                    continue

                # Try direct concatenation and spaced concatenation
                text_concat = "".join(t.text for t in sub)
                parsed = parse_numeric_value(text_concat)
                if parsed is None:
                    parsed = parse_numeric_value(" ".join(t.text for t in sub))

                if parsed is not None and self._is_num_close(parsed, target_num):
                    matched_sub = list(sub)
                    if i + w < n and tokens[i + w].text.strip() == "%":
                        gap_pct = tokens[i + w].bbox.x - (sub[-1].bbox.x + sub[-1].bbox.width)
                        if gap_pct <= 0.035:
                            matched_sub.append(tokens[i + w])
                            text_concat = "".join(t.text for t in matched_sub)

                    ub = union_bbox_list([t.bbox for t in matched_sub])
                    if ub:
                        cand = MatchCandidate(
                            page=page_num,
                            bbox=ub,
                            tokens=tuple(matched_sub),
                            matched_text=text_concat,
                            match_type="recovered_fragmented_numeric",
                            raw_similarity=0.99,
                            line_index=line.line_index,
                        )
                        candidates.append(cand)
                        self._record_diag(
                            orig_val=orig_val,
                            raw_toks=[t.text for t in matched_sub],
                            recon=text_concat,
                            cand=cand,
                            norm_path="adjacent_token_concat_parse_numeric",
                            why="OCR fragmented numeric value across tokens at comma/decimal/minus",
                            mech="ocr_fragmented_numeric",
                        )
        return candidates

    def _recover_split_symbol(
        self,
        line: VisualLine,
        target_num: float,
        orig_val: Any,
        page_num: int,
    ) -> list[MatchCandidate]:
        """Mechanism 3: Recover split percentage and currency tokens.

        Example: ['33.3333333', '%'] or ['$', '1,200.00']
        """
        tokens = line.tokens
        n = len(tokens)
        candidates: list[MatchCandidate] = []

        for i, t in enumerate(tokens):
            parsed = parse_numeric_value(t.text)
            if parsed is None or not self._is_num_close(parsed, target_num):
                continue

            # Check if adjacent token is % or currency
            to_union = [t]
            # Trailing %
            if i + 1 < n and tokens[i + 1].text.strip() == "%":
                gap = tokens[i + 1].bbox.x - (t.bbox.x + t.bbox.width)
                if gap <= 0.030:
                    to_union.append(tokens[i + 1])
            # Leading currency
            if i > 0 and tokens[i - 1].text.strip() in ("$", "€", "£", "¥", "USD"):
                gap = t.bbox.x - (tokens[i - 1].bbox.x + tokens[i - 1].bbox.width)
                if gap <= 0.030:
                    to_union.insert(0, tokens[i - 1])

            if len(to_union) > 1:
                ub = union_bbox_list([tok.bbox for tok in to_union])
                if ub:
                    matched_str = " ".join(tok.text for tok in to_union)
                    cand = MatchCandidate(
                        page=page_num,
                        bbox=ub,
                        tokens=tuple(to_union),
                        matched_text=matched_str,
                        match_type="recovered_split_symbol",
                        raw_similarity=0.99,
                        line_index=line.line_index,
                    )
                    candidates.append(cand)
                    self._record_diag(
                        orig_val=orig_val,
                        raw_toks=[tok.text for tok in to_union],
                        recon=matched_str,
                        cand=cand,
                        norm_path="symbol_token_pairing",
                        why="Symbol separated from numeric token, leading to partial bbox coverage",
                        mech="split_percentage_currency",
                    )
        return candidates

    def _recover_interleaved_tokens(
        self,
        line: VisualLine,
        q_words: list[str],
        clean_q: str,
        orig_val: Any,
        page_num: int,
    ) -> list[MatchCandidate]:
        """Mechanism 4: Recover multi-token phrases when lines contain interleaved sub-rows.

        Example: 'TELCO 38 PARK EXPERTS AVENUE LLC' -> 'TELCO EXPERTS LLC'
        """
        tokens = line.tokens
        if len(tokens) < len(q_words):
            return []

        # Sub-cluster tokens in visual line by vertical coordinate (y-sub-band)
        # Using tight tolerance = 0.007 of page height
        sub_bands: list[list[DocumentToken]] = []
        for t in tokens:
            placed = False
            for band in sub_bands:
                band_y = sum(x.bbox.y for x in band) / len(band)
                if abs(t.bbox.y - band_y) <= 0.007:
                    band.append(t)
                    placed = True
                    break
            if not placed:
                sub_bands.append([t])

        candidates: list[MatchCandidate] = []
        norm_q_words = [w.lower().strip(" ,.;:-_()[]{}'\"") for w in q_words]

        for band in sub_bands:
            if len(band) < len(q_words):
                continue
            band.sort(key=lambda x: x.bbox.x)
            band_words = [t.text.lower().strip(" ,.;:-_()[]{}'\"") for t in band]

            # Match subsequence in band
            w_idx = 0
            matched: list[DocumentToken] = []
            for t, bw in zip(band, band_words):
                if w_idx < len(norm_q_words):
                    target_w = norm_q_words[w_idx]
                    if target_w == bw or target_w in bw or (len(bw) >= 3 and bw in target_w):
                        matched.append(t)
                        w_idx += 1

            if w_idx == len(norm_q_words) and len(matched) == len(norm_q_words):
                ub = union_bbox_list([t.bbox for t in matched])
                if ub:
                    matched_str = " ".join(t.text for t in matched)
                    cand = MatchCandidate(
                        page=page_num,
                        bbox=ub,
                        tokens=tuple(matched),
                        matched_text=matched_str,
                        match_type="recovered_interleaved",
                        raw_similarity=0.97,
                        line_index=line.line_index,
                    )
                    candidates.append(cand)
                    self._record_diag(
                        orig_val=orig_val,
                        raw_toks=[t.text for t in matched],
                        recon=matched_str,
                        cand=cand,
                        norm_path="y_subband_interleaved_subsequence",
                        why="Tokens were non-contiguous in visual line due to multi-column/sub-row interleaving",
                        mech="y_band_interleaved",
                    )
        return candidates

    def _recover_bounded_spans(
        self,
        line: VisualLine,
        q_words: list[str],
        clean_q: str,
        orig_val: Any,
        page_num: int,
    ) -> list[MatchCandidate]:
        """Mechanism 5: Recover candidate spans when 1 expected token was dropped or corrupted by OCR."""
        tokens = line.tokens
        n = len(tokens)
        k = len(q_words)
        if n < k - 1:
            return []

        norm_q_words = [w.lower().strip(" ,.;:-_()[]{}'\"") for w in q_words]
        candidates: list[MatchCandidate] = []

        # Window sizes from k - 1 to k + 1
        for w in range(max(2, k - 1), min(n + 1, k + 3)):
            for i in range(n - w + 1):
                sub = tokens[i : i + w]
                sub_words = [t.text.lower().strip(" ,.;:-_()[]{}'\"") for t in sub]

                # Count matched query words in sub and track indices
                matched_indices: list[int] = []
                q_ptr = 0
                for idx, sw in enumerate(sub_words):
                    if q_ptr < len(norm_q_words):
                        qw = norm_q_words[q_ptr]
                        if qw == sw or qw in sw or (len(sw) >= 3 and sw in qw):
                            matched_indices.append(idx)
                            q_ptr += 1

                # If at least 80% of query words matched (or k-1 out of k)
                matches = len(matched_indices)
                if matches >= max(2, k - 1) and matches / k >= 0.75:
                    # Strictly trim leading and trailing unmatched tokens
                    trimmed_sub = sub[matched_indices[0] : matched_indices[-1] + 1]
                    ub = union_bbox_list([t.bbox for t in trimmed_sub])
                    if ub:
                        matched_str = " ".join(t.text for t in trimmed_sub)
                        cand = MatchCandidate(
                            page=page_num,
                            bbox=ub,
                            tokens=tuple(trimmed_sub),
                            matched_text=matched_str,
                            match_type="recovered_bounded_span",
                            raw_similarity=0.90 + 0.08 * (matches / k),
                            line_index=line.line_index,
                        )
                        candidates.append(cand)
                        self._record_diag(
                            orig_val=orig_val,
                            raw_toks=[t.text for t in trimmed_sub],
                            recon=matched_str,
                            cand=cand,
                            norm_path="bounded_span_anchor_fallback",
                            why=f"One or more anchor tokens missing or corrupted in OCR ({matches}/{k} words matched)",
                            mech="bounded_span",
                        )
        return candidates

    def _is_num_close(self, a: float, b: float) -> bool:
        """Check numeric closeness within tight floating point / percentage tolerance."""
        if abs(a - b) <= max(1e-6, abs(b) * 1e-5):
            return True
        # Handle decimal vs percentage representation (e.g. 0.333333 vs 33.3333)
        if abs(a * 100.0 - b) <= 1e-3 or abs(a - b * 100.0) <= 1e-3:
            return True
        return False

    def _record_diag(
        self,
        orig_val: Any,
        raw_toks: list[str],
        recon: str,
        cand: MatchCandidate,
        norm_path: str,
        why: str,
        mech: str,
    ) -> None:
        self.diagnostics.append(
            RecoveredDiagnostic(
                original_value=orig_val,
                raw_document_tokens=tuple(raw_toks),
                reconstructed_candidate=recon,
                candidate_bbox=cand.bbox.to_coco(),
                normalization_path=norm_path,
                why_previously_missed=why,
                page=cand.page,
                mechanism=mech,
            )
        )
