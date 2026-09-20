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

import re
from dataclasses import dataclass, field as dc_field
from typing import Any

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import DocumentToken, VisualLine
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
        max_candidate_per_field: int = 15,
    ) -> None:
        self.index = index
        self.enable_spaced_numeric = enable_spaced_numeric
        self.enable_fragmented_numeric = enable_fragmented_numeric
        self.enable_split_symbol = enable_split_symbol
        self.enable_interleaved_tokens = enable_interleaved_tokens
        self.enable_bounded_spans = enable_bounded_spans
        self.max_candidate_per_field = max_candidate_per_field
        self.diagnostics: list[RecoveredDiagnostic] = []

    def recover(
        self,
        value: Any,
        page_hint: int | None = None,
        field_name: str | None = None,
        field_context: str | None = None,
        evidence_text: str | None = None,
        has_exact_match: bool = False,
    ) -> list[MatchCandidate]:
        """Run modular candidate recovery passes for a given extraction."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return []

        # Target pages: prioritize page_hint, else check all pages if doc is small (<= 10 pages)
        if page_hint is not None and self.index.get_page(page_hint):
            target_pages = [page_hint]
        elif len(self.index.pages) <= 10:
            target_pages = [p.page_number for p in self.index.pages]
        else:
            return []

        recovered: list[MatchCandidate] = []
        is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
        is_bool = isinstance(value, bool) or (
            isinstance(value, str)
            and value.strip().lower() in ("true", "false", "yes", "no")
            and any(k in (field_name or "").lower() for k in ("_box", "checkbox", "is_", "has_", "flag"))
        )

        # 1. Numeric / Percentage / Currency recovery passes
        if is_num or (isinstance(value, str) and not is_bool and parse_numeric_value(value) is not None):
            target_num = parse_numeric_value(value)
            if target_num is not None:
                for p_num in target_pages:
                    page = self.index.get_page(p_num)
                    if not page:
                        continue
                    for line in page.lines:
                        # Mechanism 1: Spaced-token numeric recovery
                        if self.enable_spaced_numeric:
                            recovered.extend(self._recover_spaced_numeric(line, target_num, value, p_num))

                        # Mechanism 2: OCR-fragmented numeric values
                        if self.enable_fragmented_numeric:
                            recovered.extend(self._recover_fragmented_numeric(line, target_num, value, p_num))

                        # Mechanism 3: Split currency / percentage tokens
                        if self.enable_split_symbol:
                            recovered.extend(self._recover_split_symbol(line, target_num, value, p_num))

        # 2. String / multi-token recovery passes
        query_str = evidence_text if evidence_text else (str(value) if isinstance(value, str) else None)
        if query_str and not is_bool:
            clean_q = query_str.strip()
            q_words = [w for w in clean_q.split() if len(w.strip(" ,.;:-_()[]{}'\"")) >= 2]

            for p_num in target_pages:
                page = self.index.get_page(p_num)
                if not page:
                    continue
                for line in page.lines:
                    # Mechanism 4: Non-contiguous / Interleaved line tokens
                    if self.enable_interleaved_tokens and len(q_words) >= 2:
                        recovered.extend(self._recover_interleaved_tokens(line, q_words, clean_q, value, p_num))

                    # Mechanism 5: Bounded span / missing anchor recovery (only when no exact candidate was found)
                    if self.enable_bounded_spans and len(q_words) >= 3 and not has_exact_match:
                        recovered.extend(self._recover_bounded_spans(line, q_words, clean_q, value, p_num))

        # Deduplicate recovered candidates by page and high spatial IoU (>= 0.85)
        unique: list[MatchCandidate] = []
        for c in sorted(
            recovered,
            key=lambda x: (x.raw_similarity, len(x.tokens), len(x.matched_text)),
            reverse=True,
        ):
            if not any(u.page == c.page and u.bbox.iou(c.bbox) >= 0.85 for u in unique):
                unique.append(c)

        return unique[: self.max_candidate_per_field]

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
