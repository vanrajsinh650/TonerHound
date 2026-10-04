"""Contextual evidence resolution engine for TonerHound.

Integrates multi-tier matching with spatial disambiguation,
multi-line region recovery, ambiguity gating, and provenance assignment.
"""

from __future__ import annotations

import math
from typing import Any

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.matching.candidate_recovery import CandidateRecoveryEngine
from tonerhound.matching.matcher import EvidenceMatcher, MatchCandidate
from tonerhound.models.types import (
    ExtractionInput,
    ProvenanceStatus,
    ResolutionResult,
)
from tonerhound.normalization.normalizers import (
    normalize_unicode_and_case,
    parse_numeric_value,
)
from tonerhound.resolution.reranker import StructuralReranker
from tonerhound.resolution.verifier import CandidateVerifier


# Feature flag for EXP-028F Phase 1.2: Token-gated global fallback for long documents
ENABLE_GLOBAL_FALLBACK: bool = True

# Feature flags for EXP-033: Candidate Generation Reconciliation & Productionization
ENABLE_EXP033_CANDIDATE_EXPANSION: bool = False
ENABLE_TOKEN_STRIP_RECOVERY: bool = False
ENABLE_MULTI_LINE_RECOVERY: bool = False
ENABLE_GLOBAL_SEARCH_RELAXATION: bool = False
ENABLE_BOOLEAN_EXPANSION: bool = False


class EvidenceResolver:
    """Independent evidence resolution engine."""

    def __init__(
        self,
        index: DocumentIndex,
        enable_verification: bool = True,
        score_margin_threshold: float = 0.05,
        reranker: StructuralReranker | None = None,
        enable_candidate_recovery: bool = True,
        enable_global_fallback: bool | None = None,
        enable_exp033_candidate_expansion: bool | None = None,
        enable_token_strip_recovery: bool | None = None,
        enable_multi_line_recovery: bool | None = None,
        enable_global_search_relaxation: bool | None = None,
        enable_boolean_expansion: bool | None = None,
    ) -> None:
        self.index = index
        self.matcher = EvidenceMatcher(index)
        self.enable_verification = enable_verification
        self.verifier = (
            CandidateVerifier(score_margin_threshold=score_margin_threshold)
            if enable_verification
            else None
        )
        self.reranker = (
            reranker
            if reranker is not None
            else StructuralReranker(
                enabled=True,
                w_column=8.0,
                w_row=10.0,
                w_sibling=8.0,
                w_sequence=5.0,
                w_page=15.0,
            )
        )
        self.enable_candidate_recovery = enable_candidate_recovery
        self.enable_global_fallback = enable_global_fallback
        self.enable_exp033_candidate_expansion = enable_exp033_candidate_expansion
        self.enable_token_strip_recovery = enable_token_strip_recovery
        self.enable_multi_line_recovery = enable_multi_line_recovery
        self.enable_global_search_relaxation = enable_global_search_relaxation
        self.enable_boolean_expansion = enable_boolean_expansion
        self.recovery_engine = (
            CandidateRecoveryEngine(
                index,
                enable_global_fallback=enable_global_fallback,
            )
            if enable_candidate_recovery
            else None
        )
        self.last_field_candidates: dict[str, list[MatchCandidate]] = {}

    def _resolve_global_fallback(self) -> bool:
        """Resolve effective global fallback setting, respecting instance/module overrides."""
        if self.enable_global_fallback is not None:
            return self.enable_global_fallback
        return ENABLE_GLOBAL_FALLBACK

    def _is_flag_enabled(self, flag_attr: bool | None, module_flag: bool) -> bool:
        """Resolve effective candidate generation flag."""
        if flag_attr is not None:
            return flag_attr
        if self.enable_exp033_candidate_expansion is True or ENABLE_EXP033_CANDIDATE_EXPANSION:
            return True
        return module_flag

    def collect_candidates(self, extraction: ExtractionInput) -> list[MatchCandidate]:
        """Collect all matching candidates across standard tiers and recovery engine."""
        field = extraction.field
        value = extraction.value
        evidence_text = extraction.evidence_text
        context = extraction.field_context or field
        page_hint = extraction.page_hint

        if value is None or (isinstance(value, str) and not value.strip()):
            return []

        candidates: list[MatchCandidate] = []

        # Tier 1a: Exact match on evidence_text (if provided)
        if evidence_text:
            candidates.extend(self.matcher.find_exact_candidates(evidence_text, page_hint=page_hint))

        # Tier 1b: If value is numeric, prioritize normalized numeric matching over raw float str()
        is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
        enable_bool = self._is_flag_enabled(self.enable_boolean_expansion, ENABLE_BOOLEAN_EXPANSION)
        if enable_bool:
            is_bool = isinstance(value, bool) or (
                isinstance(value, str)
                and value.strip().lower() in ("true", "false", "yes", "no")
            )
        else:
            is_bool = isinstance(value, bool) or (
                isinstance(value, str)
                and value.strip().lower() in ("true", "false", "yes", "no")
                and any(k in field.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no", "final", "amended", "general", "domestic", "contributed"))
            )

        # Tier 1-bool: Checkbox candidate generation
        if not candidates and is_bool:
            candidates.extend(self.matcher.find_boolean_candidates(value, field_name=field, page_hint=page_hint, field_context=context))

        if not candidates and is_num:
            candidates.extend(self.matcher.find_normalized_numeric_candidates(value, page_hint=page_hint))

        # Tier 1c: Exact match on stringified value (excluding pure booleans)
        if not candidates and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            candidates.extend(self.matcher.find_exact_candidates(str(value), page_hint=page_hint))

        # Tier 1d: Normalized date match
        if not candidates and isinstance(value, str) and not is_bool:
            candidates.extend(self.matcher.find_normalized_date_candidates(value, page_hint=page_hint))

        # Tier 2: Normalized numeric match for strings that might be formatted numbers
        if not candidates and isinstance(value, str) and not is_bool:
            candidates.extend(self.matcher.find_normalized_numeric_candidates(value, page_hint=page_hint))

        # Tier 3: Fuzzy sequence alignment fallback (only for text strings)
        if not candidates and isinstance(value, str) and not is_num and not is_bool:
            query = evidence_text if evidence_text else str(value)
            candidates.extend(self.matcher.find_fuzzy_candidates(query, threshold=0.82, page_hint=page_hint))

        # If still no candidates found on page_hint, relax page_hint to search all pages
        if not candidates and page_hint is not None:
            if evidence_text:
                candidates.extend(self.matcher.find_exact_candidates(evidence_text, page_hint=None))
            if not candidates and is_bool:
                candidates.extend(self.matcher.find_boolean_candidates(value, field_name=field, page_hint=None, field_context=context))
            if not candidates and is_num:
                candidates.extend(self.matcher.find_normalized_numeric_candidates(value, page_hint=None))
            if not candidates and isinstance(value, (str, int, float)) and not isinstance(value, bool):
                candidates.extend(self.matcher.find_exact_candidates(str(value), page_hint=None))
            if not candidates and isinstance(value, str) and not is_bool:
                candidates.extend(self.matcher.find_normalized_date_candidates(value, page_hint=None))
            if not candidates and isinstance(value, str) and not is_bool:
                candidates.extend(self.matcher.find_normalized_numeric_candidates(value, page_hint=None))

        # EXP-033 Fix 2: Punctuation and token strip fallback
        enable_token_strip = self._is_flag_enabled(self.enable_token_strip_recovery, ENABLE_TOKEN_STRIP_RECOVERY)
        if not candidates and enable_token_strip and isinstance(value, str) and not is_bool:
            stripped = value.strip(" -.,;:_()[]{}/'\"")
            if stripped and stripped != value:
                candidates.extend(self.matcher.find_exact_candidates(stripped, page_hint=page_hint))
                if not candidates:
                    candidates.extend(self.matcher.find_normalized_numeric_candidates(stripped, page_hint=page_hint))
                if not candidates and page_hint is not None:
                    candidates.extend(self.matcher.find_exact_candidates(stripped, page_hint=None))
                    if not candidates:
                        candidates.extend(self.matcher.find_normalized_numeric_candidates(stripped, page_hint=None))

        # EXP-033 Fix 4: Multi-line and newline span recovery
        enable_multiline = self._is_flag_enabled(self.enable_multi_line_recovery, ENABLE_MULTI_LINE_RECOVERY)
        if not candidates and enable_multiline and isinstance(value, str) and "\n" in value:
            clean_ml = " ".join(value.split())
            if clean_ml and clean_ml != value:
                candidates.extend(self.matcher.find_exact_candidates(clean_ml, page_hint=page_hint))
                if not candidates and page_hint is not None:
                    candidates.extend(self.matcher.find_exact_candidates(clean_ml, page_hint=None))

        # EXP-013: Modular Candidate Recovery Engine
        if self.recovery_engine is not None:
            has_exact = any(c.match_type == "exact" for c in candidates)
            rec_cands = self.recovery_engine.recover(
                value=value,
                page_hint=page_hint,
                field_name=field,
                field_context=context,
                evidence_text=evidence_text,
                has_exact_match=has_exact,
            )
            if not candidates:
                candidates.extend(rec_cands)
            elif rec_cands:
                for rc in rec_cands:
                    if not any(rc.page == c.page and rc.bbox.iou(c.bbox) >= 0.70 for c in candidates):
                        candidates.append(rc)


        # Enforce page priority ranking (page_hint > page_hint +/- 1 > global) and candidate cap
        if candidates and page_hint is not None and len(candidates) > 1:
            def _candidate_priority(c: MatchCandidate) -> tuple[int, float]:
                if c.page == page_hint:
                    prio = 0
                elif abs(c.page - page_hint) == 1:
                    prio = 1
                else:
                    prio = 2
                return (prio, -c.raw_similarity)

            candidates.sort(key=_candidate_priority)

        return candidates[:200]

    def resolve(self, extraction: ExtractionInput) -> ResolutionResult:
        """Resolve physical evidence for an extracted field."""
        field = extraction.field
        value = extraction.value
        evidence_text = extraction.evidence_text
        context = extraction.field_context or field
        page_hint = extraction.page_hint

        # Handle null / empty extraction
        if value is None or (isinstance(value, str) and not value.strip()):
            return ResolutionResult(
                field=field,
                value=value,
                status=ProvenanceStatus.EXACT,
                page=None,
                bbox=None,
                confidence=1.0,
                explanation="Null / empty field",
            )

        # Step 1: Candidate Generation across tiers + recovery
        candidates = self.collect_candidates(extraction)
        self.last_field_candidates[field] = list(candidates)

        # If zero candidates found: Check for derived vs not_found
        if not candidates:
            is_derived = self._is_likely_derived(field, value)
            return ResolutionResult(
                field=field,
                value=value,
                status=ProvenanceStatus.DERIVED if is_derived else ProvenanceStatus.NOT_FOUND,
                page=None,
                bbox=None,
                confidence=0.0,
                explanation=f"Evidence could not be physically sighted ({'derived calculation' if is_derived else 'not found'})",
            )

        # Step 2: Candidate Ranking
        if len(candidates) == 1:
            if extraction.y_hint is not None:
                cand_cy = candidates[0].bbox.y + candidates[0].bbox.height / 2.0
                if abs(cand_cy - extraction.y_hint) > 0.080:
                    return ResolutionResult(
                        field=field,
                        value=value,
                        status=ProvenanceStatus.NOT_FOUND,
                        page=None,
                        bbox=None,
                        confidence=0.0,
                        explanation="Single candidate outside row vertical tolerance",
                    )
            scored_candidates = [(candidates[0], 5.0)]
        else:
            scored_candidates = self._score_candidates_with_context(
                candidates, context, y_hint=extraction.y_hint
            )
            scored_candidates.sort(key=lambda item: item[1], reverse=True)

            # EXP-011: Structural Evidence Reranking
            if self.reranker is not None and self.reranker.enabled:
                base_cands = [c for c, _ in scored_candidates]
                base_scores = [s for _, s in scored_candidates]
                is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
                target_p = extraction.target_page or (
                    extraction.page_hint
                    if (extraction.page_confidence and extraction.page_confidence > 0)
                    else None
                )
                p_conf = (
                    extraction.page_confidence
                    if extraction.page_confidence > 0
                    else (1.0 if target_p is not None else 0.0)
                )
                reranked = self.reranker.rank_candidates(
                    candidates=base_cands,
                    base_scores=base_scores,
                    column_corridor=extraction.column_corridor,
                    column_peers=extraction.column_peers,
                    target_y=extraction.expected_row_y,
                    row_corridor=extraction.row_corridor,
                    sibling_boxes=extraction.sibling_boxes,
                    expected_row_y=extraction.expected_row_y,
                    prev_row_y=extraction.prev_row_y,
                    next_row_y=extraction.next_row_y,
                    target_page=target_p,
                    page_confidence=p_conf,
                    is_numeric=is_num,
                    is_header_field=extraction.is_header,
                    total_pages=len(self.index.pages),
                )
                scored_candidates = [(r.candidate, r.total_score) for r in reranked]

        top_cand, top_score = scored_candidates[0]
        if extraction.y_hint is not None:
            top_cy = top_cand.bbox.y + top_cand.bbox.height / 2.0
            if abs(top_cy - extraction.y_hint) > 0.040:
                within_tol = [
                    (c, s) for c, s in scored_candidates
                    if abs((c.bbox.y + c.bbox.height / 2.0) - extraction.y_hint) <= 0.040
                ]
                if within_tol:
                    top_cand, top_score = within_tol[0]
                elif abs(top_cy - extraction.y_hint) > 0.080:
                    return ResolutionResult(
                        field=field,
                        value=value,
                        status=ProvenanceStatus.NOT_FOUND,
                        page=None,
                        bbox=None,
                        confidence=0.0,
                        explanation="Top candidate outside row vertical tolerance",
                    )

        # Step 3: Candidate Verification (Strict Verification Stage)
        if self.verifier is not None:
            decision = self.verifier.verify(
                field=field,
                value=value,
                top_candidate=top_cand,
                scored_candidates=scored_candidates,
                field_context=context,
                page_hint=page_hint,
            )
            if not decision.is_accepted:
                return ResolutionResult(
                    field=field,
                    value=value,
                    status=decision.status,
                    page=decision.page,
                    bbox=decision.bbox,
                    confidence=decision.confidence,
                    matched_text=decision.matched_text,
                    explanation=decision.reason,
                )
            return self._build_result(
                field,
                value,
                top_cand,
                confidence=decision.confidence,
                explanation=decision.reason,
            )

        # Baseline path (when verification is disabled)
        if len(candidates) == 1:
            return self._build_result(field, value, candidates[0], confidence=0.95, explanation="Single unambiguous match")

        second_cand, second_score = scored_candidates[1]
        score_diff = top_score - second_score
        if score_diff < 0.05 and abs(top_score - second_score) < 1e-4:
            if top_cand.page == second_cand.page and top_cand.bbox.iou(second_cand.bbox) >= 0.8:
                return self._build_result(field, value, top_cand, confidence=0.90, explanation="Overlapping duplicate candidates")
            return ResolutionResult(
                field=field,
                value=value,
                status=ProvenanceStatus.AMBIGUOUS,
                page=None,
                bbox=None,
                confidence=0.5,
                explanation=f"Multiple indistinguishable candidates found ({len(candidates)} occurrences)",
            )

        confidence = min(0.99, max(0.60, 0.70 + (score_diff * 0.25)))
        return self._build_result(field, value, top_cand, confidence=confidence, explanation="Contextually disambiguated")

    def _score_candidates_with_context(
        self,
        candidates: list[MatchCandidate],
        context: str,
        y_hint: float | None = None,
    ) -> list[tuple[MatchCandidate, float]]:
        """Score each candidate based on spatial proximity to context label and row y_hint."""
        norm_context = normalize_unicode_and_case(context).text.strip()
        cand_texts = {normalize_unicode_and_case(c.matched_text).text.strip() for c in candidates}
        context_words = [
            w for w in norm_context.split()
            if len(w) > 1 and w not in cand_texts and not any(w == ct for ct in cand_texts)
        ]
        if not context_words:
            context_words = [w for w in norm_context.split() if len(w) > 1]

        # Precompute context label boxes with match weights per page once
        # (lbox, match_count)
        labels_by_page: dict[int, list[tuple[BBox, int]]] = {}
        target_pages = {cand.page for cand in candidates}
        for p_num in target_pages:
            p = self.index.get_page(p_num)
            boxes: list[tuple[BBox, int]] = []
            if p and context_words:
                for line in p.lines:
                    line_norm = line.norm_text
                    m_count = sum(1 for w in context_words if w in line_norm)
                    if m_count > 0:
                        boxes.append((line.bbox, m_count))
            labels_by_page[p_num] = boxes

        scored: list[tuple[MatchCandidate, float]] = []

        for cand in candidates:
            score = 1.0
            page = self.index.get_page(cand.page)
            if not page:
                scored.append((cand, score))
                continue

            # Row y_hint guidance: heavily reward candidates on the specified row line
            if y_hint is not None:
                cand_cy = cand.bbox.y + cand.bbox.height / 2.0
                dy = abs(cand_cy - y_hint)
                if dy <= 0.012:
                    score += 12.0 * (1.0 - dy / 0.012)
                elif dy <= 0.025:
                    score += 4.0 * (1.0 - (dy - 0.012) / 0.013)
                else:
                    score -= min(25.0, dy * 60.0)

            context_label_boxes = labels_by_page.get(cand.page, [])

            if context_label_boxes:
                cand_cx = cand.bbox.x + cand.bbox.width / 2.0
                cand_cy = cand.bbox.y + cand.bbox.height / 2.0
                cand_x0 = cand.bbox.x
                cand_x1 = cand.bbox.x + cand.bbox.width
                best_spatial_bonus = 0.0

                for lbox, m_count in context_label_boxes:
                    lbl_cx = lbox.x + lbox.width / 2.0
                    lbl_cy = lbox.y + lbox.height / 2.0
                    lbl_weight = 1.0 + 0.5 * (m_count - 1)

                    # Euclidean distance
                    dist = math.sqrt((cand_cx - lbl_cx) ** 2 + (cand_cy - lbl_cy) ** 2)
                    dist_score = (1.0 / (1.0 + dist * 5.0)) * lbl_weight

                    bonus = 0.0
                    # 1. Horizontal key-value alignment (label to left of candidate on same line)
                    if abs(cand_cy - lbl_cy) < (cand.bbox.height * 0.8):
                        if cand.bbox.x >= lbox.x:
                            bonus = max(bonus, 5.0)
                        else:
                            bonus = max(bonus, 1.5)
                    elif abs(cand_cy - lbl_cy) < (cand.bbox.height * 1.5) and cand.bbox.x >= lbox.x:
                        bonus = max(bonus, 3.5)

                    # 2. Vertical form field alignment (label directly above candidate)
                    v_dist = cand.bbox.y - (lbox.y + lbox.height)
                    h_overlap = (max(cand_x0, lbox.x) <= min(cand_x1, lbox.x + lbox.width) + 0.05)
                    if 0.0 <= v_dist <= (cand.bbox.height * 2.5) and h_overlap:
                        bonus = max(bonus, 3.0)

                    combined = (dist_score * 2.0) + (bonus * lbl_weight)
                    best_spatial_bonus = max(best_spatial_bonus, bonus)
                    score += combined

                # Deprioritize running headers at top of multi-page documents if lacking direct label
                if cand.bbox.y < 0.055 and len(self.index.pages) > 2 and best_spatial_bonus < 2.0:
                    score -= 2.0

            # Bonus for exact match type over normalized or fuzzy
            if cand.match_type == "exact" or cand.match_type.startswith("recovered"):
                score += 0.5
            elif cand.match_type.startswith("normalized"):
                score += 0.3

            scored.append((cand, score))

        return scored

    def _build_result(
        self,
        field: str,
        value: Any,
        cand: MatchCandidate,
        confidence: float,
        explanation: str,
    ) -> ResolutionResult:
        """Construct ResolutionResult with line-aware geometry."""
        # Determine provenance status
        if cand.is_multi_line:
            status = ProvenanceStatus.MULTI_REGION
            # Group tokens by line
            lines_dict: dict[int, list[BBox]] = {}
            for t in cand.tokens:
                lines_dict.setdefault(t.line_index, []).append(t.bbox)
            regions = [union_bbox_list(boxes) for boxes in lines_dict.values()]
            regions = [r for r in regions if r is not None]
        elif cand.match_type == "exact":
            status = ProvenanceStatus.EXACT
            regions = [cand.bbox]
        elif cand.match_type.startswith("normalized"):
            status = ProvenanceStatus.NORMALIZED
            regions = [cand.bbox]
        else:
            status = ProvenanceStatus.FUZZY
            regions = [cand.bbox]

        return ResolutionResult(
            field=field,
            value=value,
            status=status,
            page=cand.page,
            bbox=cand.bbox,
            regions=regions,
            confidence=confidence,
            matched_text=cand.matched_text,
            explanation=explanation,
        )

    def _is_likely_derived(self, field: str, value: Any) -> bool:
        """Heuristic to detect mathematically or semantically derived values."""
        derived_keywords = {"total", "sum", "subtotal", "tax_rate", "average", "ratio", "percent", "count"}
        field_lower = field.lower()
        return any(kw in field_lower for kw in derived_keywords)
