"""Flat-Form Label-Grounded Reranker for TonerHound (EXP-012).

Post-selection filter and redirect layer that grounds evidence citations to the
correct column in flat tax/regulatory forms where identical values appear in
multiple columns (e.g. K-1 Beginning vs Ending percentages).

Pipeline:
  Stage 1: Vertical family-gated filter — flags suspicious wrong-occurrences
  Stage 2: Label discovery — finds the PDF label nearest the predicted value
  Stage 3: Label-proximity scoring — scores candidates by distance to label row
  Stage 4: Competing-label suppression — resolves multiple label matches by key similarity
  Stage 5: Passthrough — returns original if no better candidate is found
"""

from __future__ import annotations

import math
import re
from typing import Any

from tonerhound.document.index import DocumentIndex
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import ProvenanceStatus, ResolutionResult, VisualLine
from tonerhound.normalization.normalizers import normalize_unicode_and_case

# ---------------------------------------------------------------------------
# Family detection patterns
# ---------------------------------------------------------------------------
FAMILY_PATTERNS: dict[str, list[str]] = {
    "w2": ["passcoag-", "W2-27-"],
    "1040": ["arif-", "bar-lev-", "cabrera-"],
    "1065_k1": ["07021-", "00581-"],
    "texas_rrc": ["H-12", "H-9", "W-1-", "W14"],
}

# ---------------------------------------------------------------------------
# Family-gated parameters
# From ablation: T values where safety >= 100%
# ---------------------------------------------------------------------------
FAMILY_PARAMS: dict[str, dict[str, Any]] = {
    "w2": {"T": 0.025, "label_dir": "RIGHT", "search_radius": 0.30},
    "1040": {"T": 0.008, "label_dir": "LEFT", "search_radius": 0.25},
    "1065_k1": {"T": 0.010, "label_dir": "LEFT", "search_radius": 0.20},
    "texas_rrc": {"T": 0.008, "label_dir": "LEFT", "search_radius": 0.20},
}

# Row-band half-height for label proximity checks (normalized page units)
_ROW_BAND_HALF: float = 0.012


def _detect_family(doc_id: str) -> str | None:
    """Detect form family from document ID string."""
    for family, patterns in FAMILY_PATTERNS.items():
        for pat in patterns:
            if pat.lower() in doc_id.lower():
                return family
    return None


def _normalize_field_tokens(field: str) -> list[str]:
    """Split a snake_case / camelCase / space-separated field name into lowercase word tokens."""
    # Split on underscores, camelCase boundaries, spaces, and other separators
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", field)
    parts = re.split(r"[_\-\.\[\]\s]+", s)
    return [p.lower() for p in parts if len(p) >= 2]


def _token_overlap_score(a_tokens: list[str], b_text: str) -> float:
    """Fraction of a_tokens that appear in b_text (case-insensitive)."""
    if not a_tokens:
        return 0.0
    b_lower = b_text.lower()
    hits = sum(1 for t in a_tokens if t in b_lower)
    return hits / len(a_tokens)


class FlatFormLabelReranker:
    """EXP-012 post-selection label-grounded reranker for flat-form documents.

    This is a *post-selection* filter: it receives the already-resolved
    ``ResolutionResult`` from ``EvidenceResolver`` plus the full candidate list,
    and may redirect to a spatially better candidate if the current prediction
    looks like a wrong-occurrence (same value in multiple columns).

    Usage::

        reranker = FlatFormLabelReranker(index, doc_id="07021-2016-p0014")
        result = reranker.rerank(result, field, value, candidates, field_context)
    """

    def __init__(
        self,
        index: DocumentIndex,
        doc_id: str,
        enabled: bool = True,
        stages_enabled: frozenset[str] | set[str] = frozenset({"vertical", "direction", "label", "suppress"}),
    ) -> None:
        self.index = index
        self.doc_id = doc_id
        self.enabled = enabled
        self.stages_enabled: frozenset[str] = frozenset(stages_enabled)
        self._family: str | None = _detect_family(doc_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def family(self) -> str | None:
        """Detected form family (w2, 1040, 1065_k1, texas_rrc) or None."""
        return self._family

    def rerank(
        self,
        result: ResolutionResult,
        field: str,
        value: Any,
        candidates: list[MatchCandidate],
        field_context: str | None = None,
    ) -> ResolutionResult:
        """Apply EXP-012 label-grounded filter and re-selection.

        The pipeline stages are gated by ``stages_enabled``:

        - ``'vertical'``:  Stage 1 — vertical-outlier trigger filter.
          If absent, all multi-candidate results pass through to label stages.
        - ``'direction'``: Directional bonus in label-proximity scoring.
          If absent, label scoring is purely distance-based (handled in
          ``_score_candidate_by_label``).
        - ``'label'``:     Stages 2-3 — label discovery + proximity re-selection.
          If absent, the reranker always returns the original result unchanged.
        - ``'suppress'``:  Stage 4 — competing-label suppression margin (≥ 0.05 margin
          required before overriding EXP-011 selection). If absent, any score
          improvement causes a redirect.

        Returns the original result if no better candidate is found.
        """
        # Guard 1: disabled or unknown family
        if not self.enabled or self._family is None:
            return result

        params = FAMILY_PARAMS[self._family]

        # Guard 2: result has no bbox (NOT_FOUND / DERIVED already handled upstream)
        if result.bbox is None or result.page is None:
            return result

        # Guard 3: only 1 candidate — no wrong-occurrence risk
        if len(candidates) <= 1:
            return result

        # Guard 4: boolean fields are handled by specialized boolean matcher
        if isinstance(value, bool):
            return result

        # Guard 5: no stages enabled (Config A baseline)
        if not self.stages_enabled:
            return result

        # Find label lines on the predicted page
        page_num = result.page
        label_lines = self._find_label_lines(field, field_context, page_num)
        if not label_lines:
            return result

        # Stage 1: Vertical filter / trigger check
        if "vertical" in self.stages_enabled:
            if not self._detect_trigger(result, candidates, params):
                return result

        same_page_cands = [c for c in candidates if c.page == page_num]
        if not same_page_cands:
            same_page_cands = candidates

        # Score candidates with active stages
        scored: list[tuple[MatchCandidate, float]] = []
        for cand in same_page_cands:
            s = self._score_candidate_by_label(
                cand, label_lines, params, field=field, field_context=field_context
            )
            scored.append((cand, s))

        if not scored:
            return result

        scored.sort(key=lambda x: x[1], reverse=True)
        best_cand, best_score = scored[0]

        if best_score <= 0.0:
            return result

        current_score = self._score_candidate_by_label(
            _make_pseudo_candidate(result),
            label_lines,
            params,
            field=field,
            field_context=field_context,
        )

        # Stage 4: competing-label suppression — require margin before overriding
        if "suppress" in self.stages_enabled:
            if best_score <= current_score + 0.05:
                return result
        else:
            if best_score <= current_score:
                return result

        # Check the best candidate is meaningfully different from current result
        if (
            best_cand.page == result.page
            and best_cand.bbox.iou(result.bbox) >= 0.50
        ):
            return result

        return self._build_improved_result(field, value, best_cand, result)

    # ------------------------------------------------------------------
    # Stage 1 — Vertical suspicion filter
    # ------------------------------------------------------------------

    def _detect_trigger(
        self,
        result: ResolutionResult,
        candidates: list[MatchCandidate],
        params: dict[str, Any],
    ) -> bool:
        """Return True if the prediction has competing same-row column candidates.

        Column ambiguity occurs when multiple occurrences of the same value
        share the same visual row (|dy| <= 3*T_family) on the predicted page,
        but span different columns (horizontal spread >= 0.06).
        """
        T = params["T"]
        pred_page = result.page
        pred_cy = result.bbox.y + result.bbox.height / 2.0

        same_page_cands = [c for c in candidates if c.page == pred_page]
        if len(same_page_cands) < 2:
            return False

        # Find candidates on the same visual row band
        same_row_cands = [
            c for c in same_page_cands
            if abs((c.bbox.y + c.bbox.height / 2.0) - pred_cy) <= T * 3.0
        ]
        if len(same_row_cands) >= 2:
            x_centers = [c.bbox.x + c.bbox.width / 2.0 for c in same_row_cands]
            x_spread = max(x_centers) - min(x_centers)
            if x_spread >= 0.06:
                return True

        return False

    # ------------------------------------------------------------------
    # Stage 2 — Label line discovery
    # ------------------------------------------------------------------

    def _find_label_lines(
        self,
        field: str,
        field_context: str | None,
        page_num: int,
    ) -> list[VisualLine]:
        """Find PDF lines that contain the field label text."""
        query_tokens = _normalize_field_tokens(field_context or field)
        _STOP = {
            "the", "and", "for", "to", "of", "in", "a", "an", "or", "at",
            "on", "with", "per", "as", "is", "no", "yes",
        }
        meaningful = [t for t in query_tokens if t not in _STOP and len(t) >= 2]
        if not meaningful:
            meaningful = query_tokens

        page = self.index.get_page(page_num)
        if not page:
            return []

        results: list[tuple[VisualLine, float]] = []
        for line in page.lines:
            line_lower = line.norm_text.lower()
            hits = sum(1 for t in meaningful if t in line_lower)
            if hits == 0:
                continue
            score = hits / max(1, len(meaningful))
            results.append((line, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [line for line, _ in results]

    # ------------------------------------------------------------------
    # Stages 3+4 — Scoring decomposed across stages A-F
    # ------------------------------------------------------------------

    def _score_candidate_by_label(
        self,
        cand: MatchCandidate,
        label_lines: list[VisualLine],
        params: dict[str, Any],
        field: str = "",
        field_context: str | None = None,
    ) -> float:
        """Score a candidate with active stages."""
        if not label_lines:
            return 0.0

        T = params["T"]
        label_dir: str = params["label_dir"]
        search_radius: float = params["search_radius"]

        cand_cx = cand.bbox.x + cand.bbox.width / 2.0
        cand_cy = cand.bbox.y + cand.bbox.height / 2.0
        sigma = search_radius / 3.0

        query_tokens = set(_normalize_field_tokens(field_context or field))

        # Known column qualifier pairs (mutually exclusive columns)
        COLUMN_OPPOSITES: dict[str, str] = {
            "beginning": "ending",
            "ending": "beginning",
            "employee": "employer",
            "employer": "employee",
            "federal": "state",
            "state": "federal",
            "current": "prior",
            "prior": "current",
            "gross": "net",
            "net": "gross",
            "yes": "no",
            "no": "yes",
        }
        active_qualifiers = {q: COLUMN_OPPOSITES[q] for q in query_tokens if q in COLUMN_OPPOSITES}

        best: float = 0.0

        for label_line in label_lines:
            lbl_cx = label_line.bbox.x + label_line.bbox.width / 2.0
            lbl_cy = label_line.bbox.y + label_line.bbox.height / 2.0
            line_dist = math.sqrt((cand_cx - lbl_cx) ** 2 + (cand_cy - lbl_cy) ** 2)

            # Base proximity (only in label stage)
            if "label" in self.stages_enabled and line_dist <= search_radius:
                line_prox = math.exp(-(line_dist**2) / (2.0 * sigma**2))
                if line_prox > best:
                    best = line_prox

            # Token-level scoring
            for t in label_line.tokens:
                t_clean = t.clean_text.lower().strip(".,:;()[]")
                if not t_clean or len(t_clean) < 2:
                    continue

                tcx = t.bbox.x + t.bbox.width / 2.0
                tcy = t.bbox.y + t.bbox.height / 2.0
                tdist = math.sqrt((cand_cx - tcx) ** 2 + (cand_cy - tcy) ** 2)

                if tdist > search_radius:
                    continue

                is_query_match = t_clean in query_tokens
                is_competing_qualifier = any(t_clean == opp for opp in active_qualifiers.values())
                is_target_qualifier = t_clean in active_qualifiers

                # 1. Vertical row-band component (Stage B)
                row_score = 0.0
                dy = abs(cand_cy - tcy)
                row_band = T * 2.0
                if dy <= row_band and is_query_match:
                    row_fit = 1.0 - dy / row_band

                    # 2. Direction component (Stage C)
                    tdx = tcx - cand_cx
                    dir_bonus = 0.0
                    if "direction" in self.stages_enabled:
                        if label_dir == "RIGHT":
                            dir_bonus = 1.0 if tdx > 0.01 else -0.5
                        else:
                            dir_bonus = 1.0 if tdx < -0.01 else -0.5

                    row_score = row_fit + dir_bonus

                # 3. Label proximity component (Stage D)
                t_prox = 0.0
                if "label" in self.stages_enabled:
                    t_prox = math.exp(-(tdist**2) / (2.0 * sigma**2))

                # 4. Column qualifier component (Stage F)
                col_score = 0.0
                if (
                    "qualifiers" in self.stages_enabled
                    and tcy <= cand_cy + 0.005
                    and (cand_cy - tcy) <= 0.12
                    and abs(cand_cx - tcx) <= 0.06
                ):
                    h_fit = 1.0 - abs(cand_cx - tcx) / 0.06
                    if is_target_qualifier:
                        col_score = 3.5 * h_fit
                    elif is_competing_qualifier:
                        col_score = -3.0 * h_fit
                    elif is_query_match:
                        col_score = 1.5 * h_fit

                token_total = row_score + t_prox + col_score
                if token_total > best:
                    best = token_total

        return best

    # ------------------------------------------------------------------
    # Stage 5 — Build improved result
    # ------------------------------------------------------------------

    def _build_improved_result(
        self,
        field: str,
        value: Any,
        best_cand: MatchCandidate,
        original: ResolutionResult,
    ) -> ResolutionResult:
        """Construct a new ResolutionResult from the label-selected candidate."""
        # Preserve status semantics from original but update bbox/page
        status = original.status
        if status in (ProvenanceStatus.NOT_FOUND, ProvenanceStatus.DERIVED, ProvenanceStatus.AMBIGUOUS):
            status = ProvenanceStatus.NORMALIZED

        return ResolutionResult(
            field=field,
            value=value,
            status=status,
            page=best_cand.page,
            bbox=best_cand.bbox,
            regions=[best_cand.bbox],
            confidence=min(original.confidence, 0.92),  # slightly conservative
            matched_text=best_cand.matched_text,
            explanation=(
                f"EXP-012: label-grounded rerank from ({original.bbox.x:.4f},{original.bbox.y:.4f}) "
                f"→ ({best_cand.bbox.x:.4f},{best_cand.bbox.y:.4f})"
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pseudo_candidate(result: ResolutionResult) -> MatchCandidate:
    """Create a minimal MatchCandidate from a ResolutionResult for scoring."""
    from tonerhound.matching.matcher import MatchCandidate as MC

    return MC(
        page=result.page or 1,
        bbox=result.bbox,  # type: ignore[arg-type]
        tokens=(),
        matched_text=result.matched_text or "",
        match_type="exact",
        raw_similarity=1.0,
        line_index=0,
    )
