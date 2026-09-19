"""Structural Evidence Reranker for TonerHound (EXP-011).

Disambiguates candidate evidence occurrences using extraction-aware structural signals:
1. Column Alignment (horizontal rail alignment and corridor containment)
2. Same-Row Consistency (vertical row-band confinement and anchor affinity)
3. Sibling Anchor Locking (binding low-entropy scalars to high-entropy record keys)
4. Row-Pitch Continuity (enforcing top-to-bottom monotonic array order)
5. Page & Section Context (routing candidates to verified pages and section boundaries)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate


@dataclass(slots=True)
class RerankedCandidate:
    """A candidate with a decomposed structural score breakdown."""

    candidate: MatchCandidate
    total_score: float
    base_score: float
    column_score: float
    row_score: float
    sibling_score: float
    sequence_score: float
    page_score: float
    is_ambiguous: bool = False
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def page(self) -> int:
        return self.candidate.page

    @property
    def bbox(self) -> BBox:
        return self.candidate.bbox

    @property
    def matched_text(self) -> str:
        return self.candidate.matched_text


@dataclass(slots=True)
class StructuralReranker:
    """Extractor-independent structural evidence reranker."""

    enabled: bool = True
    w_column: float = 7.0
    w_row: float = 8.0
    w_sibling: float = 5.0
    w_sequence: float = 5.0
    w_page: float = 12.0
    ambiguity_margin: float = 0.05

    def column_score(
        self,
        candidate_bbox: BBox,
        column_corridor: tuple[float, float] | None = None,
        column_peers: list[Any] | None = None,
        is_numeric: bool = True,
    ) -> float:
        """Compute horizontal column rail and corridor alignment score in [-1.0, 1.0].

        Returns 0.0 if no column information is available.
        """
        if column_corridor is None and not column_peers:
            return 0.0

        c_x0 = candidate_bbox.x
        c_w = candidate_bbox.width
        c_x1 = c_x0 + c_w
        c_xc = c_x0 + c_w / 2.0

        # Derive corridor bounds from corridor tuple or peer consensus
        if column_corridor is not None:
            col_x0, col_x1 = column_corridor
        else:
            peer_xs = []
            for p in column_peers or []:
                if isinstance(p, dict):
                    px = p.get("cx", p.get("x", 0.0))
                    pw = p.get("w", p.get("width", 0.05))
                    peer_xs.append((px - pw / 2.0, px + pw / 2.0))
                elif isinstance(p, BBox):
                    peer_xs.append((p.x, p.x + p.width))
            if not peer_xs:
                return 0.0
            col_x0 = sorted(p[0] for p in peer_xs)[len(peer_xs) // 2]
            col_x1 = sorted(p[1] for p in peer_xs)[len(peer_xs) // 2]

        col_w = max(0.015, col_x1 - col_x0)
        col_xc = (col_x0 + col_x1) / 2.0

        # 1D Horizontal Overlap / Containment
        overlap = max(0.0, min(c_x1, col_x1) - max(c_x0, col_x0))
        containment = overlap / max(1e-4, c_w)

        # Rail Distance: Right rail for numeric/amounts, Left rail for text, Center for codes
        if is_numeric:
            delta_rail = abs(c_x1 - col_x1)
        else:
            delta_rail = abs(c_x0 - col_x0)

        sigma_x = max(0.008, min(0.030, col_w * 0.35))
        rail_score = math.exp(-(delta_rail**2) / (2.0 * sigma_x**2))

        # Composite score in [0.0, 1.0]
        comp = 0.65 * rail_score + 0.35 * containment

        # Center distance penalty if candidate is completely outside corridor
        dist_to_corridor = max(0.0, col_x0 - c_x1, c_x0 - col_x1)
        if dist_to_corridor > 0.035:
            # Candidate is in an adjacent column rail -> negative score
            return max(-1.0, -1.0 * min(1.0, dist_to_corridor / 0.050))

        # Maps comp in [0.0, 1.0] to [-0.5, 1.0]
        return max(-1.0, min(1.0, 1.5 * comp - 0.5))

    def row_score(
        self,
        candidate_bbox: BBox,
        target_y: float | None = None,
        row_corridor: tuple[float, float] | None = None,
        row_height: float = 0.012,
    ) -> float:
        """Compute vertical row confinement score in [-1.0, 1.0].

        Returns 0.0 if target_y and row_corridor are None.
        """
        if target_y is None and row_corridor is None:
            return 0.0

        c_yc = candidate_bbox.y + candidate_bbox.height / 2.0

        if row_corridor is not None:
            y_min, y_max = row_corridor
            effective_target_y = (y_min + y_max) / 2.0
            effective_h = max(row_height, y_max - y_min)
            if y_min - 0.003 <= c_yc <= y_max + 0.003:
                delta_y = abs(c_yc - effective_target_y)
                return max(0.0, 1.0 - (delta_y / (effective_h / 2.0 + 1e-4)))
        else:
            effective_target_y = target_y
            effective_h = row_height

        delta_y = abs(c_yc - effective_target_y)
        sigma_y = max(0.004, effective_h * 0.60)

        if delta_y <= effective_h * 0.75:
            # Within single row line height
            return math.exp(-(delta_y**2) / (2.0 * sigma_y**2))

        # Outside row corridor -> penalty scaling with row distance
        excess_y = delta_y - (effective_h * 0.75)
        return max(-1.0, -min(1.0, excess_y / (effective_h * 1.5)))

    def sibling_score(
        self,
        candidate_bbox: BBox,
        candidate_page: int,
        sibling_boxes: list[Any] | None = None,
    ) -> float:
        """Compute co-linearity and proximity score to verified record siblings in [0.0, 1.0].

        Returns 0.0 if no sibling boxes are provided on the candidate page.
        """
        if not sibling_boxes:
            return 0.0

        c_xc = candidate_bbox.x + candidate_bbox.width / 2.0
        c_yc = candidate_bbox.y + candidate_bbox.height / 2.0

        same_page_sibs: list[tuple[float, float, float, float]] = []
        for s in sibling_boxes:
            if isinstance(s, dict):
                if s.get("page") == candidate_page:
                    b = s.get("bbox", [0, 0, 0, 0])
                    same_page_sibs.append((b[0], b[1], b[2], b[3]))
            elif isinstance(s, BBox):
                if s.page is None or s.page == candidate_page:
                    same_page_sibs.append((s.x, s.y, s.width, s.height))

        if not same_page_sibs:
            return 0.0

        scores: list[float] = []
        for sx, sy, sw, sh in same_page_sibs:
            s_yc = sy + sh / 2.0
            dy = abs(c_yc - s_yc)

            # Horizontal line co-linearity (same baseline)
            colinear_affinity = math.exp(-(dy**2) / (2.0 * 0.006**2)) if dy < 0.020 else 0.0

            # 2D Euclidean proximity
            dist = math.sqrt((c_xc - (sx + sw / 2.0)) ** 2 + (c_yc - s_yc) ** 2)
            prox_affinity = 1.0 / (1.0 + 6.0 * dist)

            scores.append(0.60 * colinear_affinity + 0.40 * prox_affinity)

        return max(scores) if scores else 0.0

    def sequence_score(
        self,
        candidate_bbox: BBox,
        expected_y: float | None = None,
        prev_row_y: float | None = None,
        next_row_y: float | None = None,
        pitch: float = 0.015,
    ) -> float:
        """Compute array row sequence and monotonicity score in [-1.0, 1.0].

        Strictly penalizes backward row ordering or impossible sequence skips.
        """
        if prev_row_y is None and next_row_y is None and expected_y is None:
            return 0.0

        c_yc = candidate_bbox.y + candidate_bbox.height / 2.0

        # Monotonicity violation: row cannot appear above preceding row in table
        if prev_row_y is not None and c_yc <= prev_row_y + 0.002:
            return -1.0

        # Inversion violation: row cannot appear below succeeding row in table
        if next_row_y is not None and c_yc >= next_row_y - 0.002:
            return -1.0

        if expected_y is not None:
            delta = abs(c_yc - expected_y)
            sigma_p = max(0.005, pitch * 0.35)
            return math.exp(-(delta**2) / (2.0 * sigma_p**2))

        return 0.5  # Consistent with sequence bounds

    def page_context_score(
        self,
        candidate: MatchCandidate | Any,
        target_page: int | None = None,
        page_confidence: float = 1.0,
        is_header_field: bool = False,
        total_pages: int = 1,
    ) -> float:
        """Compute page context score in [-1.0, 1.0].

        Positive bonus on target page; strong barrier penalty on competing pages.
        """
        cand_page = getattr(candidate, "page", None)
        if cand_page is None and isinstance(candidate, dict):
            cand_page = candidate.get("page")

        if target_page is None or cand_page is None or page_confidence <= 0.0:
            return 0.0

        c_bbox = getattr(candidate, "bbox", None)
        if c_bbox is None and isinstance(candidate, dict):
            b = candidate.get("bbox", [0, 0, 0, 0])
            c_bbox = BBox(b[0], b[1], b[2], b[3], page=cand_page)

        c_yc = c_bbox.y + c_bbox.height / 2.0 if c_bbox is not None else 0.5

        if cand_page == target_page:
            score = 1.0 * page_confidence
            # Running header suppression on target page if not a header field
            if not is_header_field and total_pages > 2 and c_yc < 0.080:
                score -= 0.40 * (1.0 - c_yc / 0.080)
            return max(-1.0, min(1.0, score))

        # Different page: strict penalty
        score = -1.0 * page_confidence
        return max(-1.0, min(1.0, score))

    def rank_candidates(
        self,
        candidates: list[MatchCandidate],
        base_scores: list[float] | None = None,
        column_corridor: tuple[float, float] | None = None,
        column_peers: list[Any] | None = None,
        target_y: float | None = None,
        row_corridor: tuple[float, float] | None = None,
        sibling_boxes: list[Any] | None = None,
        expected_row_y: float | None = None,
        prev_row_y: float | None = None,
        next_row_y: float | None = None,
        target_page: int | None = None,
        page_confidence: float = 1.0,
        is_numeric: bool = True,
        is_header_field: bool = False,
        total_pages: int = 1,
    ) -> list[RerankedCandidate]:
        """Rank candidates using combined base scores and normalized structural signals."""
        if not candidates:
            return []

        if base_scores is None or len(base_scores) != len(candidates):
            base_scores = [5.0] * len(candidates)

        if not self.enabled or len(candidates) == 1:
            reranked = []
            for c, bs in zip(candidates, base_scores):
                reranked.append(
                    RerankedCandidate(
                        candidate=c,
                        total_score=bs,
                        base_score=bs,
                        column_score=0.0,
                        row_score=0.0,
                        sibling_score=0.0,
                        sequence_score=0.0,
                        page_score=0.0,
                        breakdown={"base": bs, "total": bs},
                    )
                )
            return reranked

        reranked = []
        for cand, base in zip(candidates, base_scores):
            col_s = self.column_score(cand.bbox, column_corridor, column_peers, is_numeric=is_numeric)
            row_s = self.row_score(cand.bbox, target_y, row_corridor)
            sib_s = self.sibling_score(cand.bbox, cand.page, sibling_boxes)
            seq_s = self.sequence_score(cand.bbox, expected_row_y, prev_row_y, next_row_y)
            page_s = self.page_context_score(
                cand, target_page, page_confidence, is_header_field=is_header_field, total_pages=total_pages
            )

            total = (
                base
                + self.w_column * col_s
                + self.w_row * row_s
                + self.w_sibling * sib_s
                + self.w_sequence * seq_s
                + self.w_page * page_s
            )

            breakdown = {
                "base": round(base, 3),
                "column": round(self.w_column * col_s, 3),
                "row": round(self.w_row * row_s, 3),
                "sibling": round(self.w_sibling * sib_s, 3),
                "sequence": round(self.w_sequence * seq_s, 3),
                "page": round(self.w_page * page_s, 3),
                "total": round(total, 3),
            }

            reranked.append(
                RerankedCandidate(
                    candidate=cand,
                    total_score=total,
                    base_score=base,
                    column_score=col_s,
                    row_score=row_s,
                    sibling_score=sib_s,
                    sequence_score=seq_s,
                    page_score=page_s,
                    breakdown=breakdown,
                )
            )

        # Sort descending by total score
        reranked.sort(key=lambda r: r.total_score, reverse=True)

        # Check for ambiguity between top 2 candidates
        if len(reranked) >= 2:
            top = reranked[0]
            sec = reranked[1]
            diff = top.total_score - sec.total_score
            if diff < self.ambiguity_margin:
                # If they do not physically overlap the same region (IoU < 0.50), mark ambiguous
                if top.page != sec.page or top.bbox.iou(sec.bbox) < 0.50:
                    top.is_ambiguous = True

        return reranked
