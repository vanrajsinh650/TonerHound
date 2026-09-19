"""Comprehensive unit and regression tests for Structural Evidence Reranker (EXP-011).

Tests every individual structural signal:
- Column rail alignment and corridor containment (column_score)
- Vertical row confinement and Gaussian penalty (row_score)
- Record sibling co-occurrence and proximity (sibling_score)
- Table array monotonicity and row-pitch sequence (sequence_score)
- Page context routing and running header suppression (page_context_score)
- Ambiguity gating across distinct regions
- Pass-through identity when disabled
- Integration with EvidenceResolver
"""

from __future__ import annotations

import pytest

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import (
    DocumentPage,
    DocumentToken,
    ExtractionInput,
    ProvenanceStatus,
    VisualLine,
)
from tonerhound.resolution.reranker import RerankedCandidate, StructuralReranker
from tonerhound.resolution.resolver import EvidenceResolver


@pytest.fixture
def reranker() -> StructuralReranker:
    return StructuralReranker(
        enabled=True,
        w_column=8.0,
        w_row=10.0,
        w_sibling=8.0,
        w_sequence=5.0,
        w_page=15.0,
        ambiguity_margin=0.05,
    )


def make_cand(
    page: int,
    x: float,
    y: float,
    w: float = 0.05,
    h: float = 0.012,
    text: str = "val",
    match_type: str = "exact",
) -> MatchCandidate:
    bbox = BBox(x=x, y=y, width=w, height=h, page=page)
    tok = DocumentToken(text=text, bbox=bbox, page=page, char_index_in_page=0)
    return MatchCandidate(
        page=page,
        bbox=bbox,
        tokens=(tok,),
        matched_text=text,
        match_type=match_type,
        raw_similarity=1.0,
    )


# -------------------------------------------------------------------------
# Signal 1: Column Score Tests
# -------------------------------------------------------------------------

def test_column_score_corridor_containment(reranker: StructuralReranker) -> None:
    # Candidate centered inside column corridor [0.40, 0.50]
    box_inside = BBox(x=0.42, y=0.30, width=0.06, height=0.012)
    score_inside = reranker.column_score(box_inside, column_corridor=(0.40, 0.50), is_numeric=True)
    assert score_inside > 0.5

    # Candidate in an adjacent column rail far to the left
    box_outside = BBox(x=0.10, y=0.30, width=0.06, height=0.012)
    score_outside = reranker.column_score(box_outside, column_corridor=(0.40, 0.50), is_numeric=True)
    assert score_outside < 0.0

    # No column corridor provided -> 0.0
    score_none = reranker.column_score(box_inside, column_corridor=None, column_peers=None)
    assert score_none == 0.0


def test_column_score_peers_consensus(reranker: StructuralReranker) -> None:
    # Column peers around x=0.60
    peers = [
        {"cx": 0.60, "w": 0.04},
        {"cx": 0.61, "w": 0.04},
        {"cx": 0.59, "w": 0.04},
    ]
    box_aligned = BBox(x=0.58, y=0.40, width=0.04, height=0.012)
    box_misaligned = BBox(x=0.20, y=0.40, width=0.04, height=0.012)

    score_aligned = reranker.column_score(box_aligned, column_peers=peers, is_numeric=True)
    score_misaligned = reranker.column_score(box_misaligned, column_peers=peers, is_numeric=True)

    assert score_aligned > 0.3
    assert score_misaligned < 0.0


# -------------------------------------------------------------------------
# Signal 2: Row Score Tests
# -------------------------------------------------------------------------

def test_row_score_target_y_gaussian(reranker: StructuralReranker) -> None:
    # Candidate exactly on target_y
    target_y = 0.450
    box_exact = BBox(x=0.20, y=target_y - 0.005, width=0.05, height=0.010)
    score_exact = reranker.row_score(box_exact, target_y=target_y)
    assert score_exact == pytest.approx(1.0, rel=1e-2)

    # Candidate displaced vertically by 3 rows
    box_far = BBox(x=0.20, y=target_y + 0.050, width=0.05, height=0.010)
    score_far = reranker.row_score(box_far, target_y=target_y)
    assert score_far < 0.0

    # No row target provided -> 0.0
    assert reranker.row_score(box_exact, target_y=None, row_corridor=None) == 0.0


def test_row_score_corridor_bounds(reranker: StructuralReranker) -> None:
    row_corridor = (0.200, 0.220)
    box_inside = BBox(x=0.20, y=0.205, width=0.05, height=0.010)
    box_outside = BBox(x=0.20, y=0.350, width=0.05, height=0.010)

    score_in = reranker.row_score(box_inside, row_corridor=row_corridor)
    score_out = reranker.row_score(box_outside, row_corridor=row_corridor)

    assert score_in > 0.5
    assert score_out < 0.0


# -------------------------------------------------------------------------
# Signal 3: Sibling Score Tests
# -------------------------------------------------------------------------

def test_sibling_score_colinearity_and_proximity(reranker: StructuralReranker) -> None:
    # Sibling at x=0.10, y=0.300 on page 2
    siblings = [BBox(x=0.10, y=0.300, width=0.08, height=0.012, page=2)]

    # Candidate on same page, same horizontal baseline (y=0.300), nearby x=0.25
    cand_same_row = BBox(x=0.25, y=0.300, width=0.05, height=0.012, page=2)
    score_same_row = reranker.sibling_score(cand_same_row, candidate_page=2, sibling_boxes=siblings)

    # Candidate on same page but different row (y=0.700)
    cand_diff_row = BBox(x=0.25, y=0.700, width=0.05, height=0.012, page=2)
    score_diff_row = reranker.sibling_score(cand_diff_row, candidate_page=2, sibling_boxes=siblings)

    # Candidate on different page (page 1)
    score_diff_page = reranker.sibling_score(cand_same_row, candidate_page=1, sibling_boxes=siblings)

    assert score_same_row > 0.5
    assert score_diff_row < score_same_row
    assert score_diff_page == 0.0


# -------------------------------------------------------------------------
# Signal 4: Sequence Score Tests
# -------------------------------------------------------------------------

def test_sequence_score_monotonicity(reranker: StructuralReranker) -> None:
    prev_row_y = 0.400
    next_row_y = 0.450

    # Inverted candidate (appears above prev_row_y) -> strict -1.0 penalty
    cand_inverted = BBox(x=0.20, y=0.350, width=0.05, height=0.010)
    score_inv = reranker.sequence_score(cand_inverted, prev_row_y=prev_row_y, next_row_y=next_row_y)
    assert score_inv == -1.0

    # Inversion below next_row_y -> strict -1.0 penalty
    cand_below = BBox(x=0.20, y=0.480, width=0.05, height=0.010)
    score_below = reranker.sequence_score(cand_below, prev_row_y=prev_row_y, next_row_y=next_row_y)
    assert score_below == -1.0

    # Candidate inside sequence window matching expected_y = 0.425
    cand_valid = BBox(x=0.20, y=0.420, width=0.05, height=0.010)
    score_valid = reranker.sequence_score(
        cand_valid, expected_y=0.425, prev_row_y=prev_row_y, next_row_y=next_row_y
    )
    assert score_valid > 0.8


# -------------------------------------------------------------------------
# Signal 5: Page Context Score Tests
# -------------------------------------------------------------------------

def test_page_context_score_bonus_and_penalty(reranker: StructuralReranker) -> None:
    target_page = 5
    cand_target = make_cand(page=5, x=0.20, y=0.50)
    cand_other = make_cand(page=12, x=0.20, y=0.50)

    score_target = reranker.page_context_score(cand_target, target_page=target_page, page_confidence=1.0)
    score_other = reranker.page_context_score(cand_other, target_page=target_page, page_confidence=1.0)

    assert score_target == pytest.approx(1.0)
    assert score_other == pytest.approx(-1.0)

    # Running header suppression at top of multi-page document (y < 0.08)
    cand_header = make_cand(page=5, x=0.20, y=0.02)
    score_header = reranker.page_context_score(
        cand_header, target_page=target_page, page_confidence=1.0, is_header_field=False, total_pages=10
    )
    assert score_header < score_target


# -------------------------------------------------------------------------
# Ranking & Ambiguity Tests
# -------------------------------------------------------------------------

def test_rank_candidates_sorting_and_breakdown(reranker: StructuralReranker) -> None:
    c1 = make_cand(page=1, x=0.20, y=0.40, text="33.33%")  # Target candidate
    c2 = make_cand(page=8, x=0.20, y=0.40, text="33.33%")  # Wrong page candidate

    reranked = reranker.rank_candidates(
        candidates=[c1, c2],
        base_scores=[10.0, 12.0],  # Baseline wrongly favored c2
        target_page=1,
        page_confidence=1.0,
        total_pages=10,
    )

    assert len(reranked) == 2
    # c1 should be promoted to rank 1 due to page context (+15.0 vs -15.0)
    assert reranked[0].candidate == c1
    assert reranked[1].candidate == c2
    assert reranked[0].total_score > reranked[1].total_score
    assert "page" in reranked[0].breakdown


def test_ambiguity_gating_distinct_regions(reranker: StructuralReranker) -> None:
    # Two identical candidates on same page with almost identical positions but non-overlapping
    c1 = make_cand(page=1, x=0.20, y=0.40, w=0.05, h=0.01)
    c2 = make_cand(page=1, x=0.20, y=0.42, w=0.05, h=0.01)

    reranked = reranker.rank_candidates(
        candidates=[c1, c2],
        base_scores=[5.0, 5.0],
    )

    # Identical score margin across separated bboxes -> flagged is_ambiguous
    assert reranked[0].is_ambiguous is True


def test_disabled_reranker_leaves_scores_unchanged() -> None:
    disabled_reranker = StructuralReranker(enabled=False)
    c1 = make_cand(page=1, x=0.20, y=0.40)
    c2 = make_cand(page=2, x=0.20, y=0.40)

    reranked = disabled_reranker.rank_candidates(
        candidates=[c1, c2],
        base_scores=[8.0, 9.0],
        target_page=1,
    )

    # Base scores and relative order remain preserved
    assert [r.total_score for r in reranked] == [8.0, 9.0]
    assert reranked[0].candidate == c1


# -------------------------------------------------------------------------
# EvidenceResolver Integration Test
# -------------------------------------------------------------------------

def test_evidence_resolver_with_reranker() -> None:
    # Create a synthetic 2-page document
    tok1 = DocumentToken(text="100.00", bbox=BBox(0.20, 0.50, 0.05, 0.012, page=1), page=1, char_index_in_page=0)
    line1 = VisualLine(tokens=[tok1], page=1, line_index=0, bbox=tok1.bbox)
    p1 = DocumentPage(page_number=1, width=100.0, height=100.0, tokens=[tok1], lines=[line1])

    tok2 = DocumentToken(text="100.00", bbox=BBox(0.20, 0.50, 0.05, 0.012, page=2), page=2, char_index_in_page=0)
    line2 = VisualLine(tokens=[tok2], page=2, line_index=0, bbox=tok2.bbox)
    p2 = DocumentPage(page_number=2, width=100.0, height=100.0, tokens=[tok2], lines=[line2])

    index = DocumentIndex([p1, p2])
    resolver = EvidenceResolver(index=index, enable_verification=True)

    # Resolve with target_page hint = 2
    inp = ExtractionInput(
        field="total_amount",
        value=100.00,
        page_hint=2,
        target_page=2,
        page_confidence=1.0,
    )

    res = resolver.resolve(inp)
    assert res.is_grounded
    assert res.page == 2
    assert res.status == ProvenanceStatus.NORMALIZED
