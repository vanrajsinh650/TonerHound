"""Unit tests for CandidateVerifier (EXP-003 Steps 4, 5, 6, 7, 8)."""

import pytest
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import ProvenanceStatus
from tonerhound.resolution.verifier import CandidateVerifier, FieldType, infer_field_type


def test_field_type_inference():
    assert infer_field_type("is_active", True) == FieldType.BOOLEAN
    assert infer_field_type("presidential_campaign_you_box", False) == FieldType.BOOLEAN
    assert infer_field_type("total_amount", 125.50) == FieldType.NUMERIC
    assert infer_field_type("invoice_total", "$1,450.00") == FieldType.CURRENCY
    assert infer_field_type("measurement_date", "2026-03-15") == FieldType.DATE
    assert infer_field_type("patient_ssn", "123-45-6789") == FieldType.IDENTIFIER
    assert infer_field_type("items[0].description", "Consulting service") == FieldType.TABLE_CELL


def test_verifier_rejects_out_of_bounds_bbox():
    verifier = CandidateVerifier()
    cand = MatchCandidate(
        page=1,
        bbox=BBox(-0.1, 0.5, 0.2, 0.05, page=1),
        tokens=(),
        matched_text="Test",
        match_type="exact",
        raw_similarity=1.0,
    )
    decision = verifier.verify("field", "Test", cand, [(cand, 5.0)])
    assert not decision.is_accepted
    assert decision.status == ProvenanceStatus.NOT_FOUND


def test_verifier_rejects_overwide_scalar():
    verifier = CandidateVerifier()
    cand = MatchCandidate(
        page=1,
        bbox=BBox(0.05, 0.5, 0.95, 0.05, page=1),  # width 0.95 > max_scalar_width
        tokens=(),
        matched_text="100.00",
        match_type="normalized_number",
        raw_similarity=1.0,
    )
    decision = verifier.verify("subtotal", 100.00, cand, [(cand, 5.0)])
    assert not decision.is_accepted
    assert decision.status == ProvenanceStatus.NOT_FOUND


def test_verifier_abstains_on_close_score_margin():
    verifier = CandidateVerifier(score_margin_threshold=0.05)
    # Two candidates with nearly identical score across distinct regions
    cand1 = MatchCandidate(
        page=1,
        bbox=BBox(0.1, 0.2, 0.05, 0.02, page=1),
        tokens=(),
        matched_text="50.00",
        match_type="normalized_number",
        raw_similarity=1.0,
    )
    cand2 = MatchCandidate(
        page=1,
        bbox=BBox(0.1, 0.6, 0.05, 0.02, page=1),
        tokens=(),
        matched_text="50.00",
        match_type="normalized_number",
        raw_similarity=1.0,
    )
    scored = [(cand1, 5.02), (cand2, 5.00)]  # margin 0.02 < 0.05
    decision = verifier.verify("tax", 50.0, cand1, scored)
    assert not decision.is_accepted
    assert decision.status == ProvenanceStatus.AMBIGUOUS


def test_verifier_accepts_when_duplicates_overlap_same_region():
    verifier = CandidateVerifier(score_margin_threshold=0.05)
    # Two candidates in the same visual region (IoU >= 0.50)
    cand1 = MatchCandidate(
        page=1,
        bbox=BBox(0.1, 0.2, 0.05, 0.02, page=1),
        tokens=(),
        matched_text="50.00",
        match_type="normalized_number",
        raw_similarity=1.0,
    )
    cand2 = MatchCandidate(
        page=1,
        bbox=BBox(0.101, 0.201, 0.049, 0.019, page=1),
        tokens=(),
        matched_text="50.00",
        match_type="normalized_number",
        raw_similarity=1.0,
    )
    scored = [(cand1, 5.01), (cand2, 5.00)]
    decision = verifier.verify("tax", 50.0, cand1, scored)
    assert decision.is_accepted
    assert decision.status == ProvenanceStatus.NORMALIZED


def test_verifier_rejects_row_anchor_violation():
    verifier = CandidateVerifier()
    cand = MatchCandidate(
        page=1,
        bbox=BBox(0.1, 0.70, 0.05, 0.02, page=1),
        tokens=(),
        matched_text="07:45",
        match_type="exact",
        raw_similarity=1.0,
    )
    # Row anchor is at y=0.30, cand is at y=0.70
    row_anchor = (1, 0.30, 0.02, BBox(0.05, 0.30, 0.1, 0.02, page=1))
    decision = verifier.verify("items[0].time", "07:45", cand, [(cand, 6.0)], row_anchor=row_anchor)
    assert not decision.is_accepted
    assert decision.status == ProvenanceStatus.NOT_FOUND


def test_verifier_rejects_non_checkbox_for_boolean():
    verifier = CandidateVerifier()
    cand = MatchCandidate(
        page=1,
        bbox=BBox(0.1, 0.2, 0.05, 0.02, page=1),
        tokens=(),
        matched_text="Federal",  # Non-checkbox word
        match_type="exact",
        raw_similarity=1.0,
    )
    decision = verifier.verify("presidential_campaign_you_box", False, cand, [(cand, 5.0)])
    assert not decision.is_accepted
    assert decision.status == ProvenanceStatus.NOT_FOUND
