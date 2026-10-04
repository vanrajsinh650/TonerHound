"""V3 Adversarial Injection Tests for Failure Microscope (Section 16).

Verifies that controlled cases of known root causes are 100% correctly classified:
- 2 genuine non-text boolean fields -> NON_TEXT_BOOLEAN_GROUNDING
- 2 non-standard date failures -> DATE_INDEX_MISS
- 2 text-layer-empty fields -> NO_TEXT_AT_GOLD_REGION
- 2 wrong-row cases -> WRONG_ROW
- 2 bbox-too-narrow cases -> BBOX_TOO_NARROW
- 2 already-resolved cases -> ALREADY_RESOLVED
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
from research.observer.field_classifier import (
    FailureMicroscopeClassifier,
    compute_iou_xywh,
)
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine, ResolutionResult
from tonerhound.document.index import DocumentIndex


def _create_mock_index(tokens: list[tuple[str, tuple[float, float, float, float]]], page_num: int = 1) -> DocumentIndex:
    """Create a minimal synthetic DocumentIndex for controlled adversarial tests."""
    doc_tokens: list[DocumentToken] = []
    char_idx = 0
    for txt, (x, y, w, h) in tokens:
        tok = DocumentToken(
            text=txt,
            bbox=BBox(x=x, y=y, width=w, height=h, page=page_num),
            page=page_num,
            char_index_in_page=char_idx,
            line_index=0,
        )
        char_idx += len(txt) + 1
        doc_tokens.append(tok)

    page = DocumentPage(
        page_number=page_num,
        width=1000.0,
        height=1000.0,
        tokens=doc_tokens,
        lines=[VisualLine(tokens=doc_tokens, page=page_num, line_index=0, bbox=BBox(0, 0, 1, 1, page_num))],
    )
    return DocumentIndex([page])


def test_v3_adversarial_injection_all_cases():
    """Run all 12 controlled injection cases and assert 100% exact classification."""
    # 1. Two Genuine Non-Text Boolean Fields
    idx_empty = _create_mock_index([])
    clf_bool = FailureMicroscopeClassifier(doc_index=idx_empty)

    # Case 1: Python bool True with no text
    res1 = clf_bool.classify_field(
        document_id="doc_adv_1",
        field_path="irs_1040.presidential_election_campaign_yes_box",
        gold_value=True,
        predicted_value=None,
        gold_page=1,
        predicted_page=None,
        gold_bbox=[0.82, 0.27, 0.03, 0.015],
        predicted_bbox=None,
        candidates=[],
    )
    assert res1.failure_class == "NON_TEXT_BOOLEAN_GROUNDING"

    # Case 2: String 'False' on checkbox field with no matching text
    res2 = clf_bool.classify_field(
        document_id="doc_adv_2",
        field_path="w14.productive_zone_no",
        gold_value="False",
        predicted_value=None,
        gold_page=1,
        predicted_page=None,
        gold_bbox=[0.51, 0.63, 0.02, 0.013],
        predicted_bbox=None,
        candidates=[],
    )
    assert res2.failure_class == "NON_TEXT_BOOLEAN_GROUNDING"

    # 2. Two Non-Standard Date Failures (tokens present at gold box, but date indexing failed)
    idx_date = _create_mock_index([
        ("11/04/2025", (0.70, 0.05, 0.10, 0.02)),
        ("October", (0.60, 0.08, 0.05, 0.02)),
        ("3,", (0.66, 0.08, 0.02, 0.02)),
        ("2026", (0.69, 0.08, 0.03, 0.02)),
    ])
    clf_date = FailureMicroscopeClassifier(doc_index=idx_date)

    # Case 3: Date string '11/04/2025' parsed as date, but 0 candidates found
    res3 = clf_date.classify_field(
        document_id="doc_adv_3",
        field_path="recording_date",
        gold_value="11/04/2025",
        predicted_value=None,
        gold_page=1,
        predicted_page=None,
        gold_bbox=[0.70, 0.05, 0.10, 0.02],
        predicted_bbox=None,
        candidates=[],
    )
    assert res3.failure_class == "DATE_INDEX_MISS"

    # Case 4: Date string 'October 3, 2026'
    res4 = clf_date.classify_field(
        document_id="doc_adv_4",
        field_path="effective_date",
        gold_value="October 3, 2026",
        predicted_value=None,
        gold_page=1,
        predicted_page=None,
        gold_bbox=[0.60, 0.08, 0.12, 0.02],
        predicted_bbox=None,
        candidates=[],
    )
    assert res4.failure_class == "DATE_INDEX_MISS"

    # 3. Two Text-Layer-Empty Fields (non-boolean string/numeric with 0 tokens at gold bbox)
    # Case 5: Scanned receipt total amount in bitmap region
    res5 = clf_date.classify_field(
        document_id="doc_adv_5",
        field_path="receipt_total_amount",
        gold_value="1250.00",
        predicted_value=None,
        gold_page=1,
        predicted_page=None,
        gold_bbox=[0.85, 0.90, 0.10, 0.02],  # coordinates with 0 tokens
        predicted_bbox=None,
        candidates=[],
    )
    assert res5.failure_class == "NO_TEXT_AT_GOLD_REGION"

    # Case 6: Scanned vendor name in image header
    res6 = clf_date.classify_field(
        document_id="doc_adv_6",
        field_path="vendor_legal_name",
        gold_value="Acme Industrial Corp",
        predicted_value=None,
        gold_page=1,
        predicted_page=None,
        gold_bbox=[0.05, 0.02, 0.30, 0.03],  # coordinates with 0 tokens
        predicted_bbox=None,
        candidates=[],
    )
    assert res6.failure_class == "NO_TEXT_AT_GOLD_REGION"

    # 4. Two Wrong-Row Cases (Table array items where candidate pool has IoU>=0.50, but selected wrong row)
    idx_table = _create_mock_index([
        ("Row0_Val", (0.50, 0.20, 0.10, 0.02)),
        ("Row1_Val", (0.50, 0.25, 0.10, 0.02)),
        ("Row3_Val", (0.60, 0.40, 0.08, 0.02)),
        ("Row4_Val", (0.60, 0.45, 0.08, 0.02)),
    ])
    clf_table = FailureMicroscopeClassifier(doc_index=idx_table)

    cand_row0 = MatchCandidate(
        page=1,
        bbox=BBox(x=0.50, y=0.20, width=0.10, height=0.02, page=1),
        tokens=(),
        matched_text="100.00",
        match_type="exact",
        raw_similarity=0.90,
    )
    cand_row1 = MatchCandidate(
        page=1,
        bbox=BBox(x=0.50, y=0.25, width=0.10, height=0.02, page=1),
        tokens=(),
        matched_text="100.00",
        match_type="exact",
        raw_similarity=0.88,
    )

    # Case 7: Target is line_items[1], gold is at y=0.25, but resolver selected y=0.20 (row 0)
    res7 = clf_table.classify_field(
        document_id="doc_adv_7",
        field_path="line_items[1].amount",
        gold_value="100.00",
        predicted_value="100.00",
        gold_page=1,
        predicted_page=1,
        gold_bbox=[0.50, 0.25, 0.10, 0.02],
        predicted_bbox=[0.50, 0.20, 0.10, 0.02],  # Selected row 0!
        candidates=[cand_row0, cand_row1],
    )
    assert res7.failure_class == "WRONG_ROW"

    # Case 8: Target is transactions[4], gold is at y=0.45, resolver selected y=0.40
    res8 = clf_table.classify_field(
        document_id="doc_adv_8",
        field_path="transactions[4].price",
        gold_value="50.00",
        predicted_value="50.00",
        gold_page=1,
        predicted_page=1,
        gold_bbox=[0.60, 0.45, 0.08, 0.02],
        predicted_bbox=[0.60, 0.40, 0.08, 0.02],
        candidates=[
            MatchCandidate(1, BBox(0.60, 0.45, 0.08, 0.02, 1), (), "50.00", "exact", 0.9),
            MatchCandidate(1, BBox(0.60, 0.40, 0.08, 0.02, 1), (), "50.00", "exact", 0.9),
        ],
    )
    assert res8.failure_class == "WRONG_ROW"

    # 5. Two Bbox-Too-Narrow Cases (candidates exist on page, but cand_w / gold_w < 0.70)
    idx_words = _create_mock_index([
        ("International", (0.20, 0.30, 0.10, 0.02)),
        ("Business", (0.31, 0.30, 0.08, 0.02)),
        ("Machines", (0.40, 0.30, 0.08, 0.02)),
        ("Post", (0.10, 0.50, 0.05, 0.02)),
        ("Holdings", (0.16, 0.50, 0.08, 0.02)),
        ("Inc", (0.25, 0.50, 0.04, 0.02)),
    ])
    clf_words = FailureMicroscopeClassifier(doc_index=idx_words)

    # Case 9: Gold box covers "International Business Machines" (w=0.28), but candidate only covers "Machines" (w=0.08)
    cand_narrow_1 = MatchCandidate(1, BBox(0.40, 0.30, 0.08, 0.02, 1), (), "Machines", "exact", 0.7)
    res9 = clf_words.classify_field(
        document_id="doc_adv_9",
        field_path="company_name",
        gold_value="International Business Machines",
        predicted_value="Machines",
        gold_page=1,
        predicted_page=1,
        gold_bbox=[0.20, 0.30, 0.28, 0.02],  # gold w = 0.28
        predicted_bbox=[0.40, 0.30, 0.08, 0.02],  # ratio = 0.08 / 0.28 = 0.285 < 0.70
        candidates=[cand_narrow_1],
    )
    assert res9.failure_class == "BBOX_TOO_NARROW"

    # Case 10: Gold box covers 3 words w=0.20, candidate only 1 word w=0.05
    cand_narrow_2 = MatchCandidate(1, BBox(0.10, 0.50, 0.05, 0.02, 1), (), "Post", "exact", 0.7)
    res10 = clf_words.classify_field(
        document_id="doc_adv_10",
        field_path="entity_name",
        gold_value="Post Holdings Inc",
        predicted_value="Post",
        gold_page=1,
        predicted_page=1,
        gold_bbox=[0.10, 0.50, 0.20, 0.02],
        predicted_bbox=[0.10, 0.50, 0.05, 0.02],
        candidates=[cand_narrow_2],
    )
    assert res10.failure_class == "BBOX_TOO_NARROW"

    # 6. Two Already-Resolved Cases (grounding_success == True, IoU >= 0.50 on same page)
    # Case 11: Perfect match IoU = 1.0
    res11 = clf_words.classify_field(
        document_id="doc_adv_11",
        field_path="total_amount",
        gold_value="100.00",
        predicted_value="100.00",
        gold_page=1,
        predicted_page=1,
        gold_bbox=[0.50, 0.50, 0.10, 0.02],
        predicted_bbox=[0.50, 0.50, 0.10, 0.02],
        candidates=[],
    )
    assert res11.failure_class == "ALREADY_RESOLVED"
    assert res11.grounding_success is True

    # Case 12: High IoU = 0.85
    res12 = clf_words.classify_field(
        document_id="doc_adv_12",
        field_path="invoice_number",
        gold_value="INV-2026-001",
        predicted_value="INV-2026-001",
        gold_page=2,
        predicted_page=2,
        gold_bbox=[0.30, 0.10, 0.15, 0.025],
        predicted_bbox=[0.302, 0.101, 0.148, 0.024],
        candidates=[],
    )
    assert res12.failure_class == "ALREADY_RESOLVED"
    assert res12.grounding_success is True
