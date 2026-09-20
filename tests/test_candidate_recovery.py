"""Unit tests for EXP-013 Candidate Generation Recovery Engine.

Tests each of the 5 priority failure classes:
1. Spaced-token numeric recovery ("3 3 . 3 3 3 3 %" -> 33.3333%)
2. OCR-fragmented numeric values (split across multiple tokens/glyphs)
3. Split currency and percentage tokens
4. Fragmented multi-token field values (interleaved sub-rows)
5. Bounded character-span matching with missing anchor tokens
6. Critical diagnostic recording
"""

from __future__ import annotations

import pytest

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.candidate_recovery import (
    CandidateRecoveryEngine,
    RecoveredDiagnostic,
)
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


def _create_synthetic_index(lines_data: list[list[tuple[str, tuple[float, float, float, float]]]]) -> DocumentIndex:
    """Helper to build DocumentIndex with exact token geometries."""
    lines: list[VisualLine] = []
    page_tokens: list[DocumentToken] = []

    for l_idx, line_tokens_info in enumerate(lines_data):
        line_toks: list[DocumentToken] = []
        for char_idx, (txt, (x, y, w, h)) in enumerate(line_tokens_info):
            tok = DocumentToken(
                text=txt,
                bbox=BBox(x=x, y=y, width=w, height=h, page=1),
                page=1,
                char_index_in_page=char_idx,
                line_index=l_idx,
            )
            line_toks.append(tok)
            page_tokens.append(tok)

        min_x = min(t.bbox.x for t in line_toks)
        min_y = min(t.bbox.y for t in line_toks)
        max_x = max(t.bbox.x + t.bbox.width for t in line_toks)
        max_y = max(t.bbox.y + t.bbox.height for t in line_toks)
        line_bbox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y, page=1)

        lines.append(
            VisualLine(
                tokens=line_toks,
                page=1,
                line_index=l_idx,
                bbox=line_bbox,
            )
        )

    page = DocumentPage(
        page_number=1,
        width=1000.0,
        height=1000.0,
        tokens=page_tokens,
        lines=lines,
    )
    return DocumentIndex.from_pages([page])


def test_recovery_class_1_spaced_token_numeric() -> None:
    """Class 1: '3 3 . 3 3 3 3 %' should be recovered as 33.3333%."""
    # Line with "Profit 3 3 . 3 3 3 3 %"
    tokens_info = [
        ("Profit", (0.10, 0.65, 0.05, 0.01)),
        ("3", (0.19, 0.65, 0.01, 0.01)),
        ("3", (0.20, 0.65, 0.01, 0.01)),
        (".", (0.21, 0.65, 0.005, 0.01)),
        ("3", (0.215, 0.65, 0.01, 0.01)),
        ("3", (0.225, 0.65, 0.01, 0.01)),
        ("3", (0.235, 0.65, 0.01, 0.01)),
        ("3", (0.245, 0.65, 0.01, 0.01)),
        ("%", (0.258, 0.65, 0.01, 0.01)),
    ]
    idx = _create_synthetic_index([tokens_info])
    engine = CandidateRecoveryEngine(idx)

    # Search for numeric 33.3333
    cands = engine.recover(value=33.3333, page_hint=1, field_name="profit_share")
    assert len(cands) >= 1
    best = cands[0]
    assert "33.3333" in best.matched_text
    assert best.match_type in ("recovered_spaced_numeric", "recovered_fragmented_numeric")
    # Bounding box should span from the first '3' to '%'
    assert pytest.approx(best.bbox.x, abs=1e-3) == 0.19
    assert pytest.approx(best.bbox.x + best.bbox.width, abs=1e-3) == 0.268


def test_recovery_class_2_ocr_fragmented_numeric() -> None:
    """Class 2: Reconstruct values split across multiple OCR tokens (e.g. '-3' + '186' -> -3186)."""
    tokens_info = [
        ("Box1", (0.10, 0.12, 0.05, 0.01)),
        ("-3", (0.64, 0.12, 0.02, 0.01)),
        ("186", (0.665, 0.12, 0.03, 0.01)),
    ]
    idx = _create_synthetic_index([tokens_info])
    engine = CandidateRecoveryEngine(idx)

    cands = engine.recover(value=-3186, page_hint=1, field_name="box_1_income")
    assert len(cands) >= 1
    best = cands[0]
    assert best.matched_text == "-3186"
    assert best.match_type == "recovered_fragmented_numeric"
    assert pytest.approx(best.bbox.x, abs=1e-3) == 0.64
    assert pytest.approx(best.bbox.x + best.bbox.width, abs=1e-3) == 0.695


def test_recovery_class_2_thousands_separator_fragment() -> None:
    """Class 2: Reconstruct '21,' + '693.' -> 21693."""
    tokens_info = [
        ("Balance", (0.10, 0.50, 0.05, 0.01)),
        ("21,", (0.40, 0.50, 0.025, 0.01)),
        ("693.", (0.43, 0.50, 0.030, 0.01)),
    ]
    idx = _create_synthetic_index([tokens_info])
    engine = CandidateRecoveryEngine(idx)

    cands = engine.recover(value=21693, page_hint=1, field_name="ending_balance")
    assert len(cands) >= 1
    best = cands[0]
    assert "21,693." in best.matched_text
    assert best.match_type == "recovered_fragmented_numeric"


def test_recovery_class_3_split_percentage_symbol() -> None:
    """Class 3: Number in token i, '%' in token i+1 -> union includes '%'."""
    tokens_info = [
        ("Share:", (0.10, 0.40, 0.05, 0.01)),
        ("33.3333333", (0.20, 0.40, 0.08, 0.01)),
        ("%", (0.285, 0.40, 0.01, 0.01)),
    ]
    idx = _create_synthetic_index([tokens_info])
    engine = CandidateRecoveryEngine(idx)

    cands = engine.recover(value=33.3333333, page_hint=1, field_name="capital_share")
    assert len(cands) >= 1
    best = cands[0]
    assert "%" in best.matched_text
    # Width must cover both the number and the percent sign
    assert pytest.approx(best.bbox.x + best.bbox.width, abs=1e-3) == 0.295


def test_recovery_class_4_interleaved_sub_row_tokens() -> None:
    """Class 4: Multi-token value horizontally interleaved with another sub-row."""
    # Line containing interleaved tokens:
    # y ≈ 0.269: TELCO (x=0.08), EXPERTS (x=0.15), LLC (x=0.24)
    # y ≈ 0.284: 38 (x=0.08), PARK (x=0.11), AVENUE (x=0.17)
    tokens_info = [
        ("TELCO", (0.08, 0.269, 0.05, 0.01)),
        ("38", (0.08, 0.284, 0.02, 0.01)),
        ("PARK", (0.11, 0.284, 0.04, 0.01)),
        ("EXPERTS", (0.15, 0.269, 0.07, 0.01)),
        ("AVENUE", (0.17, 0.284, 0.06, 0.01)),
        ("LLC", (0.24, 0.269, 0.03, 0.01)),
    ]
    idx = _create_synthetic_index([tokens_info])
    engine = CandidateRecoveryEngine(idx)

    cands = engine.recover(value="TELCO EXPERTS LLC", page_hint=1, field_name="partnership_name")
    assert len(cands) >= 1
    best = cands[0]
    assert best.matched_text == "TELCO EXPERTS LLC"
    assert best.match_type == "recovered_interleaved"
    # Should only span from x=0.08 to x=0.27 at y=0.269
    assert pytest.approx(best.bbox.x, abs=1e-3) == 0.08
    assert pytest.approx(best.bbox.x + best.bbox.width, abs=1e-3) == 0.27
    assert pytest.approx(best.bbox.y, abs=1e-3) == 0.269


def test_recovery_class_5_bounded_span_missing_anchor() -> None:
    """Class 5: Bounded span recovery when 1 token is corrupted/missing in OCR."""
    # Query: "Fruitland Park FL 34731"
    # Line has OCR artifact: ["itland", "Park", "FL", "34731"] (dropped "Fru")
    tokens_info = [
        ("itland", (0.10, 0.30, 0.05, 0.01)),
        ("Park", (0.16, 0.30, 0.04, 0.01)),
        ("FL", (0.21, 0.30, 0.02, 0.01)),
        ("34731", (0.24, 0.30, 0.04, 0.01)),
    ]
    idx = _create_synthetic_index([tokens_info])
    engine = CandidateRecoveryEngine(idx)

    cands = engine.recover(
        value="Fruitland Park FL 34731",
        page_hint=1,
        field_name="address",
        has_exact_match=False,
    )
    assert len(cands) >= 1
    best = cands[0]
    assert "34731" in best.matched_text
    assert best.match_type == "recovered_bounded_span"
    assert pytest.approx(best.bbox.x, abs=1e-3) == 0.10
    assert pytest.approx(best.bbox.x + best.bbox.width, abs=1e-3) == 0.28


def test_recovery_diagnostics_recorded() -> None:
    """Verify that every recovered candidate generates a structured RecoveredDiagnostic record."""
    tokens_info = [
        ("-3", (0.64, 0.12, 0.02, 0.01)),
        ("186", (0.665, 0.12, 0.03, 0.01)),
    ]
    idx = _create_synthetic_index([tokens_info])
    engine = CandidateRecoveryEngine(idx)

    cands = engine.recover(value=-3186, page_hint=1, field_name="box_1")
    assert len(cands) >= 1
    assert len(engine.diagnostics) >= 1

    diag = engine.diagnostics[0]
    assert isinstance(diag, RecoveredDiagnostic)
    assert diag.original_value == -3186
    assert diag.raw_document_tokens == ("-3", "186")
    assert diag.reconstructed_candidate == "-3186"
    assert len(diag.candidate_bbox) == 4
    assert diag.normalization_path in ("adjacent_token_concat_parse_numeric", "whitespace_elision_parse_numeric")
    assert "fragmented" in diag.why_previously_missed.lower() or "separated" in diag.why_previously_missed.lower()
