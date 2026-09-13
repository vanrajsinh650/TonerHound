"""Tests for structural table disambiguation and row-level evidence grounding."""

from __future__ import annotations

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


def test_table_row_anchoring_disambiguation() -> None:
    """Test that repeated identical values across rows are resolved to their own rows."""
    # Create a 2-row table fixture where both rows have bank="01" and type="C",
    # but row 0 has check_num="1001" and row 1 has check_num="1002".
    # Row 0 at y=0.10
    r0_bank = DocumentToken(text="01", bbox=BBox(0.05, 0.10, 0.04, 0.02, 1), page=1, char_index_in_page=0)
    r0_num = DocumentToken(text="1001", bbox=BBox(0.12, 0.10, 0.08, 0.02, 1), page=1, char_index_in_page=3)
    r0_type = DocumentToken(text="C", bbox=BBox(0.22, 0.10, 0.02, 0.02, 1), page=1, char_index_in_page=8)
    line0 = VisualLine(tokens=[r0_bank, r0_num, r0_type], page=1, line_index=0, bbox=BBox(0.05, 0.10, 0.19, 0.02, 1))

    # Row 1 at y=0.15
    r1_bank = DocumentToken(text="01", bbox=BBox(0.05, 0.15, 0.04, 0.02, 1), page=1, char_index_in_page=11)
    r1_num = DocumentToken(text="1002", bbox=BBox(0.12, 0.15, 0.08, 0.02, 1), page=1, char_index_in_page=14)
    r1_type = DocumentToken(text="C", bbox=BBox(0.22, 0.15, 0.02, 0.02, 1), page=1, char_index_in_page=19)
    line1 = VisualLine(tokens=[r1_bank, r1_num, r1_type], page=1, line_index=1, bbox=BBox(0.05, 0.15, 0.19, 0.02, 1))

    all_tokens = [r0_bank, r0_num, r0_type, r1_bank, r1_num, r1_type]
    page = DocumentPage(page_number=1, width=600, height=800, tokens=all_tokens, lines=[line0, line1])
    idx = DocumentIndex.from_pages([page])

    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)
    payload = {
        "checks": [
            {"bank": "01", "check_number": "1001", "type": "C"},
            {"bank": "01", "check_number": "1002", "type": "C"},
        ]
    }

    result = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in result["field_citations"]}

    # Check 0 bank must match line 0
    assert "checks[0].bank" in citations
    assert abs(citations["checks[0].bank"]["bbox"][1] - 0.10) < 0.01

    # Check 1 bank must match line 1, NOT line 0!
    assert "checks[1].bank" in citations
    assert abs(citations["checks[1].bank"]["bbox"][1] - 0.15) < 0.01

    # Check 1 type must match line 1
    assert "checks[1].type" in citations
    assert abs(citations["checks[1].type"]["bbox"][1] - 0.15) < 0.01
