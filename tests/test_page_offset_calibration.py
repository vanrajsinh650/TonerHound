"""Unit and regression tests for systematic page-offset calibration."""

from __future__ import annotations

from tonerhound import DocumentIndex, DocumentPage, DocumentToken, VisualLine
from tonerhound.benchmark.adapter import ExtractBenchAdapter, OffsetCalibrationResult
from tonerhound.geometry.coordinates import BBox


def _create_synthetic_document(num_pages: int, page_contents: dict[int, list[str]]) -> DocumentIndex:
    """Create a synthetic DocumentIndex with specified text lines on given pages."""
    pages: list[DocumentPage] = []
    for p_num in range(1, num_pages + 1):
        lines_text = page_contents.get(p_num, [])
        tokens: list[DocumentToken] = []
        lines: list[VisualLine] = []
        char_offset = 0
        for l_idx, text in enumerate(lines_text):
            line_tokens: list[DocumentToken] = []
            words = text.split()
            x_cursor = 0.1
            y_cursor = 0.1 + l_idx * 0.05
            for w in words:
                tok_box = BBox(x_cursor, y_cursor, 0.08, 0.025, page=p_num)
                tok = DocumentToken(
                    text=w,
                    bbox=tok_box,
                    page=p_num,
                    char_index_in_page=char_offset,
                    line_index=l_idx,
                )
                line_tokens.append(tok)
                tokens.append(tok)
                x_cursor += 0.09
                char_offset += len(w) + 1

            if line_tokens:
                line_box = BBox(0.1, y_cursor, max(0.1, x_cursor - 0.1), 0.025, page=p_num)
                lines.append(VisualLine(tokens=line_tokens, page=p_num, line_index=l_idx, bbox=line_box))

        pages.append(DocumentPage(page_number=p_num, width=612, height=792, tokens=tokens, lines=lines))

    return DocumentIndex.from_pages(pages)


def test_scenario_1_zero_offset() -> None:
    """1. Verify accurate detection of zero offset when page hints match physical pages."""
    idx = _create_synthetic_document(
        num_pages=5,
        page_contents={
            1: ["Acme Holdings Corporation", "Industrial Solutions Group", "Contract Number CN-4029"],
            2: ["Schedule of Assets", "Inventory Valuation Summary", "Fiscal Year 2025"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    leaves = [
        ("company_name", "Acme Holdings Corporation", 1, "", "root"),
        ("division", "Industrial Solutions Group", 1, "", "root"),
        ("contract_id", "Contract Number CN-4029", 1, "", "root"),
        ("schedule_title", "Schedule of Assets", 2, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    assert res.offset == 0
    assert res.confidence >= 0.6
    assert adapter._detect_page_offset(leaves) == 0


def test_scenario_2_plus_one_offset() -> None:
    """2. Verify accurate detection of +1 page offset (e.g. cover page shift)."""
    idx = _create_synthetic_document(
        num_pages=6,
        page_contents={
            1: ["Document Cover Page", "Table of Contents"],
            2: ["Acme Holdings Corporation", "Industrial Solutions Group", "Contract Number CN-4029"],
            3: ["Schedule of Assets", "Inventory Valuation Summary"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    leaves = [
        ("company_name", "Acme Holdings Corporation", 1, "", "root"),
        ("division", "Industrial Solutions Group", 1, "", "root"),
        ("contract_id", "Contract Number CN-4029", 1, "", "root"),
        ("schedule_title", "Schedule of Assets", 2, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    assert res.offset == 1
    assert res.confidence >= 0.75
    assert res.tier == "tier1_narrow"
    assert adapter._detect_page_offset(leaves) == 1


def test_scenario_3_plus_two_offset() -> None:
    """3. Verify accurate detection of +2 page offset."""
    idx = _create_synthetic_document(
        num_pages=7,
        page_contents={
            1: ["Document Cover Page"],
            2: ["Executive Notice"],
            3: ["Acme Holdings Corporation", "Industrial Solutions Group", "Contract Number CN-4029"],
            4: ["Schedule of Assets", "Inventory Valuation Summary"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    leaves = [
        ("company_name", "Acme Holdings Corporation", 1, "", "root"),
        ("division", "Industrial Solutions Group", 1, "", "root"),
        ("contract_id", "Contract Number CN-4029", 1, "", "root"),
        ("schedule_title", "Schedule of Assets", 2, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    assert res.offset == 2
    assert res.confidence >= 0.75
    assert adapter._detect_page_offset(leaves) == 2


def test_scenario_4_plus_seven_offset() -> None:
    """4. Verify detection of +7 offset (real_credit_strategies_full benchmark case)."""
    idx = _create_synthetic_document(
        num_pages=20,
        page_contents={
            8: ["Asset-Backed Securities", "720 East CLO Ltd.", "Series 2022-1A"],
            9: ["Abry Liquid Credit CLO", "Series 2025-2A", "Maturity 01/15/39"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    leaves = [
        ("holdings[0].asset_class", "Asset-Backed Securities", 1, "", "holdings[0]"),
        ("holdings[0].issuer", "720 East CLO Ltd.", 1, "", "holdings[0]"),
        ("holdings[0].series", "Series 2022-1A", 1, "", "holdings[0]"),
        ("holdings[1].issuer", "Abry Liquid Credit CLO", 2, "", "holdings[1]"),
        ("holdings[1].series", "Series 2025-2A", 2, "", "holdings[1]"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    assert res.offset == 7
    assert res.confidence >= 0.75
    assert res.tier == "tier2_medium"
    assert adapter._detect_page_offset(leaves) == 7


def test_scenario_5_large_positive_offset() -> None:
    """5. Verify detection of large positive offset (+24) in wide search tier."""
    idx = _create_synthetic_document(
        num_pages=35,
        page_contents={
            25: ["Exhibit B Schedule", "Underwriting Agreement", "Syndication Agent"],
            26: ["Credit Facility Provisions", "Revolving Commitment"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    leaves = [
        ("title", "Exhibit B Schedule", 1, "", "root"),
        ("agreement", "Underwriting Agreement", 1, "", "root"),
        ("agent", "Syndication Agent", 1, "", "root"),
        ("provisions", "Credit Facility Provisions", 2, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    assert res.offset == 24
    assert res.confidence >= 0.70
    assert res.tier == "tier3_wide"
    assert adapter._detect_page_offset(leaves) == 24


def test_scenario_6_negative_offset() -> None:
    """6. Verify detection of negative page offset (-2)."""
    idx = _create_synthetic_document(
        num_pages=10,
        page_contents={
            2: ["Prior Section Notes"],
            3: ["Acme Holdings Corporation", "Industrial Solutions Group", "Contract Number CN-4029"],
            4: ["Schedule of Assets", "Inventory Valuation Summary"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    # Logical hints report page 5 and 6, but content is physically on pages 3 and 4 (offset -2)
    leaves = [
        ("company_name", "Acme Holdings Corporation", 5, "", "root"),
        ("division", "Industrial Solutions Group", 5, "", "root"),
        ("contract_id", "Contract Number CN-4029", 5, "", "root"),
        ("schedule_title", "Schedule of Assets", 6, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    assert res.offset == -2
    assert res.confidence >= 0.75
    assert adapter._detect_page_offset(leaves) == -2


def test_scenario_7_sparse_anchors_safely_degrades() -> None:
    """7. Verify that sparse anchors (insufficient consensus) safely degrade to 0."""
    idx = _create_synthetic_document(
        num_pages=10,
        page_contents={
            4: ["Solitary Match Candidate"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    # Only 1 anchor candidate with non-zero offset
    leaves = [
        ("solitary_field", "Solitary Match Candidate", 1, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    # With only 1 anchor, we cannot reliably prove an offset away from 0
    assert res.offset == 0
    assert adapter._detect_page_offset(leaves) == 0


def test_scenario_8_repeated_boilerplate_anchors_suppressed() -> None:
    """8. Verify that repeated boilerplate strings do not skew offset calibration."""
    # "COMMON STOCK" and "UNITED STATES" appear on every page as boilerplate headers
    idx = _create_synthetic_document(
        num_pages=10,
        page_contents={
            p: ["COMMON STOCK", "UNITED STATES", "QUARTERLY REPORT"]
            for p in range(1, 11)
        },
    )
    # Add real distinct anchor on page 4
    idx.pages[3].lines.append(
        VisualLine(
            tokens=[
                DocumentToken(
                    "Alpha Distinctive Investment",
                    BBox(0.1, 0.5, 0.4, 0.02, page=4),
                    page=4,
                    char_index_in_page=0,
                    line_index=3,
                )
            ],
            page=4,
            line_index=3,
            bbox=BBox(0.1, 0.5, 0.4, 0.02, page=4),
        )
    )

    adapter = ExtractBenchAdapter(idx)
    leaves = [
        ("type", "COMMON STOCK", 1, "", "root"),
        ("country", "UNITED STATES", 1, "", "root"),
        ("report", "QUARTERLY REPORT", 1, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    # Boilerplate should not produce a non-zero spurious offset
    assert res.offset == 0


def test_scenario_9_ambiguous_offset_falls_back_to_zero() -> None:
    """9. Verify that tied/ambiguous offset candidates fall back safely to 0."""
    idx = _create_synthetic_document(
        num_pages=10,
        page_contents={
            3: ["Candidate Group Alpha", "Unique Alpha Token"],
            5: ["Candidate Group Beta", "Unique Beta Token"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    # Two anchors pointing to offset +2 (page 1 -> 3) and two pointing to offset +4 (page 1 -> 5)
    leaves = [
        ("field1", "Candidate Group Alpha", 1, "", "root"),
        ("field2", "Unique Alpha Token", 1, "", "root"),
        ("field3", "Candidate Group Beta", 1, "", "root"),
        ("field4", "Unique Beta Token", 1, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    # Because margin is tied between +2 and +4, system must fall back safely to 0
    assert res.offset == 0


def test_scenario_10_no_usable_anchors() -> None:
    """10. Verify safe handling when no usable text anchors exist (all numeric, None, boolean)."""
    idx = _create_synthetic_document(
        num_pages=5,
        page_contents={
            1: ["1000", "2000.50", "TRUE"],
            2: ["3000", "4000.00", "FALSE"],
        },
    )
    adapter = ExtractBenchAdapter(idx)
    leaves = [
        ("num1", 1000, 1, "", "root"),
        ("num2", "2000.50", 1, "", "root"),
        ("flag", True, 1, "", "root"),
        ("empty", None, 1, "", "root"),
    ]
    res: OffsetCalibrationResult = adapter._calibrate_page_offset(leaves)
    assert res.offset == 0
    assert res.reason == "no_usable_anchors"
    assert adapter._detect_page_offset(leaves) == 0
