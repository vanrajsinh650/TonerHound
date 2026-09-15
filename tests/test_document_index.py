"""Tests for TonerHound DocumentIndex."""

from pathlib import Path

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


def test_synthetic_document_indexing():
    # Construct a 1-page document with 2 lines:
    # Line 0: "Invoice Number: INV-2024-001"
    # Line 1: "Total Amount: $1,450.00"
    t1 = DocumentToken("Invoice", BBox(0.1, 0.1, 0.08, 0.02, page=1), page=1, char_index_in_page=0)
    t2 = DocumentToken("Number:", BBox(0.19, 0.1, 0.08, 0.02, page=1), page=1, char_index_in_page=8)
    t3 = DocumentToken("INV-2024-001", BBox(0.3, 0.1, 0.15, 0.02, page=1), page=1, char_index_in_page=16)

    t4 = DocumentToken("Total", BBox(0.1, 0.15, 0.06, 0.02, page=1), page=1, char_index_in_page=29)
    t5 = DocumentToken("Amount:", BBox(0.17, 0.15, 0.08, 0.02, page=1), page=1, char_index_in_page=35)
    t6 = DocumentToken("$1,450.00", BBox(0.3, 0.15, 0.1, 0.02, page=1), page=1, char_index_in_page=43)

    line1 = VisualLine([t1, t2, t3], page=1, line_index=0, bbox=BBox(0.1, 0.1, 0.35, 0.02, page=1))
    line2 = VisualLine([t4, t5, t6], page=1, line_index=1, bbox=BBox(0.1, 0.15, 0.3, 0.02, page=1))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=[t1, t2, t3, t4, t5, t6], lines=[line1, line2])
    index = DocumentIndex.from_pages([page])

    assert index.total_pages == 1
    matches = index.search_exact("INV-2024-001")
    assert len(matches) == 1
    box, matched_text = matches[0]
    assert "INV-2024-001" in matched_text
    assert abs(box.x - 0.3) < 1e-5
    assert abs(box.y - 0.1) < 1e-5


def test_real_pdf_indexing():
    pdf_path = Path("research/reference/groundmark/tests/data/two_pages.pdf")
    if not pdf_path.exists():
        return

    index = DocumentIndex.from_pdf(pdf_path)
    assert index.total_pages == 2
    page1 = index.get_page(1)
    assert page1 is not None
    assert len(page1.tokens) > 0
    assert len(page1.lines) > 0
    # Search for known text on page 1
    matches = index.search_exact("Page", page=1)
    assert len(matches) >= 1


def test_inverted_indexes_and_caching(tmp_path):
    # Construct a 20-page synthetic document to test multi-page scaling
    pages = []
    for p_num in range(1, 21):
        t_num = DocumentToken(
            f"${p_num * 100}.00",
            BBox(0.2, 0.2, 0.1, 0.02, page=p_num),
            page=p_num,
            char_index_in_page=0,
            line_index=0,
        )
        t_date = DocumentToken(
            "2024-03-15",
            BBox(0.4, 0.2, 0.15, 0.02, page=p_num),
            page=p_num,
            char_index_in_page=12,
            line_index=0,
        )
        t_word = DocumentToken(
            f"EntityName_{p_num}",
            BBox(0.6, 0.2, 0.2, 0.02, page=p_num),
            page=p_num,
            char_index_in_page=25,
            line_index=0,
        )
        line = VisualLine(
            tokens=[t_num, t_date, t_word],
            page=p_num,
            line_index=0,
            bbox=BBox(0.2, 0.2, 0.6, 0.02, page=p_num),
        )
        pages.append(
            DocumentPage(
                page_number=p_num,
                width=612,
                height=792,
                tokens=[t_num, t_date, t_word],
                lines=[line],
            )
        )

    index = DocumentIndex.from_pages(pages)
    assert index.total_pages == 20

    # 1. Numeric index O(1) lookup
    from tonerhound.matching.matcher import EvidenceMatcher

    matcher = EvidenceMatcher(index)

    # Search $1500 (which is on page 15) without any page hint on a 20-page document
    cands = matcher.find_normalized_numeric_candidates(1500.0, page_hint=None)
    assert len(cands) == 1
    assert cands[0].page == 15
    assert cands[0].matched_text == "$1500.00"

    # Search with integer key
    cands_int = matcher.find_normalized_numeric_candidates(700, page_hint=None)
    assert len(cands_int) == 1
    assert cands_int[0].page == 7

    # 2. Date index O(1) lookup without page hint
    date_cands = matcher.find_normalized_date_candidates("March 15, 2024", page_hint=None)
    assert len(date_cands) == 20
    assert {c.page for c in date_cands} == set(range(1, 21))

    # Specific hint page
    date_p3 = matcher.find_normalized_date_candidates("2024-03-15", page_hint=3)
    assert len(date_p3) == 1
    assert date_p3[0].page == 3

    # 3. Exact candidate lookup on 20-page document without page hint
    word_cands = matcher.find_exact_candidates("EntityName_12", page_hint=None)
    assert len(word_cands) == 1
    assert word_cands[0].page == 12

    # 4. Fuzzy candidate lookup with n-gram pruning on 20-page document without page hint
    fuzzy_cands = matcher.find_fuzzy_candidates("EntityName_18", page_hint=None)
    assert len(fuzzy_cands) >= 1
    assert fuzzy_cands[0].page == 18

