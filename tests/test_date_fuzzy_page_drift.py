"""Unit tests for EXP-028F Phase 1.1: Date and Fuzzy matching page drift.

Validates that when ENABLE_STRICT_PAGE_HINT = False:
- Evidence on page_hint is found (priority rank 0).
- Evidence on page_hint + 1 is found (when flag is False), but blocked when flag is True.
- Evidence on page_hint - 1 is found (when flag is False).
- Evidence on page_hint + 2 is NOT found (verifies global fallback is inactive in Phase 1.1).
- Page-hint candidate priority is maintained at index 0.
"""

import pytest

import tonerhound.matching.matcher as matcher_module
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import ENABLE_STRICT_PAGE_HINT, EvidenceMatcher
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


def _build_test_page(page_num: int, text: str) -> DocumentPage:
    """Helper to construct a single DocumentPage with one line containing the given text."""
    words = text.split()
    tokens = []
    x = 0.1
    char_idx = 0
    for i, w in enumerate(words):
        tok_bbox = BBox(x=x, y=0.2, width=0.08, height=0.03, page=page_num)
        tokens.append(
            DocumentToken(
                text=w,
                bbox=tok_bbox,
                page=page_num,
                char_index_in_page=char_idx,
                line_index=0,
            )
        )
        x += 0.09
        char_idx += len(w) + 1

    line = VisualLine(
        tokens=tokens,
        page=page_num,
        line_index=0,
        bbox=BBox(x=0.1, y=0.2, width=min(0.8, x - 0.1), height=0.03, page=page_num),
    )
    return DocumentPage(
        page_number=page_num,
        width=612,
        height=792,
        tokens=tokens,
        lines=[line],
    )


@pytest.fixture
def multi_page_index() -> DocumentIndex:
    """Build a 5-page document index with distinct dates and fuzzy phrases on each page."""
    pages = [
        _build_test_page(1, "Document title and overview page header"),
        _build_test_page(2, "Agreement entered on 2024-01-15 by Global Corporation Services Inc"),
        _build_test_page(3, "Filing completed on 2024-02-20 by Acme Industrial Supply Solutions"),
        _build_test_page(4, "Audit finalized on 2024-03-25 by Northern Pacific Exploration Ltd"),
        _build_test_page(5, "Closing executed on 2024-04-30 by Southern Atlantic Resources Group"),
    ]
    return DocumentIndex.from_pages(pages)


@pytest.fixture
def duplicate_evidence_index() -> DocumentIndex:
    """Build index where the exact same date and phrase exist on both page 3 and page 4."""
    pages = [
        _build_test_page(1, "Document header"),
        _build_test_page(2, "General overview section"),
        _build_test_page(3, "Primary contract signed on 2024-02-20 by Acme Industrial Supply Solutions"),
        _build_test_page(4, "Duplicate contract signed on 2024-02-20 by Acme Industrial Supply Solutions"),
        _build_test_page(5, "Concluding terms"),
    ]
    return DocumentIndex.from_pages(pages)


# ---------------------------------------------------------------------------
# Test 1: Target evidence exists on page_hint
# ---------------------------------------------------------------------------


def test_target_evidence_on_page_hint_date(multi_page_index: DocumentIndex):
    """Verify candidate is returned when target date exists on page_hint with flag True and False."""
    matcher = EvidenceMatcher(multi_page_index)

    # Flag = False
    cands_drift = matcher.find_normalized_date_candidates(
        "2024-02-20", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands_drift) >= 1
    assert cands_drift[0].page == 3
    assert cands_drift[0].match_type == "normalized_date"

    # Flag = True
    cands_strict = matcher.find_normalized_date_candidates(
        "2024-02-20", page_hint=3, enable_strict_page_hint=True
    )
    assert len(cands_strict) >= 1
    assert cands_strict[0].page == 3
    assert cands_strict[0].match_type == "normalized_date"


def test_target_evidence_on_page_hint_fuzzy(multi_page_index: DocumentIndex):
    """Verify candidate is returned when target phrase exists on page_hint with flag True and False."""
    matcher = EvidenceMatcher(multi_page_index)

    # Flag = False
    cands_drift = matcher.find_fuzzy_candidates(
        "Acme Industrial Supply Solutions", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands_drift) >= 1
    assert cands_drift[0].page == 3
    assert cands_drift[0].match_type == "fuzzy"

    # Flag = True
    cands_strict = matcher.find_fuzzy_candidates(
        "Acme Industrial Supply Solutions", page_hint=3, enable_strict_page_hint=True
    )
    assert len(cands_strict) >= 1
    assert cands_strict[0].page == 3
    assert cands_strict[0].match_type == "fuzzy"


# ---------------------------------------------------------------------------
# Test 2: Target evidence exists on page_hint + 1
# ---------------------------------------------------------------------------


def test_target_evidence_on_page_hint_plus_one_date(multi_page_index: DocumentIndex):
    """Verify date on page_hint + 1 is returned when flag=False, but NOT returned when flag=True."""
    matcher = EvidenceMatcher(multi_page_index)

    # Target date 2024-03-25 is on page 4. We query with page_hint=3.
    # When ENABLE_STRICT_PAGE_HINT = False: drift is allowed, candidate found on page 4
    cands_drift = matcher.find_normalized_date_candidates(
        "2024-03-25", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands_drift) >= 1
    assert cands_drift[0].page == 4
    assert cands_drift[0].match_type == "normalized_date"

    # When ENABLE_STRICT_PAGE_HINT = True: strict filter blocks page 4, returns []
    cands_strict = matcher.find_normalized_date_candidates(
        "2024-03-25", page_hint=3, enable_strict_page_hint=True
    )
    assert len(cands_strict) == 0


def test_target_evidence_on_page_hint_plus_one_fuzzy(multi_page_index: DocumentIndex):
    """Verify fuzzy phrase on page_hint + 1 is returned when flag=False, but NOT returned when flag=True."""
    matcher = EvidenceMatcher(multi_page_index)

    # Target phrase "Northern Pacific Exploration" is on page 4. Query with page_hint=3.
    # When ENABLE_STRICT_PAGE_HINT = False: drift is allowed, candidate found on page 4
    cands_drift = matcher.find_fuzzy_candidates(
        "Northern Pacific Exploration", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands_drift) >= 1
    assert cands_drift[0].page == 4
    assert cands_drift[0].match_type == "fuzzy"

    # When ENABLE_STRICT_PAGE_HINT = True: strict filter blocks page 4, returns []
    cands_strict = matcher.find_fuzzy_candidates(
        "Northern Pacific Exploration", page_hint=3, enable_strict_page_hint=True
    )
    assert len(cands_strict) == 0


# ---------------------------------------------------------------------------
# Test 3: Target evidence exists on page_hint - 1
# ---------------------------------------------------------------------------


def test_target_evidence_on_page_hint_minus_one_date(multi_page_index: DocumentIndex):
    """Verify date on page_hint - 1 is returned when flag=False."""
    matcher = EvidenceMatcher(multi_page_index)

    # Target date 2024-01-15 is on page 2. Query with page_hint=3.
    cands_drift = matcher.find_normalized_date_candidates(
        "2024-01-15", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands_drift) >= 1
    assert cands_drift[0].page == 2

    # Verify strict blocks it
    cands_strict = matcher.find_normalized_date_candidates(
        "2024-01-15", page_hint=3, enable_strict_page_hint=True
    )
    assert len(cands_strict) == 0


def test_target_evidence_on_page_hint_minus_one_fuzzy(multi_page_index: DocumentIndex):
    """Verify fuzzy phrase on page_hint - 1 is returned when flag=False."""
    matcher = EvidenceMatcher(multi_page_index)

    # Target phrase "Global Corporation Services" is on page 2. Query with page_hint=3.
    cands_drift = matcher.find_fuzzy_candidates(
        "Global Corporation Services", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands_drift) >= 1
    assert cands_drift[0].page == 2

    # Verify strict blocks it
    cands_strict = matcher.find_fuzzy_candidates(
        "Global Corporation Services", page_hint=3, enable_strict_page_hint=True
    )
    assert len(cands_strict) == 0


# ---------------------------------------------------------------------------
# Test 4: Target evidence exists on page_hint + 2 (confirms global fallback inactive)
# ---------------------------------------------------------------------------


def test_target_evidence_on_page_hint_plus_two_date(multi_page_index: DocumentIndex):
    """Verify date on page_hint + 2 is NOT returned when flag=False (no unrestricted global fallback)."""
    matcher = EvidenceMatcher(multi_page_index)

    # Target date 2024-04-30 is on page 5. Query with page_hint=3 (diff = +2).
    # Phase 1.1 strictly restricts drift to +/- 1 page.
    cands = matcher.find_normalized_date_candidates(
        "2024-04-30", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands) == 0


def test_target_evidence_on_page_hint_plus_two_fuzzy(multi_page_index: DocumentIndex):
    """Verify fuzzy phrase on page_hint + 2 is NOT returned when flag=False (no unrestricted global fallback)."""
    matcher = EvidenceMatcher(multi_page_index)

    # Target phrase "Southern Atlantic Resources" is on page 5. Query with page_hint=3 (diff = +2).
    cands = matcher.find_fuzzy_candidates(
        "Southern Atlantic Resources", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands) == 0


# ---------------------------------------------------------------------------
# Test 5: Page-hint candidate priority
# ---------------------------------------------------------------------------


def test_page_hint_candidate_priority_date(duplicate_evidence_index: DocumentIndex):
    """When evidence exists on both page_hint and page_hint + 1, candidate on page_hint must appear at index 0."""
    matcher = EvidenceMatcher(duplicate_evidence_index)

    # Evidence exists on page 3 and page 4. Query with page_hint=3.
    cands = matcher.find_normalized_date_candidates(
        "2024-02-20", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands) >= 1
    assert cands[0].page == 3


def test_page_hint_candidate_priority_fuzzy(duplicate_evidence_index: DocumentIndex):
    """When fuzzy evidence exists on both page_hint and page_hint + 1, candidate on page_hint must appear at index 0."""
    matcher = EvidenceMatcher(duplicate_evidence_index)

    # Evidence exists on page 3 and page 4. Query with page_hint=3.
    cands = matcher.find_fuzzy_candidates(
        "Acme Industrial Supply Solutions", page_hint=3, enable_strict_page_hint=False
    )
    assert len(cands) >= 1
    assert cands[0].page == 3


# ---------------------------------------------------------------------------
# Test 6: Module-level flag and instance default behavior
# ---------------------------------------------------------------------------


def test_module_flag_and_instance_default(multi_page_index: DocumentIndex, monkeypatch):
    """Verify module-level flag and EvidenceMatcher constructor parameter interaction."""
    # Default matcher instance uses ENABLE_STRICT_PAGE_HINT (False in Phase 1.1)
    matcher_default = EvidenceMatcher(multi_page_index)
    cands_default = matcher_default.find_normalized_date_candidates("2024-03-25", page_hint=3)
    assert len(cands_default) >= 1
    assert cands_default[0].page == 4

    # Setting module-level ENABLE_STRICT_PAGE_HINT = True restores strict behavior
    monkeypatch.setattr(matcher_module, "ENABLE_STRICT_PAGE_HINT", True)
    cands_strict = matcher_default.find_normalized_date_candidates("2024-03-25", page_hint=3)
    assert len(cands_strict) == 0

    # Instance-level enable_strict_page_hint=False overrides module-level True
    matcher_explicit_drift = EvidenceMatcher(multi_page_index, enable_strict_page_hint=False)
    cands_explicit = matcher_explicit_drift.find_normalized_date_candidates("2024-03-25", page_hint=3)
    assert len(cands_explicit) >= 1
    assert cands_explicit[0].page == 4
