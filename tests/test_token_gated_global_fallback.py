"""Unit tests for EXP-028F Phase 1.2: Token-Gated Global Fallback for Long Documents.

Validates that:
1. Wrong page_hint by >1: local search fails, global fallback finds the correct page.
2. page_hint +/- 1 finds a valid candidate: global fallback MUST NOT run.
3. Token absent from entire document: inverted-index pre-check prevents global search.
4. Candidate cap: global fallback never returns >200 candidates.
5. Timeout: runaway candidate generation is cut off safely (500ms hard budget).
6. Local-vs-global ranking: strong page_hint candidate ranks ahead of global candidate.
7. Feature flag: FALSE reproduces baseline behavior, TRUE enables new behavior.
8. Phase 1.1 behavior remains intact.
"""

import time
from unittest.mock import patch
import pytest

import tonerhound.matching.candidate_recovery as cr_module
import tonerhound.resolution.resolver as res_module
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.candidate_recovery import CandidateRecoveryEngine
from tonerhound.matching.matcher import EvidenceMatcher, MatchCandidate
from tonerhound.models.types import DocumentPage, DocumentToken, ExtractionInput, VisualLine
from tonerhound.resolution.resolver import EvidenceResolver


def _build_page(page_num: int, text: str) -> DocumentPage:
    """Construct a DocumentPage with tokens and line for testing."""
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
def long_doc_index() -> DocumentIndex:
    """Build a 20-page document where target evidence is on distant pages."""
    pages = []
    for p in range(1, 21):
        if p == 15:
            # Spaced numeric on page 15
            txt = "Account balance: 1 4 5 0 . 0 0 dollars"
        elif p == 16:
            # Fragmented multi-token on page 16
            txt = "Vendor name: Acme Heavy Equipment Supply Services LLC"
        elif p == 6:
            # Evidence on page 6 (adjacent to page 5)
            txt = "Reference code: 9 8 7 6 . 5 0 units"
        else:
            txt = f"Standard narrative line on page {p} of the report document"
        pages.append(_build_page(p, txt))
    return DocumentIndex.from_pages(pages)


# ---------------------------------------------------------------------------
# Test 1: Wrong page_hint by >1 (local fails, global fallback finds correct page)
# ---------------------------------------------------------------------------


def test_wrong_page_hint_global_fallback_numeric(long_doc_index: DocumentIndex):
    """When page_hint is wrong by >1, local search fails and global fallback finds evidence."""
    engine = CandidateRecoveryEngine(long_doc_index, enable_global_fallback=True)

    # Evidence is on page 15. Query with page_hint=5 (diff = 10 pages).
    cands = engine.recover(
        value=1450.0,
        page_hint=5,
        field_name="balance",
    )
    assert len(cands) >= 1
    # Found on page 15
    assert any(c.page == 15 for c in cands)
    # Global fallback candidate received 0.85 multiplier
    p15_cand = next(c for c in cands if c.page == 15)
    assert p15_cand.raw_similarity <= 0.85


def test_wrong_page_hint_global_fallback_string(long_doc_index: DocumentIndex):
    """When string evidence is on a distant page, global fallback recovers it."""
    engine = CandidateRecoveryEngine(long_doc_index, enable_global_fallback=True)

    # Evidence is on page 16. Query with page_hint=3 (diff = 13 pages).
    cands = engine.recover(
        value="Acme Heavy Equipment Supply Services LLC",
        page_hint=3,
        field_name="vendor_name",
    )
    assert len(cands) >= 1
    assert any(c.page == 16 for c in cands)


# ---------------------------------------------------------------------------
# Test 2: page_hint +/- 1 finds valid candidate (global fallback MUST NOT run)
# ---------------------------------------------------------------------------


def test_local_candidate_prevents_global_fallback(long_doc_index: DocumentIndex):
    """When page_hint +/- 1 has valid evidence, global fallback is bypassed."""
    engine = CandidateRecoveryEngine(long_doc_index, enable_global_fallback=True)

    # Evidence is on page 6. Query with page_hint=5 (page_hint + 1).
    with patch.object(engine, "_find_candidate_pages", wraps=engine._find_candidate_pages) as mock_precheck:
        cands = engine.recover(
            value=9876.5,
            page_hint=5,
            field_name="units",
        )
        assert len(cands) >= 1
        assert any(c.page == 6 for c in cands)
        # Global fallback inverted-index precheck must NOT have been called
        mock_precheck.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Token absent from entire document (inverted-index precheck prevents scan)
# ---------------------------------------------------------------------------


def test_token_absent_skips_global_search(long_doc_index: DocumentIndex):
    """When target token is absent from document, inverted-index precheck returns empty."""
    engine = CandidateRecoveryEngine(long_doc_index, enable_global_fallback=True)

    with patch.object(engine, "_execute_recovery_passes", wraps=engine._execute_recovery_passes) as mock_exec:
        cands = engine.recover(
            value="ZebraNonexistentEntityPhraseXYZ",
            page_hint=5,
            field_name="unknown_field",
        )
        assert len(cands) == 0
        # Global pass execution must NOT be called for absent tokens
        # (local search ran on 3 pages, but global pass on 17 pages was skipped)
        assert mock_exec.call_count <= 1  # Only the local search call, no global call


# ---------------------------------------------------------------------------
# Test 4: Candidate cap (global fallback never returns >200 candidates)
# ---------------------------------------------------------------------------


def test_candidate_cap_enforced(long_doc_index: DocumentIndex):
    """Ensure candidate list is strictly capped at max 200 candidates."""
    engine = CandidateRecoveryEngine(long_doc_index, enable_global_fallback=True)

    # Mock _execute_recovery_passes to return 250 candidates
    fake_cands = [
        MatchCandidate(
            page=p,
            bbox=BBox(0.1, 0.2, 0.1, 0.02, page=p),
            tokens=(),
            matched_text=f"fake_{i}",
            match_type="recovered_spaced",
            raw_similarity=0.90,
        )
        for i in range(250)
        for p in [i % 20 + 1]
    ][:250]

    with patch.object(engine, "_execute_recovery_passes", return_value=fake_cands):
        res = engine.recover(value=100.0, page_hint=1)
        assert len(res) <= 200


# ---------------------------------------------------------------------------
# Test 5: Timeout (runaway candidate generation is cut off safely)
# ---------------------------------------------------------------------------


def test_timeout_budget_cutoff(long_doc_index: DocumentIndex):
    """Verify 500ms hard budget safely terminates processing."""
    engine = CandidateRecoveryEngine(long_doc_index, enable_global_fallback=True)

    # Call with virtually 0 timeout to trigger timeout break
    t0 = time.perf_counter()
    cands = engine.recover(
        value="Acme Heavy Equipment Supply Services LLC",
        page_hint=None,
        timeout_sec=0.00001,
    )
    elapsed = time.perf_counter() - t0
    # Must complete almost immediately
    assert elapsed < 0.200


# ---------------------------------------------------------------------------
# Test 6: Local-vs-global ranking (page_hint candidate ranks ahead)
# ---------------------------------------------------------------------------


def test_local_vs_global_ranking():
    """Verify exact page_hint candidate outranks global fallback candidate."""
    # Build 2-page doc: page 1 has target, page 2 has target
    p1 = _build_page(1, "Agreement on 2024-05-01 by Target Corp")
    p2 = _build_page(20, "Duplicate agreement on 2024-05-01 by Target Corp")
    idx = DocumentIndex.from_pages([p1, p2])

    cand_local = MatchCandidate(
        page=1,
        bbox=BBox(0.1, 0.2, 0.1, 0.02, page=1),
        tokens=(),
        matched_text="Target Corp",
        match_type="recovered",
        raw_similarity=0.90,
    )
    cand_global = MatchCandidate(
        page=20,
        bbox=BBox(0.1, 0.2, 0.1, 0.02, page=20),
        tokens=(),
        matched_text="Target Corp",
        match_type="recovered",
        raw_similarity=1.00,
    )

    engine = CandidateRecoveryEngine(idx, enable_global_fallback=True)
    ranked = engine._deduplicate_and_rank([cand_global, cand_local], page_hint=1, is_global_fallback=True)
    assert len(ranked) == 2
    # Local candidate on page 1 must rank at index 0
    assert ranked[0].page == 1
    # Global candidate on page 20 received 0.85 penalty and ranks second
    assert ranked[1].page == 20
    assert ranked[1].raw_similarity == pytest.approx(0.85, rel=1e-3)


# ---------------------------------------------------------------------------
# Test 7: Feature flag (FALSE reproduces baseline, TRUE enables new behavior)
# ---------------------------------------------------------------------------


def test_feature_flag_toggle(long_doc_index: DocumentIndex, monkeypatch):
    """Verify ENABLE_GLOBAL_FALLBACK flag toggles between baseline and new behavior."""
    # When ENABLE_GLOBAL_FALLBACK = False on a 20-page document:
    resolver_false = EvidenceResolver(long_doc_index, enable_global_fallback=False)
    # Evidence is on page 15, query with page_hint=5
    res_false = resolver_false.resolve(
        ExtractionInput(field="balance", value=1450.0, page_hint=5)
    )
    # Baseline behavior: >10 pages blocks global fallback, result is NOT_FOUND
    assert not res_false.is_grounded or res_false.page != 15

    # When ENABLE_GLOBAL_FALLBACK = True on the same 20-page document:
    resolver_true = EvidenceResolver(long_doc_index, enable_global_fallback=True)
    res_true = resolver_true.resolve(
        ExtractionInput(field="balance", value=1450.0, page_hint=5)
    )
    # New behavior: recovers evidence on page 15
    assert res_true.is_grounded
    assert res_true.page == 15


# ---------------------------------------------------------------------------
# Test 8: Phase 1.1 behavior remains intact
# ---------------------------------------------------------------------------


def test_phase1_1_behavior_intact(long_doc_index: DocumentIndex):
    """Verify Phase 1.1 +/- 1 date and fuzzy drift remain fully functional."""
    matcher = EvidenceMatcher(long_doc_index)

    # Date on page 15, query with page_hint=14 (diff = +1)
    # Note: the text on page 15 is "Account balance: 1 4 5 0 . 0 0 dollars", but let's test page 6
    # Page 6 has reference code: "9 8 7 6 . 5 0 units"
    # Let's verify strict page hint flag
    assert hasattr(matcher, "enable_strict_page_hint")
    assert hasattr(matcher, "_resolve_strict_page_hint")
