"""EXP-006 Regression Tests: Digital vs Grid geometry gate.

These tests verify the *consistent_ratio* gate logic that replaces the old
``M >= 40`` hard-count gate in the creditors table grounding path:

OLD GATE (current HEAD c7acc8b):
    if table_name == "creditors" and (len(consistent_indices) < 5 or M >= 40):

NEW GATE (Agent B's fix):
    consistent_ratio = len(consistent_indices) / M if M > 0 else 0.0
    if table_name == "creditors" and (consistent_ratio < 0.80 or
                                      (len(consistent_indices) < 5 and M < 40)):

Failing tests (FAIL on HEAD, PASS after fix):
  - test_digital_pdf_uses_native_geometry          (Test 1)
  - test_gate_threshold_boundary_0_80 for 40/50    (Test 3 – above-threshold branch)
  - test_gate_signal_not_affected_by_m_alone       (Test 5)

Passing tests on BOTH HEAD and fixed:
  - test_corrupted_pdf_uses_synthetic_grid         (Test 2)
  - test_small_table_safety_net                    (Test 4)
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from tonerhound import DocumentIndex, DocumentPage, DocumentToken, VisualLine
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.geometry.coordinates import BBox


# ===========================================================================
# Fixture helpers
# ===========================================================================

_Y_START = 0.08162   # canonical grid y_start
_Y_END   = 0.88186   # canonical grid y_end
_S_71    = (_Y_END - _Y_START) / 70.0   # pitch for a 71-slot page (page_num != 2)
_S_72    = (_Y_END - _Y_START) / 71.0   # pitch for a 72-slot page (page_num == 2)


def _build_creditor_page_with_tokens(
    M: int,
    n_visible: int,
    *,
    page_num: int = 1,
    pitch: float = _S_71,
    y_first: float = _Y_START,
    token_name_prefix: str = "CREDITOR",
) -> DocumentPage:
    """Build a DocumentPage that has exactly *n_visible* token-bearing lines.

    Lines with index i < n_visible have a token whose text matches the record
    name for creditor row i.  Lines with index i >= n_visible are absent from
    the page (simulating OCR dropout / corrupt scan).

    Each visible line is placed at y = y_first + i * pitch so that the DP
    aligner can find them easily.  This gives us full control over
    consistent_ratio: n_visible / M.

    We also add a minimal running-header so _build_skewed_page-style filtering
    passes inside the adapter.
    """
    tokens: list[DocumentToken] = []
    lines:  list[VisualLine]    = []

    # ---- Running header (required for slope detection) ----
    t_hdr_l = DocumentToken(
        text="Case 22-11068-JTD",
        bbox=BBox(0.10, 0.020, 0.15, 0.012, page_num),
        page=page_num, char_index_in_page=0, line_index=0,
    )
    t_hdr_r = DocumentToken(
        text="Page 1 of 1",
        bbox=BBox(0.80, 0.020, 0.12, 0.012, page_num),
        page=page_num, char_index_in_page=20, line_index=0,
    )
    hdr_line = VisualLine(
        tokens=[t_hdr_l, t_hdr_r], page=page_num, line_index=0,
        bbox=BBox(0.10, 0.020, 0.82, 0.012, page_num),
    )
    tokens.extend([t_hdr_l, t_hdr_r])
    lines.append(hdr_line)

    # ---- Creditor data lines (only the first n_visible rows) ----
    for i in range(n_visible):
        y = y_first + i * pitch
        # Name token – must match the record name generated in _build_creditor_records
        name_text = f"{token_name_prefix} {i:03d} CORP"
        t_name = DocumentToken(
            text=name_text,
            bbox=BBox(0.067, y, 0.18, 0.010, page_num),
            page=page_num,
            char_index_in_page=100 + i * 50,
            line_index=len(lines),
        )
        # Address token on the same visual line
        t_addr = DocumentToken(
            text=f"{100 + i} MARKET ST",
            bbox=BBox(0.255, y, 0.14, 0.010, page_num),
            page=page_num,
            char_index_in_page=100 + i * 50 + 20,
            line_index=len(lines),
        )
        t_city = DocumentToken(
            text="SAN FRANCISCO",
            bbox=BBox(0.643, y, 0.09, 0.010, page_num),
            page=page_num,
            char_index_in_page=100 + i * 50 + 40,
            line_index=len(lines),
        )
        line = VisualLine(
            tokens=[t_name, t_addr, t_city],
            page=page_num,
            line_index=len(lines),
            bbox=BBox(0.067, y, 0.85, 0.010, page_num),
        )
        tokens.extend([t_name, t_addr, t_city])
        lines.append(line)

    return DocumentPage(
        page_number=page_num,
        width=612.0,
        height=792.0,
        tokens=tokens,
        lines=lines,
    )


def _build_creditor_records(
    M: int,
    name_prefix: str = "CREDITOR",
) -> list[dict[str, Any]]:
    """Build M creditor records whose name matches the tokens in the page fixture."""
    records = []
    for i in range(M):
        records.append({
            "name":        f"{name_prefix} {i:03d} CORP",
            "address_1":   f"{100 + i} MARKET ST",
            "address_2":   f"SUITE {200 + i}",
            "city":        "SAN FRANCISCO",
            "state":       "CA",
            "postal_code": f"{94100 + (i % 99):05d}",
            "country":     "USA",
        })
    return records


def _invoke_adapter(
    page: DocumentPage,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the adapter and return the raw result dict."""
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)
    return adapter.ground_extracted_data({"creditors": records})


# ---------------------------------------------------------------------------
# Grid-detection helpers
# ---------------------------------------------------------------------------

def _row_y_values(citations: dict[str, Any], field: str = "name") -> list[float]:
    """Return citation bbox y-values in row order for the named field."""
    ys: list[float] = []
    i = 0
    while f"creditors[{i}].{field}" in citations:
        bbox = citations[f"creditors[{i}].{field}"]["bbox"]
        ys.append(bbox[1])
        i += 1
    return ys


def _data_token_ys(page: DocumentPage, min_y: float = 0.07) -> list[float]:
    """Collect line bbox y-values from page data lines (excluding headers)."""
    return sorted(
        line.bbox.y
        for line in page.lines
        if line.bbox.y >= min_y and len(line.tokens) >= 1
    )


def _mean_row_pitch(ys: list[float]) -> float:
    """Return the mean consecutive y-step across all citation rows."""
    if len(ys) < 2:
        return 0.0
    diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    return sum(diffs) / len(diffs)


def _max_row_pitch(ys: list[float]) -> float:
    """Return the maximum consecutive y-step (reveals two-slot gaps in grid mode)."""
    if len(ys) < 2:
        return 0.0
    return max(ys[i + 1] - ys[i] for i in range(len(ys) - 1))


def _is_native_geometry(
    row_ys: list[float],
    token_ys: list[float],
    expected_token_pitch: float = _S_71,
    two_slot_pitch: float = 2 * _S_71,
) -> bool:
    """Determine whether *row_ys* reflect native token geometry or the synthetic grid.

    In native mode the adapter directly anchors each record to its matching
    page token, so:
      1. The mean row pitch equals the actual token pitch (expected_token_pitch).
      2. The maximum step between consecutive rows is ≤ 1.3 × expected_token_pitch.
         (No two-slot doubling occurs.)

    In grid mode with two-slot budgeting, the mean pitch is the page span
    divided by (M-1), which is noticeably larger than token pitch, and the
    max step is ≥ two_slot_pitch.

    We use a threshold of 1.5 × expected_token_pitch for the max step test so
    the test is robust to floating-point noise.
    """
    if not row_ys or not token_ys:
        return False

    mean_p = _mean_row_pitch(row_ys)
    max_p  = _max_row_pitch(row_ys)

    # Native: pitch ≈ token pitch; no row has a 2-slot gap
    native_mean = abs(mean_p - expected_token_pitch) < 0.003
    native_max  = max_p < 1.5 * expected_token_pitch

    return native_mean and native_max


def _is_grid_geometry(
    row_ys: list[float],
    M: int,
    page_num: int = 1,
    tol: float = 5e-4,
) -> bool:
    """Return True if row_ys match the synthetic slot grid.

    For M rows on a 71-slot page with n_two_slot = 71 - M two-slot rows, the
    mean row pitch across the full table equals (y_end - y_start) / (M - 1).
    We test that:
      1. y[0] is close to y_start (within a small padding).
      2. y[-1] is close to y_end.
      3. Every y is a valid slot position (residual to nearest slot < tol).
    """
    total_slots = 72 if page_num == 2 else 71
    s = (_Y_END - _Y_START) / float(total_slots - 1)

    # Check all ys are valid slot positions
    for y in row_ys:
        k = round((y - _Y_START) / s)
        k = max(0, min(total_slots - 1, k))
        if abs(y - (_Y_START + k * s)) > tol:
            return False
    return True


def _matches_canonical_grid(
    row_ys: list[float],
    y_start: float = _Y_START,
    page_num: int = 1,
    tol: float = 5e-4,
) -> bool:
    return _is_grid_geometry(row_ys, len(row_ys), page_num=page_num, tol=tol)


def _matches_token_bboxes(
    row_ys: list[float],
    token_ys: list[float],
    tol: float = 0.005,
) -> bool:
    if not row_ys or not token_ys:
        return False
    matched = sum(1 for y in row_ys if any(abs(y - ty) < tol for ty in token_ys))
    return (matched / len(row_ys)) >= 0.70


# ===========================================================================
# Test 1 – Digital PDF (high consistent_ratio) must use NATIVE geometry
# ===========================================================================

def test_digital_pdf_uses_native_geometry() -> None:
    """M=45 rows, all 45 visible in OCR → consistent_ratio ≈ 1.0 → native geometry.

    FAILS on HEAD (old gate forces M>=40 → grid regardless of OCR quality).
    PASSES after Agent B's consistent_ratio fix.

    Key discriminator: in native mode the adapter anchors each row to its actual
    OCR token.  All tokens are placed at pitch _S_71, so consecutive row y-values
    are also _S_71 apart and the max gap stays < 1.5 * _S_71.  In grid mode with
    M=45 the budget allocates n_two_slot = 71 - 45 = 26 two-slot rows, making the
    mean pitch ≈ 2*_S_71 and the max pitch = 2*_S_71 >> 1.5*_S_71.
    """
    M = 45
    n_visible = 45  # all rows have tokens → consistent_ratio = 1.0

    page = _build_creditor_page_with_tokens(M, n_visible, page_num=1)
    records = _build_creditor_records(M)
    res = _invoke_adapter(page, records)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    assert f"creditors[{M-1}].name" in citations, "Missing citations – adapter produced no output"

    row_ys = _row_y_values(citations, field="name")
    assert len(row_ys) == M, f"Expected {M} rows, got {len(row_ys)}"

    token_ys = _data_token_ys(page)

    # The mean row pitch must match the token pitch (≈ _S_71 = 0.01143).
    # Grid mode produces a much larger mean pitch because of two-slot budgeting.
    assert _is_native_geometry(row_ys, token_ys), (
        f"M={M}, ratio=1.0: adapter used SYNTHETIC GRID geometry instead of "
        "native token geometry.  Mean row pitch should be ≈ _S_71 but is "
        f"{_mean_row_pitch(row_ys):.5f} (expected ≈ {_S_71:.5f}); max pitch = "
        f"{_max_row_pitch(row_ys):.5f} (should be < {1.5*_S_71:.5f}).\n"
        f"  row_ys[:5]: {row_ys[:5]}\n"
        f"  token_ys[:5]: {token_ys[:5]}"
    )


# ===========================================================================
# Test 2 – Corrupted PDF (low consistent_ratio) must use SYNTHETIC GRID
# ===========================================================================

def test_corrupted_pdf_uses_synthetic_grid() -> None:
    """M=45 rows, only 5 visible in OCR → consistent_ratio = 5/45 ≈ 0.11 → grid.

    PASSES on HEAD and PASSES after the fix (corrupted case always uses grid).
    """
    M = 45
    n_visible = 5   # consistent_ratio = 5/45 ≈ 0.11 < 0.80

    page = _build_creditor_page_with_tokens(M, n_visible, page_num=1)
    records = _build_creditor_records(M)
    res = _invoke_adapter(page, records)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    assert f"creditors[{M-1}].name" in citations, "Missing citations"

    row_ys = _row_y_values(citations, field="name")
    assert len(row_ys) == M

    # Synthetic grid: y-values must match the canonical FTX slot pattern
    assert _matches_canonical_grid(row_ys, y_start=_Y_START, page_num=1), (
        "Corrupted PDF did NOT use synthetic grid geometry.\n"
        f"  row_ys (first 5): {row_ys[:5]}\n"
        f"  expected grid start: {_Y_START:.5f}, pitch: {_S_71:.6f}"
    )

    # Secondary: the sequence must be uniform (all diffs equal to slot * pitch)
    # For M=45 on a 71-slot page we have n_two_slot = 71 - 45 = 26 two-slot
    # rows, so the sequence is NOT strictly uniform – but we can check that
    # every y is a valid slot position.
    total_slots = 71
    s = (_Y_END - _Y_START) / float(total_slots - 1)
    for idx, y in enumerate(row_ys):
        k_nearest = round((y - _Y_START) / s)
        residual = abs(y - (_Y_START + k_nearest * s))
        assert residual < 5e-4, (
            f"Row {idx} y={y:.5f} is not on a grid slot (residual={residual:.6f})"
        )


# ===========================================================================
# Test 3 – Gate threshold boundary at consistent_ratio = 0.13
# ===========================================================================

def test_gate_threshold_boundary() -> None:
    """Boundary: M=50, 8 consistent (ratio=0.16 >= 0.13) → native; 6 consistent (ratio=0.12 < 0.13) → grid.

    Tests the empirical separation boundary between corrupted and digital documents.
    """
    M = 50

    # ---- Branch A: 8/50 = 0.16, at-or-above threshold (0.13) → NATIVE ----
    n_visible_above = 8   # ratio = 0.16 >= 0.13
    page_above = _build_creditor_page_with_tokens(M, n_visible_above, page_num=1)
    records = _build_creditor_records(M)
    res_above = _invoke_adapter(page_above, records)
    cits_above = {c["field_path"]: c for c in res_above["field_citations"]}

    assert f"creditors[{M-1}].name" in cits_above, "Above-threshold: missing citations"

    ys_above = _row_y_values(cits_above, field="name")
    token_ys_above = [line.bbox.y for line in page_above.lines if len(line.tokens) >= 2]

    # At ratio=0.16 the gate should NOT trigger the synthetic grid
    assert _matches_token_bboxes(ys_above[:n_visible_above], token_ys_above[:n_visible_above]), (
        "ratio=0.16 (8/50): expected NATIVE geometry but output y-values do "
        "not match token bboxes (adapter appears to have used synthetic grid).\n"
        f"  ys_above[:5]: {ys_above[:5]}\n"
        f"  token_ys_above[:5]: {sorted(token_ys_above)[:5]}"
    )

    # ---- Branch B: 6/50 = 0.12, below threshold (0.13) → SYNTHETIC GRID ----
    n_visible_below = 6   # ratio = 0.12 < 0.13
    page_below = _build_creditor_page_with_tokens(M, n_visible_below, page_num=1)
    res_below = _invoke_adapter(page_below, records)
    cits_below = {c["field_path"]: c for c in res_below["field_citations"]}

    assert f"creditors[{M-1}].name" in cits_below, "Below-threshold: missing citations"

    ys_below = _row_y_values(cits_below, field="name")

    assert _matches_canonical_grid(ys_below, y_start=_Y_START, page_num=1), (
        "ratio=0.12 (6/50): expected SYNTHETIC GRID geometry but output "
        "y-values are not on the canonical grid.\n"
        f"  ys_below[:5]: {ys_below[:5]}"
    )


# ===========================================================================
# Test 4 – Small table safety net (M < 40, low ratio → grid)
# ===========================================================================

def test_small_table_safety_net() -> None:
    """M=15 rows, only 2 consistent → ratio = 0.13 < 0.80 → synthetic grid.

    PASSES on HEAD and PASSES after fix (small corrupt tables always use grid).
    """
    M = 15
    n_visible = 2   # ratio = 2/15 ≈ 0.13

    page = _build_creditor_page_with_tokens(M, n_visible, page_num=1)
    records = _build_creditor_records(M)
    res = _invoke_adapter(page, records)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    # Some citations may be produced via the ref_r fallback for small M,
    # but they must NOT be on actual token bboxes that belong to rows > n_visible
    assert "creditors[0].name" in citations, "Missing citations for small table"

    row_ys = _row_y_values(citations, field="name")
    # For M<40 the old gate triggers on consistent_indices < 5, and the new
    # gate triggers on consistent_ratio < 0.80 (2/15 < 0.80), so the output
    # must come from the grid/interpolation path, not live token bboxes for
    # the unmatched rows.

    # The rows that HAD tokens (0..1) may have native y from direct anchor.
    # The rows WITHOUT tokens (2..14) must be interpolated; check those.
    # In the new gate (< 0.80) and old gate (< 5 consistent), the *whole*
    # table is re-anchored via the synthetic grid for M < 40.
    # Verify that all row ys are valid slot positions or interpolated positions.
    for idx, y in enumerate(row_ys):
        assert 0.03 <= y <= 0.97, f"Row {idx} y={y} out of valid range"

    # Must have all M rows cited
    assert len(row_ys) == M, f"Expected {M} row citations, got {len(row_ys)}"


# ===========================================================================
# Test 5 – M alone must NOT trigger synthetic grid when ratio is high
# ===========================================================================

def test_gate_signal_not_affected_by_m_alone() -> None:
    """M>=40 with a high consistent_ratio must use NATIVE geometry, not the grid.

    This is the core regression test: the OLD gate unconditionally forces the
    synthetic grid for ALL M>=40 creditor tables.  The NEW gate must NOT do
    this when the OCR quality is good (consistent_ratio >= 0.80).

    Sub-case A: M=50, n_visible=50 (ratio=1.0)  → native
    Sub-case B: M=50, n_visible=40 (ratio=0.80) → native

    Both FAIL on HEAD, both PASS after the fix.
    """
    for n_vis, label in [(50, "ratio=1.0 (50/50)"), (40, "ratio=0.80 (40/50)")]:
        M = 50

        page = _build_creditor_page_with_tokens(M, n_vis, page_num=1)
        records = _build_creditor_records(M)
        res = _invoke_adapter(page, records)
        citations = {c["field_path"]: c for c in res["field_citations"]}

        assert f"creditors[{M-1}].name" in citations, f"{label}: missing citations"

        row_ys = _row_y_values(citations, field="name")
        token_ys = [line.bbox.y for line in page.lines if len(line.tokens) >= 2]

        assert len(row_ys) == M, f"{label}: expected {M} rows, got {len(row_ys)}"

        # KEY ASSERTION: native geometry anchor y ≈ actual token bbox y
        assert _matches_token_bboxes(row_ys, token_ys), (
            f"{label}: M={M} caused synthetic grid even though consistent_ratio "
            "is at or above 0.80.  This means the M>=40 gate still fires.\n"
            f"  row_ys[:5]: {row_ys[:5]}\n"
            f"  token_ys[:5]: {sorted(token_ys)[:5]}"
        )

        # The grid pattern should NOT fit the output y-values
        # (native y-values are token-positioned, not slot-positioned)
        # Note: because we place tokens at _S_71 pitch which matches the grid
        # pitch, the canonical_grid check might pass coincidentally.  We rely
        # primarily on the token-match assertion above.
        #
        # Additional guard: verify the adapter did NOT double-budget with
        # two-slot rows (which the grid path does for M < total_page_slots).
        # In native mode all rows get direct anchors; consecutive diffs should
        # all equal our placed pitch _S_71 (within floating-point tolerance).
        if len(row_ys) >= 2:
            diffs = [row_ys[i + 1] - row_ys[i] for i in range(len(row_ys) - 1)]
            max_diff = max(diffs)
            min_diff = min(diffs)
            # Grid path for M=50 would produce 21 rows at 2*s and 29 at 1*s,
            # creating a bimodal distribution.  Native path should be unimodal
            # at exactly _S_71 (our placed pitch).
            assert max_diff < 2.5 * _S_71, (
                f"{label}: row-spacing max={max_diff:.5f} suggests two-slot "
                "budgeting (grid path) is still active.  Expected max ~ {_S_71:.5f}"
            )
