"""Tests for EXP-033 Candidate Generation Expansion and Reconciliation Fixes."""

from __future__ import annotations

import pytest

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, ExtractionInput, VisualLine
from tonerhound.resolution.resolver import EvidenceResolver


def _create_test_index() -> DocumentIndex:
    """Create a synthetic DocumentIndex with multi-line, token-punct, and boolean items."""
    # Line 0: "Depth: 5482 feet"
    # Line 1: "Company: Post Holdings, Inc. Total: 99.4%"
    # Line 2: "Email line 1: alpha@domain.com"
    # Line 3: "beta@domain.com"
    # Line 4: "gamma@domain.com"
    # Line 5: "delta@domain.com"
    # Line 6: "Reason: [ ] Pressure"
    tokens_data = [
        # Line 0
        [("Depth:", (10, 10, 40, 10)), ("5482", (60, 10, 30, 10)), ("feet", (100, 10, 30, 10))],
        # Line 1
        [("Company:", (10, 30, 50, 10)), ("Post", (70, 30, 30, 10)), ("Holdings,", (110, 30, 50, 10)), ("Inc.", (170, 30, 30, 10)), ("Total:", (210, 30, 40, 10)), ("(99.4%)", (260, 30, 50, 10))],
        # Line 2
        [("Email:", (10, 50, 40, 10)), ("alpha@domain.com", (60, 50, 100, 10))],
        # Line 3
        [("beta@domain.com", (60, 65, 100, 10))],
        # Line 4
        [("gamma@domain.com", (60, 80, 100, 10))],
        # Line 5
        [("delta@domain.com", (60, 95, 100, 10))],
        # Line 6
        [("Reason:", (10, 115, 50, 10)), ("[ ]", (70, 115, 15, 10)), ("Pressure", (95, 115, 60, 10))],
    ]

    lines: list[VisualLine] = []
    page_tokens: list[DocumentToken] = []
    char_idx = 0
    for l_idx, line_toks_info in enumerate(tokens_data):
        line_toks: list[DocumentToken] = []
        for txt, (x, y, w, h) in line_toks_info:
            tok = DocumentToken(
                text=txt,
                bbox=BBox(x=x / 1000.0, y=y / 1000.0, width=w / 1000.0, height=h / 1000.0, page=1),
                page=1,
                char_index_in_page=char_idx,
                line_index=l_idx,
            )
            char_idx += len(txt) + 1
            line_toks.append(tok)
            page_tokens.append(tok)

        min_x = min(t.bbox.x for t in line_toks)
        min_y = min(t.bbox.y for t in line_toks)
        max_x = max(t.bbox.x + t.bbox.width for t in line_toks)
        max_y = max(t.bbox.y + t.bbox.height for t in line_toks)
        lines.append(
            VisualLine(
                tokens=line_toks,
                page=1,
                line_index=l_idx,
                bbox=BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y, page=1),
            )
        )

    page = DocumentPage(page_number=1, width=1000.0, height=1000.0, tokens=page_tokens, lines=lines)
    return DocumentIndex([page])


def test_boolean_expansion_flag():
    idx = _create_test_index()
    inp = ExtractionInput(field="reason_pressure", value="False", page_hint=1)

    # Baseline (flag OFF): "reason_pressure" does not contain hardcoded keywords -> 0 candidates
    res_base = EvidenceResolver(idx, enable_boolean_expansion=False)
    cands_base = res_base.collect_candidates(inp)
    assert len(cands_base) == 0

    # Expansion ON: detects value="False" as boolean -> locates [ ] near Pressure
    res_exp = EvidenceResolver(idx, enable_boolean_expansion=True)
    cands_exp = res_exp.collect_candidates(inp)
    assert len(cands_exp) >= 1
    assert cands_exp[0].match_type == "boolean"
    assert cands_exp[0].matched_text == "[ ]"


def test_token_strip_recovery():
    idx = _create_test_index()

    # Query with trailing punctuation on a string: "Post Holdings, Inc.,"
    inp_punct = ExtractionInput(field="company_name", value="Post Holdings, Inc.,", page_hint=1)
    res_exp = EvidenceResolver(idx, enable_token_strip_recovery=True)
    cands_exp = res_exp.collect_candidates(inp_punct)
    assert len(cands_exp) >= 1
    assert "Post Holdings" in cands_exp[0].matched_text


def test_multiline_recovery():
    idx = _create_test_index()
    # Query with 4-line newline-separated sequence: lines 2, 3, 4, 5
    query = "alpha@domain.com\nbeta@domain.com\ngamma@domain.com\ndelta@domain.com"
    inp = ExtractionInput(field="contact_emails", value=query, page_hint=1)

    # Without multiline recovery or with standard search, query with newlines is tested
    res_exp = EvidenceResolver(idx, enable_multi_line_recovery=True)
    cands_exp = res_exp.collect_candidates(inp)
    assert len(cands_exp) >= 1
    assert "alpha@domain.com" in cands_exp[0].matched_text


def test_master_flag_expansion():
    idx = _create_test_index()
    res = EvidenceResolver(idx, enable_exp033_candidate_expansion=True)

    # All should be active under master flag
    cands_bool = res.collect_candidates(ExtractionInput(field="reason_pressure", value="False", page_hint=1))
    assert len(cands_bool) >= 1

    cands_strip = res.collect_candidates(ExtractionInput(field="company_name", value="Post Holdings, Inc.,", page_hint=1))
    assert len(cands_strip) >= 1
