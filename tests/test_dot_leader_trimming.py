"""Unit tests for EXP-018 Safe Dot-Leader Trimming."""

import pytest
from tonerhound.geometry.coordinates import BBox
from tonerhound.geometry.dot_leader_trimming import (
    is_dot_leader_token,
    trim_dot_leaders,
    word_matches,
)
from tonerhound.models.types import DocumentToken


def test_is_dot_leader_token():
    assert is_dot_leader_token("...")
    assert is_dot_leader_token(".................................................")
    assert is_dot_leader_token("..")
    assert is_dot_leader_token("…")
    assert is_dot_leader_token(". . . . . .")
    assert not is_dot_leader_token(".")
    assert not is_dot_leader_token("Bombardier")
    assert not is_dot_leader_token("02/15/28")
    assert not is_dot_leader_token("")


def test_word_matches():
    assert word_matches("6.00%,", "6.00%")
    assert word_matches("02/15/28(a)(b)", "02/15/28")
    assert word_matches("Bombardier,", "Bombardier")
    assert not word_matches("Apple", "Microsoft")


def test_token_level_dot_leader_trimming():
    t1 = DocumentToken(
        text="6.00%,",
        bbox=BBox(x=0.05, y=0.56, width=0.03, height=0.01, page=1),
        page=1,
        char_index_in_page=0,
        line_index=0,
    )
    t2 = DocumentToken(
        text="02/15/28(a)(b)",
        bbox=BBox(x=0.08, y=0.56, width=0.06, height=0.01, page=1),
        page=1,
        char_index_in_page=7,
        line_index=0,
    )
    t3 = DocumentToken(
        text=".................................................",
        bbox=BBox(x=0.14, y=0.56, width=0.14, height=0.01, page=1),
        page=1,
        char_index_in_page=22,
        line_index=0,
    )

    # Overextended candidate box covers all 3 tokens
    cand_box = BBox(x=0.05, y=0.56, width=0.23, height=0.016, page=1)
    line_tokens = [t1, t2, t3]

    trimmed = trim_dot_leaders(
        cand_bbox=cand_box,
        reference_text="6.00%, 02/15/28(a)(b) .................................................",
        target_value="Bombardier, Inc., 6.00%, 02/15/28",
        line_tokens=line_tokens,
        confidence=0.90,
    )

    assert isinstance(trimmed, BBox)
    assert trimmed.x == pytest.approx(0.05, abs=1e-4)
    # Right edge should stop at t2 (0.08 + 0.06 = 0.14), width ~ 0.09
    assert trimmed.width == pytest.approx(0.09, abs=1e-4)
    assert trimmed.height == cand_box.height
    assert trimmed.page == 1


def test_intra_token_dot_trimming():
    cand_box = (0.10, 0.20, 0.20, 0.015)
    ref_text = "Senior Secured Notes 5.50% ......"
    target_val = "Senior Secured Notes 5.50%"

    trimmed = trim_dot_leaders(
        cand_bbox=cand_box,
        reference_text=ref_text,
        target_value=target_val,
        line_tokens=None,
        confidence=0.90,
    )

    assert isinstance(trimmed, tuple)
    assert trimmed[0] == 0.10
    # Width should be trimmed proportionally
    assert trimmed[2] < cand_box[2]
    assert trimmed[2] > 0.10


def test_pass_protection():
    cand_box = BBox(x=0.05, y=0.56, width=0.23, height=0.016, page=1)
    res = trim_dot_leaders(
        cand_bbox=cand_box,
        reference_text="6.00%, 02/15/28 ...",
        target_value="6.00%, 02/15/28",
        is_passed=True,
    )
    assert res == cand_box


def test_no_dots_unchanged():
    cand_box = BBox(x=0.05, y=0.56, width=0.10, height=0.016, page=1)
    res = trim_dot_leaders(
        cand_bbox=cand_box,
        reference_text="6.00%, 02/15/28",
        target_value="6.00%, 02/15/28",
        line_tokens=[],
        confidence=0.95,
    )
    assert res == cand_box


def test_low_confidence_unchanged():
    cand_box = BBox(x=0.05, y=0.56, width=0.23, height=0.016, page=1)
    res = trim_dot_leaders(
        cand_bbox=cand_box,
        reference_text="6.00%, 02/15/28 ...",
        target_value="6.00%, 02/15/28",
        confidence=0.70,
    )
    assert res == cand_box
