"""Focused unit tests for EXP-017 Same-Line Multi-Token Geometry Recovery."""

from __future__ import annotations

import pytest

from tonerhound.geometry.coordinates import BBox
from tonerhound.geometry.same_line_recovery import extend_same_line_tokens
from tonerhound.models.types import DocumentToken


def iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return min(1.0, inter / union)


def _make_token(text: str, x: float, y: float, w: float, h: float, page: int = 1, line_idx: int = 0) -> DocumentToken:
    return DocumentToken(
        text=text,
        bbox=BBox(x=x, y=y, width=w, height=h, page=page),
        page=page,
        char_index_in_page=0,
        line_index=line_idx,
    )


class TestSameLineMultiTokenRecovery:
    def test_multi_token_same_line_extension_success(self):
        """Target value spans multiple tokens on same line; candidate only covers word 0."""
        # Simulated line: '184.000 sh WSHFX American Funds Washington'
        tokens = [
            _make_token("184.000", 0.034, 0.468, 0.067, 0.010),
            _make_token("sh", 0.116, 0.468, 0.018, 0.010),
            _make_token("WSHFX", 0.147, 0.469, 0.049, 0.010),
            _make_token("American", 0.209, 0.468, 0.075, 0.010),
            _make_token("Funds", 0.290, 0.468, 0.055, 0.010),
            _make_token("Washington", 0.355, 0.469, 0.098, 0.010),
            _make_token("10,000", 0.470, 0.468, 0.050, 0.010),  # Next column
        ]
        target = "184.000 sh WSHFX American Funds Washington"
        old_pred = (0.034, 0.468, 0.067, 0.010)
        gt_box = (0.031, 0.466, 0.425, 0.014)

        new_box = extend_same_line_tokens(old_pred, "184.000", target, tokens)

        assert new_box[0] == pytest.approx(0.034, abs=1e-3)
        # Must stop at 'Washington' (x=0.355 + w=0.098 = 0.453), NOT consuming '10,000'
        assert (new_box[0] + new_box[2]) == pytest.approx(0.453, abs=1e-3)
        assert new_box[2] > old_pred[2] * 4.0

        old_iou = iou_xywh(gt_box, old_pred)
        new_iou = iou_xywh(gt_box, new_box)
        assert old_iou < 0.20
        assert new_iou >= 0.75

    def test_single_character_hazard_rejected(self):
        """Single character value (e.g. '0' or 'X') must NEVER be extended horizontally."""
        tokens = [
            _make_token("0", 0.903, 0.207, 0.007, 0.015),
            _make_token("125.00", 0.930, 0.207, 0.040, 0.015),  # Adjacent column
        ]
        old_pred = (0.903, 0.207, 0.007, 0.015)
        new_box = extend_same_line_tokens(old_pred, "0", "0", tokens)
        assert new_box == old_pred

    def test_single_word_hazard_rejected(self):
        """Single word target (e.g. 'California') must NEVER be extended into adjacent cells."""
        tokens = [
            _make_token("California", 0.100, 0.300, 0.060, 0.012),
            _make_token("94103", 0.180, 0.300, 0.040, 0.012),
        ]
        old_pred = (0.100, 0.300, 0.060, 0.012)
        new_box = extend_same_line_tokens(old_pred, "California", "California", tokens)
        assert new_box == old_pred

    def test_dot_leader_hazard_rejected(self):
        """Lines containing dot leaders '...' must be strictly rejected."""
        tokens = [
            _make_token("Bombardier", 0.050, 0.564, 0.080, 0.010),
            _make_token("...", 0.140, 0.564, 0.100, 0.010),
            _make_token("100,000", 0.250, 0.564, 0.050, 0.010),
        ]
        old_pred = (0.050, 0.564, 0.080, 0.010)
        target = "Bombardier, Inc. ... 100,000"
        new_box = extend_same_line_tokens(old_pred, "Bombardier", target, tokens)
        assert new_box == old_pred

    def test_pass_protection_already_passing_unmodified(self):
        """Citations already passing (is_passed=True) must remain untouched."""
        tokens = [
            _make_token("ACME", 0.100, 0.200, 0.040, 0.010),
            _make_token("Corp", 0.150, 0.200, 0.035, 0.010),
        ]
        old_pred = (0.100, 0.200, 0.040, 0.010)
        new_box = extend_same_line_tokens(old_pred, "ACME", "ACME Corp", tokens, is_passed=True)
        assert new_box == old_pred

    def test_low_confidence_unmodified(self):
        """Low confidence candidates (< 0.80) must remain strictly unmodified."""
        tokens = [
            _make_token("Alpha", 0.100, 0.200, 0.040, 0.010),
            _make_token("Beta", 0.150, 0.200, 0.035, 0.010),
        ]
        old_pred = (0.100, 0.200, 0.040, 0.010)
        new_box = extend_same_line_tokens(old_pred, "Alpha", "Alpha Beta", tokens, confidence=0.70)
        assert new_box == old_pred

    def test_unrelated_intervening_token_aborts_extension(self):
        """If an unrelated token appears between target tokens, expansion must abort."""
        tokens = [
            _make_token("Alpha", 0.100, 0.200, 0.040, 0.010),
            _make_token("INTERRUPT", 0.145, 0.200, 0.050, 0.010),
            _make_token("Beta", 0.200, 0.200, 0.040, 0.010),
        ]
        old_pred = (0.100, 0.200, 0.040, 0.010)
        new_box = extend_same_line_tokens(old_pred, "Alpha", "Alpha Beta", tokens)
        assert new_box == old_pred

    def test_column_boundary_gap_stops_extension(self):
        """Tokens separated by large whitespace gap (> 0.08) must not bridge columns."""
        tokens = [
            _make_token("Alpha", 0.100, 0.200, 0.040, 0.010),
            _make_token("Beta", 0.350, 0.200, 0.040, 0.010),  # Gap is 0.21 (> 0.08)
        ]
        old_pred = (0.100, 0.200, 0.040, 0.010)
        new_box = extend_same_line_tokens(old_pred, "Alpha", "Alpha Beta", tokens)
        assert new_box == old_pred

    def test_different_baseline_tokens_ignored(self):
        """Tokens from vertically adjacent lines (|dy| > 0.006) must not be consumed."""
        tokens = [
            _make_token("Alpha", 0.100, 0.200, 0.040, 0.010),
            _make_token("Beta", 0.150, 0.220, 0.040, 0.010),  # dy = 0.020 (different row)
        ]
        old_pred = (0.100, 0.200, 0.040, 0.010)
        new_box = extend_same_line_tokens(old_pred, "Alpha", "Alpha Beta", tokens)
        assert new_box == old_pred
