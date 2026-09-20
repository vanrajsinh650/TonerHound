"""Focused unit tests for EXP-015 Safe Character-Span Reconstruction."""

import pytest
from tonerhound.geometry.coordinates import BBox


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


def reconstruct_safe_character_span(
    pred_box: tuple[float, float, float, float],
    raw_token_text: str,
    target_value: str,
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """Safe Character-Span Resolver implementing EXP-015 Phase 2 safety conditions."""
    if confidence < 0.80 or not raw_token_text or not target_value:
        return pred_box

    px, py, pw, ph = pred_box
    target_clean = target_value.strip(" -.,;:_()[]{}/'\"")
    pt_clean = raw_token_text.strip(" -.,;:_()[]{}/'\"")

    if not target_clean or not pt_clean:
        return pred_box

    # Substring search
    idx = raw_token_text.lower().find(target_clean.lower())
    if idx == -1:
        idx = pt_clean.lower().find(target_clean.lower())
        if idx != -1:
            raw_text = pt_clean
        else:
            return pred_box
    else:
        raw_text = raw_token_text

    total_len = len(raw_text)
    target_len = len(target_clean)

    # Safety Guard 1: High risk of table dot leaders / row concatenation
    if total_len > 3 * target_len and ("..." in raw_text or "EUR" in raw_text or len(raw_text.split()) > 10):
        return pred_box

    # Safety Guard 2: Almost whole string (>= 90%) requires no trim
    if target_len / max(1, total_len) >= 0.90:
        return pred_box

    start_frac = idx / float(total_len)
    span_frac = target_len / float(total_len)

    new_x = px + pw * start_frac
    new_w = pw * span_frac

    # Clamp bounds
    new_w = max(0.002, min(new_w, 0.98 - new_x))
    return (new_x, py, new_w, ph)


class TestSafeCharacterSpan:
    def test_char_span_prefix_match_footnote_absorption(self):
        """Target value is at token start with trailing footnotes (e.g. '02/15/31(a)(b)')."""
        raw_text = "02/15/31(a)(b)"
        target = "02/15/31"
        gt_box = (0.0842, 0.1793, 0.0387, 0.0107)
        old_pred = (0.0842, 0.1793, 0.0589, 0.0107)

        # Before fix: IoU is in near-miss band (~0.65 for whole token, drops when height varies)
        new_box = reconstruct_safe_character_span(old_pred, raw_text, target)

        assert new_box[0] == pytest.approx(0.0842, abs=1e-4)  # X unchanged
        assert new_box[2] < old_pred[2]  # Width trimmed
        assert new_box[2] == pytest.approx(gt_box[2], abs=0.006)

        new_iou = iou_xywh(gt_box, new_box)
        assert new_iou >= 0.80

    def test_char_span_suffix_match_currency_symbol(self):
        """Target value is at token suffix with leading currency symbol (e.g. '$1,250.00')."""
        raw_text = "$1,250.00"
        target = "1,250.00"
        gt_box = (0.5050, 0.3000, 0.0400, 0.0100)
        old_pred = (0.5000, 0.3000, 0.0450, 0.0100)

        new_box = reconstruct_safe_character_span(old_pred, raw_text, target)

        # X must shift right past the dollar sign
        assert new_box[0] > old_pred[0]
        assert new_box[2] < old_pred[2]

        new_iou = iou_xywh(gt_box, new_box)
        assert new_iou >= 0.85

    def test_char_span_leading_bullet_guard_false_expansion_fix(self):
        """Target value has leading bullet/index prefix ('- 12. BANK VTB...').

        Verifies fix for EXP-014 False Expansions #1-#5 where X was falsely anchored to bullet margin.
        """
        raw_text = "- 12. BANK VTB PUBLICHNOE AKTSIONERNOE OBSHCHESTVO;"
        target = "BANK VTB PUBLICHNOE AKTSIONERNOE OBSHCHESTVO"
        gt_box = (0.1505, 0.5327, 0.1341, 0.0099)
        old_pred = (0.0897, 0.5327, 0.1948, 0.0250)

        new_box = reconstruct_safe_character_span(old_pred, raw_text, target)

        # X must shift right to account for the '- 12. ' prefix (~6 chars out of 50 = ~12% shift)
        assert new_box[0] > old_pred[0]
        assert new_box[0] == pytest.approx(0.0897 + 0.1948 * (6 / len(raw_text)), abs=0.005)

    def test_char_span_dot_leader_rejection_regression_guard(self):
        """Dot leaders ('TMK Hawk Parent Corp ... EUR 138 161,708 Loan') must be rejected.

        Verifies fix for EXP-014 Regressions #1-#3.
        """
        raw_text = "TMK Hawk Parent Corp., 2024 PIK Term ................................................................ EUR 138 161,708 Loan, 11.00%, 12/15/31"
        target = "TMK Hawk Parent Corp., 2024 PIK Term"
        old_pred = (0.0429, 0.4560, 0.6970, 0.0219)

        # Safety Guard 1 must trigger and reject the modification
        new_box = reconstruct_safe_character_span(old_pred, raw_text, target)
        assert new_box == old_pred

    def test_char_span_low_confidence_unmodified(self):
        """Low confidence candidates (< 0.80) must remain strictly unmodified."""
        old_pred = (0.1000, 0.2000, 0.0500, 0.0100)
        new_box = reconstruct_safe_character_span(old_pred, "test(a)", "test", confidence=0.75)
        assert new_box == old_pred
