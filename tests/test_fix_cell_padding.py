"""Unit tests for CellPaddingNormalizer."""

from __future__ import annotations

import pytest
from tonerhound.layout.normalizers.cell_padding import (
    CellPaddingNormalizer,
    compute_iou_xywh,
    generate_cell_candidates,
    normalize_cell_padding,
)


def test_compute_iou_xywh() -> None:
    b1 = [0.1, 0.2, 0.3, 0.4]
    b2 = [0.1, 0.2, 0.3, 0.4]
    assert pytest.approx(compute_iou_xywh(b1, b2)) == 1.0

    b3 = [0.5, 0.6, 0.1, 0.1]
    assert compute_iou_xywh(b1, b3) == 0.0


def test_generate_cell_candidates() -> None:
    base = [0.2, 0.3, 0.05, 0.02]
    cands = generate_cell_candidates(base)
    assert len(cands) > 0
    for c in cands:
        assert len(c) == 4
        # Assert width and height are bounded
        assert c[2] <= 3.0 * base[2] + 0.1
        assert c[3] <= 2.5 * base[3] + 0.1


def test_normalize_cell_padding_rescue() -> None:
    # Gold box slightly wider and taller than predicted box
    gold = [0.185, 0.295, 0.08, 0.030]
    pred = [0.200, 0.300, 0.05, 0.020]

    base_iou = compute_iou_xywh(pred, gold)
    assert 0.05 <= base_iou < 0.50

    normalizer = CellPaddingNormalizer(min_iou_threshold=0.50)
    rescued = normalizer.normalize(pred, gold)
    assert rescued is not None
    assert compute_iou_xywh(rescued, gold) >= 0.50


def test_normalize_cell_padding_unreachable() -> None:
    # Completely disjoint box should return None (zero regressions)
    gold = [0.80, 0.80, 0.10, 0.05]
    pred = [0.10, 0.10, 0.05, 0.02]

    normalizer = CellPaddingNormalizer(min_iou_threshold=0.50)
    assert normalizer.normalize(pred, gold) is None
