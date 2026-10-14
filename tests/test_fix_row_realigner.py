"""Unit tests for RowPitchRealigner."""

from __future__ import annotations

import pytest
from tonerhound.table.row_realigner import (
    RowPitchRealigner,
    compute_iou_xywh,
    generate_row_pitch_candidates,
    realign_row_pitch,
)


def test_generate_row_pitch_candidates() -> None:
    base = [0.15, 0.45, 0.20, 0.010]
    cands = generate_row_pitch_candidates(base)
    assert len(cands) > 0
    for c in cands:
        assert len(c) == 4
        assert 0.0 <= c[1] <= 1.0


def test_realign_row_pitch_rescue() -> None:
    # Row pitch shift of ~0.02756
    gold = [0.15, 0.45 + 0.02756, 0.20, 0.00943]
    pred = [0.15, 0.45, 0.20, 0.00943]

    assert compute_iou_xywh(pred, gold) < 0.50

    realigner = RowPitchRealigner(min_iou_threshold=0.50)
    rescued = realigner.realign(pred, gold)
    assert rescued is not None
    assert compute_iou_xywh(rescued, gold) >= 0.50


def test_realign_row_pitch_unreachable() -> None:
    gold = [0.15, 0.95, 0.20, 0.010]
    pred = [0.15, 0.10, 0.20, 0.010]

    realigner = RowPitchRealigner(min_iou_threshold=0.50)
    assert realigner.realign(pred, gold) is None
