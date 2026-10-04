"""Unit tests for Hungarian table assigner."""

from __future__ import annotations

import pytest
from tonerhound.resolution.table_assigner import (
    HungarianTableAssigner,
    compute_iou_xywh,
    parse_array_field_path,
)


def test_parse_array_field_path() -> None:
    assert parse_array_field_path("holdings[0].shares") == ("holdings", 0, ".shares")
    assert parse_array_field_path("items[12].value") == ("items", 12, ".value")
    assert parse_array_field_path("simple_field") is None


def test_compute_iou_xywh() -> None:
    b1 = (0.1, 0.2, 0.3, 0.4)
    b2 = (0.1, 0.2, 0.3, 0.4)
    assert pytest.approx(compute_iou_xywh(b1, b2)) == 1.0

    b3 = (0.5, 0.5, 0.1, 0.1)
    assert compute_iou_xywh(b1, b3) == 0.0


def test_hungarian_table_assigner_disjoint_matches() -> None:
    assigner = HungarianTableAssigner(min_iou_threshold=0.30)
    records = [
        {"field_path": "table[0].val", "gold_bbox": [0.1, 0.1, 0.2, 0.05], "gold_page": 1},
        {"field_path": "table[1].val", "gold_bbox": [0.1, 0.2, 0.2, 0.05], "gold_page": 1},
        {"field_path": "table[2].val", "gold_bbox": [0.1, 0.3, 0.2, 0.05], "gold_page": 1},
    ]

    cands = [
        {"page": 1, "bbox": [0.101, 0.101, 0.2, 0.05], "id": "c0"},
        {"page": 1, "bbox": [0.101, 0.201, 0.2, 0.05], "id": "c1"},
        {"page": 1, "bbox": [0.101, 0.301, 0.2, 0.05], "id": "c2"},
    ]

    cand_pool = {
        "table[0].val": cands,
        "table[1].val": cands,
        "table[2].val": cands,
    }

    res = assigner.assign_array_group(records, cand_pool)
    assert len(res) == 3
    assert res["table[0].val"]["id"] == "c0"
    assert res["table[1].val"]["id"] == "c1"
    assert res["table[2].val"]["id"] == "c2"
