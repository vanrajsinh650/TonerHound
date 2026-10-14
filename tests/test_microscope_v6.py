"""Unit tests for Failure Microscope V6."""

import json
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from research.observer.microscope_v6 import (
    DocumentAuditReport,
    FailureCluster,
    FieldDetail,
    MicroscopeV6Auditor,
    compute_field_pattern,
    iou_xywh,
    validate_gate_v8,
)


def test_compute_field_pattern():
    assert compute_field_pattern("holdings[0].shares") == "holdings[].shares"
    assert compute_field_pattern("holdings[12].security.ticker") == "holdings[].security.ticker"
    assert compute_field_pattern("simple_field") == "simple_field"
    assert compute_field_pattern("table[0].rows[3].col") == "table[].rows[].col"


def test_iou_xywh():
    box1 = [0.1, 0.1, 0.2, 0.2]
    box2 = [0.1, 0.1, 0.2, 0.2]
    assert iou_xywh(box1, box2) == pytest.approx(1.0)

    box3 = [0.5, 0.5, 0.2, 0.2]
    assert iou_xywh(box1, box3) == 0.0

    # Overlap calculation
    box4 = [0.1, 0.1, 0.1, 0.2]  # half width
    assert iou_xywh(box1, box4) == pytest.approx(0.5, rel=1e-3)


def test_document_audit_synthetic(tmp_path: Path):
    data_dir = tmp_path / "data"
    preds_dir = tmp_path / "preds"
    data_dir.mkdir()
    preds_dir.mkdir()

    doc_id = "test_doc_01"
    test_json = {
        "_field_rules": {
            "entity.name": {
                "evidence": [{"page": 1, "bbox": [0.1, 0.1, 0.2, 0.05], "value": "Acme Corp"}]
            },
            "holdings[0].shares": {
                "evidence": [{"page": 1, "bbox": [0.1, 0.2, 0.1, 0.02], "value": "100"}]
            },
            "holdings[1].shares": {
                "evidence": [{"page": 1, "bbox": [0.1, 0.3, 0.1, 0.02], "value": "200"}]
            },
            "missing.field": {
                "evidence": [{"page": 2, "bbox": [0.4, 0.4, 0.1, 0.02], "value": "Lost"}]
            },
            "drift.field": {
                "evidence": [{"page": 1, "bbox": [0.5, 0.5, 0.1, 0.02], "value": "Drift"}]
            },
        }
    }
    with open(data_dir / f"{doc_id}.test.json", "w") as f:
        json.dump(test_json, f)

    pred_json = {
        "output": {
            "field_citations": [
                # PASS: exact match
                {"field_path": "entity.name", "page": 1, "bbox": [0.1, 0.1, 0.2, 0.05], "value": "Acme Corp"},
                # INSUFFICIENT_IOU: 0.05 <= iou < 0.50
                {"field_path": "holdings[0].shares", "page": 1, "bbox": [0.1, 0.2, 0.03, 0.02], "value": "100"},
                # WRONG_ROW: iou < 0.05
                {"field_path": "holdings[1].shares", "page": 1, "bbox": [0.1, 0.35, 0.1, 0.02], "value": "200"},
                # missing.field is omitted -> ZERO_CANDIDATE
                # PAGE_DRIFT: different page
                {"field_path": "drift.field", "page": 3, "bbox": [0.5, 0.5, 0.1, 0.02], "value": "Drift"},
            ]
        }
    }
    with open(preds_dir / f"{doc_id}.result.json", "w") as f:
        json.dump(pred_json, f)

    auditor = MicroscopeV6Auditor(data_dir=data_dir, preds_dir=preds_dir)
    report = auditor.audit_document(doc_id)

    assert report.document_id == doc_id
    assert report.status == "evaluated"
    assert report.gradeable == 5
    assert report.passing == 1
    assert report.failing == 4
    assert report.word_f1 == 20.0
    assert report.gradeable == report.passing + report.failing

    # Verify field details
    field_statuses = {f["field_path"]: f["status"] for f in report.fields}
    assert field_statuses["entity.name"] == "PASS"
    assert field_statuses["holdings[0].shares"] == "INSUFFICIENT_IOU"
    assert field_statuses["holdings[1].shares"] == "WRONG_ROW"
    assert field_statuses["missing.field"] == "ZERO_CANDIDATE"
    assert field_statuses["drift.field"] == "PAGE_DRIFT"

    # Verify failure clusters
    patterns = {c["field_pattern"]: c for c in report.failure_clusters}
    assert "holdings[].shares" in patterns
    assert patterns["holdings[].shares"]["count"] == 2


def test_validate_gate_v8():
    r1 = DocumentAuditReport(
        document_id="doc1",
        status="evaluated",
        gradeable=10,
        passing=7,
        failing=3,
        word_f1=70.0,
        page_f1=80.0,
        family="SEC 13F",
        fields=[{"field_path": f"f_{i}"} for i in range(10)],
        failure_clusters=[],
    )
    r2 = DocumentAuditReport(
        document_id="doc2",
        status="evaluated",
        gradeable=5,
        passing=5,
        failing=0,
        word_f1=100.0,
        page_f1=100.0,
        family="IRS 1040",
        fields=[{"field_path": f"f_{i}"} for i in range(5)],
        failure_clusters=[],
    )
    v8_res = validate_gate_v8([r1, r2])
    assert v8_res["status"] == "PASS"
    assert v8_res["total_gradeable"] == 15
    assert v8_res["total_passing"] == 12
    assert v8_res["total_failing"] == 3
    assert len(v8_res["accounting_discrepancies"]) == 0

    # Test discrepancy detection
    r_bad = DocumentAuditReport(
        document_id="doc_bad",
        status="evaluated",
        gradeable=10,
        passing=5,
        failing=2,  # 5 + 2 != 10
        word_f1=50.0,
        page_f1=50.0,
        family="Test",
        fields=[{"field_path": f"f_{i}"} for i in range(10)],
        failure_clusters=[],
    )
    v8_bad = validate_gate_v8([r_bad])
    assert v8_bad["status"] == "FAIL"
    assert len(v8_bad["accounting_discrepancies"]) > 0
