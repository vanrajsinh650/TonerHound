"""Authoritative Failure Microscope V4 Runner.

Integrates calibrated empirical recovery estimations with causal classification:
1. Classifies every failing field deterministically (V3 taxonomy).
2. Generates macro-weighted theoretical ceilings (V3 ranking).
3. Applies empirical recovery rates from research/observer/recovery_rate_table.json.
4. Generates realistic expected benchmark gains (Section 3 of Directive).
5. Outputs all artifacts conforming to Microscope V4 specifications.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Prevent thread oversubscription in math libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
    path_leaf,
)
from extract_bench.test_cases.loader import load_test_case

from research.observer.field_classifier import (
    FailureMicroscopeClassifier,
    FieldClassificationResult,
)
from research.observer.report_generator import FailureMicroscopeReporter
from research.observer.realistic_estimator import RealisticEstimator
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput
from tonerhound.resolution.resolver import EvidenceResolver


def _flatten_expected_leaves(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dict/list into leaf paths and values."""
    leaves: list[tuple[str, Any]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            p = f"{prefix}.{k}" if prefix else k
            leaves.extend(_flatten_expected_leaves(v, p))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            p = f"{prefix}[{i}]"
            leaves.extend(_flatten_expected_leaves(item, p))
    else:
        leaves.append((prefix, data))
    return leaves


def _audit_document(task: dict[str, Any]) -> dict[str, Any]:
    """Audit a single document and return list of field classification dicts."""
    test_id = task["test_id"]
    pdf_path = Path(task["pdf_path"])
    pred_path = Path(task["pred_path"])
    run_id = task.get("run_id", "run")

    if not pred_path.exists():
        return {"test_id": test_id, "error": f"Prediction file not found: {pred_path}", "fields": []}

    try:
        with open(pred_path, encoding="utf-8") as f:
            pred_data = json.load(f)

        tc = load_test_case(pdf_path)
        raw_cits = pred_data.get("output", {}).get("field_citations", [])
        pred_cits_by_path = {
            c["field_path"]: c
            for c in raw_cits
            if c.get("field_path") and c.get("page") is not None
        }

        # Build ground truth rule indexes
        field_rules = tc.get_extract_field_rules()
        alt_values, ev_boxes, ev_pages, normalizers = build_rule_indexes(field_rules)

        # Build document index for causal candidate inspection
        is_corrupted = "corrupted" in test_id
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=is_corrupted, backend="hybrid")
        resolver = EvidenceResolver(doc_index)
        classifier = FailureMicroscopeClassifier(doc_index=doc_index, resolver=resolver)

        results: list[dict[str, Any]] = []

        for rule in field_rules:
            fp = rule.field_path
            expected_val = rule.evidence[0].value if rule.evidence else None
            gold_boxes = ev_boxes.get(fp, [])

            if not gold_boxes:
                continue

            gold_page, gold_bbox = gold_boxes[0]

            pred_cit = pred_cits_by_path.get(fp)
            pred_page = pred_cit.get("page") if pred_cit else None
            pred_bbox = pred_cit.get("bbox") if pred_cit else None
            pred_val = pred_cit.get("reference_text") if pred_cit else None

            res = classifier.classify_field(
                document_id=test_id,
                field_path=fp,
                gold_value=expected_val,
                predicted_value=pred_val,
                gold_page=gold_page,
                predicted_page=pred_page,
                gold_bbox=gold_bbox,
                predicted_bbox=pred_bbox,
                experiment_run=run_id,
            )
            results.append(res.to_dict())

        return {"test_id": test_id, "error": None, "fields": results}

    except Exception as e:
        return {"test_id": test_id, "error": str(e), "fields": []}


class MicroscopeV4Runner:
    """Executes full or sample benchmark microscope audit and generates V4 reports."""

    def __init__(
        self,
        pred_dir: Path | str,
        output_dir: Path | str,
        run_id: str,
        data_dir: Path | str = "research/data/full",
    ) -> None:
        self.pred_dir = Path(pred_dir)
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.data_dir = Path(data_dir)

    def run(self, max_workers: int = 1, sample_docs: list[str] | None = None) -> dict[str, Path]:
        print(f"=== Running Failure Microscope V4 [{self.run_id}] ===")
        t0 = time.perf_counter()

        pred_files = sorted(list(self.pred_dir.glob("**/*.result.json")))
        if not pred_files:
            raise FileNotFoundError(f"No prediction files found in {self.pred_dir}")

        tasks = []
        for pf in pred_files:
            test_id = pf.relative_to(self.pred_dir).as_posix().removesuffix(".result.json")
            if sample_docs is not None and test_id not in sample_docs:
                continue
            pdf_path = self.data_dir / f"{test_id}.pdf"
            if not pdf_path.exists():
                continue
            tasks.append({
                "test_id": test_id,
                "pdf_path": str(pdf_path),
                "pred_path": str(pf),
                "run_id": self.run_id,
            })

        print(f"Auditing {len(tasks)} documents...")
        all_records: list[dict[str, Any]] = []

        for idx, task in enumerate(tasks):
            t_doc = time.perf_counter()
            res = _audit_document(task)
            if res.get("error"):
                print(f"  [{idx+1}/{len(tasks)}] {task['test_id']}: ERROR {res['error']}")
            else:
                all_records.extend(res["fields"])
                if (idx + 1) % 25 == 0 or (idx + 1) == len(tasks):
                    dt = time.perf_counter() - t_doc
                    print(f"  [{idx+1}/{len(tasks)}] {task['test_id']} ({len(res['fields'])} fields, {dt:.2f}s)")
            gc.collect()

        t_audit = time.perf_counter() - t0
        print(f"Causal audit completed in {t_audit:.2f}s ({len(all_records)} total fields).")

        # Generate reports via FailureMicroscopeReporter
        reporter = FailureMicroscopeReporter(run_id=self.run_id, output_dir=self.output_dir)
        paths = reporter.generate_reports(all_records, total_benchmark_docs=236)
        print(f"Microscope V4 reports generated under: {self.output_dir}")
        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Failure Microscope V4")
    parser.add_argument("--pred-dir", required=True, help="Directory containing .result.json predictions")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    args = parser.parse_args()

    runner = MicroscopeV4Runner(
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    runner.run()
