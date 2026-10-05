"""Authoritative Failure Microscope V3 Runner.

Executes causal failure classification across ExtractBench documents using the
deterministic FailureMicroscopeClassifier and FailureMicroscopeReporter.

Adheres strictly to the required failure taxonomy and validation requirements.
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

        expected_leaves = _flatten_expected_leaves(tc.expected_output)
        field_records = []

        for path, val in expected_leaves:
            if val is None:
                continue

            gold_boxes = ev_boxes.get(path, [])
            if not gold_boxes:
                # Field has no ground truth bounding box / ungradeable for grounding
                continue

            gold_pages = sorted(list(ev_pages.get(path, set())))
            gold_page = gold_pages[0] if gold_pages else None
            gold_box = gold_boxes[0][1] if gold_boxes else None

            # Prediction citation
            pred_cit = pred_cits_by_path.get(path)
            pred_page = pred_cit["page"] if pred_cit else None
            pred_bbox = pred_cit["bbox"] if pred_cit else None
            pred_val = pred_cit.get("reference_text") if pred_cit else None

            # Fast path for resolved fields: skip expensive candidate collection
            pred_iou = 0.0
            same_page = (
                gold_page is not None
                and pred_page is not None
                and gold_page == pred_page
            )
            if same_page and gold_box and pred_bbox:
                pred_iou = iou_xywh(pred_bbox, gold_box)

            if same_page and pred_iou >= 0.50:
                res = classifier.classify_field(
                    document_id=test_id,
                    field_path=path,
                    gold_value=val,
                    predicted_value=pred_val,
                    gold_page=gold_page,
                    predicted_page=pred_page,
                    gold_bbox=gold_box,
                    predicted_bbox=pred_bbox,
                    candidates=[],
                    experiment_run=run_id,
                )
                field_records.append(res.to_dict())
                continue

            # Failing field: trace causal candidate generation
            p_hint = gold_page if gold_page is not None else pred_page
            cand_inp = ExtractionInput(field=path, value=val, page_hint=p_hint)
            candidates = resolver.collect_candidates(cand_inp)

            # Classify causal mechanism
            res = classifier.classify_field(
                document_id=test_id,
                field_path=path,
                gold_value=val,
                predicted_value=pred_val,
                gold_page=gold_page,
                predicted_page=pred_page,
                gold_bbox=gold_box,
                predicted_bbox=pred_bbox,
                candidates=candidates,
                resolver_result=None,
                experiment_run=run_id,
            )
            field_records.append(res.to_dict())

        gc.collect()
        return {"test_id": test_id, "error": None, "fields": field_records}

    except Exception as e:
        return {"test_id": test_id, "error": str(e), "fields": []}


class MicroscopeRunner:
    """Orchestrates running the Failure Microscope across a collection of documents."""

    def __init__(
        self,
        predictions_dir: Path | str,
        data_dir: Path | str = repo_root / "research" / "data" / "full",
        output_dir: Path | str = repo_root / "research" / "observer" / "reports",
        run_id: str = "microscope_v3",
        workers: int = 2,
    ) -> None:
        self.predictions_dir = Path(predictions_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) / run_id
        self.run_id = run_id
        self.workers = workers

    def run(
        self,
        test_ids: list[str] | None = None,
        max_docs: int | None = None,
    ) -> dict[str, Any]:
        """Execute microscope classification and generate reports."""
        print(f"=== Starting Failure Microscope V3 Run: {self.run_id} ===")
        print(f"Predictions Dir: {self.predictions_dir}")
        print(f"Output Dir:      {self.output_dir}")

        # Find all prediction files
        pred_files: list[Path] = []
        for p in self.predictions_dir.rglob("*.json"):
            if p.name.endswith(".result.json") or p.name.endswith(".eval.json"):
                pred_files.append(p)

        # Build mapping from test_id to (pdf_path, pred_path)
        tasks: list[dict[str, Any]] = []
        for pf in sorted(pred_files):
            # Parse test_id e.g. short/arif-2022
            rel = pf.relative_to(self.predictions_dir)
            parts = rel.parts
            if "tonerhound" in parts:
                idx = parts.index("tonerhound")
                parts = parts[idx + 1 :]

            if len(parts) >= 2:
                split = parts[0]
                doc_name = parts[1].replace(".result.json", "").replace(".eval.json", "")
                test_id = f"{split}/{doc_name}"
            else:
                doc_name = pf.stem.replace(".result", "").replace(".eval", "")
                test_id = doc_name

            if test_ids is not None and test_id not in test_ids:
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

        if max_docs is not None:
            tasks = tasks[:max_docs]

        print(f"Auditing {len(tasks)} documents with {self.workers} worker processes...")

        all_fields: list[dict[str, Any]] = []
        doc_totals: dict[str, int] = {}
        t0 = time.perf_counter()

        if self.workers <= 1:
            for i, task in enumerate(tasks, start=1):
                res = _audit_document(task)
                if res["error"]:
                    print(f"[{i}/{len(tasks)}] Error on {task['test_id']}: {res['error']}")
                else:
                    all_fields.extend(res["fields"])
                    doc_totals[task["test_id"]] = len(res["fields"])
                    print(f"[{i}/{len(tasks)}] {task['test_id']}: {len(res['fields'])} fields audited")
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(_audit_document, t): t for t in tasks}
                completed = 0
                for fut in as_completed(futures):
                    completed += 1
                    t = futures[fut]
                    try:
                        res = fut.result()
                        if res["error"]:
                            print(f"[{completed}/{len(tasks)}] Error on {t['test_id']}: {res['error']}")
                        else:
                            all_fields.extend(res["fields"])
                            doc_totals[t["test_id"]] = len(res["fields"])
                            if completed % 25 == 0 or completed == len(tasks):
                                elapsed = time.perf_counter() - t0
                                print(f"[{completed}/{len(tasks)} docs] Audited {len(all_fields)} fields ({elapsed:.1f}s)...")
                    except Exception as exc:
                        print(f"[{completed}/{len(tasks)}] Exception on {t['test_id']}: {exc}")

        elapsed_total = time.perf_counter() - t0
        print(f"Audit completed in {elapsed_total:.1f}s. Total gradeable fields audited: {len(all_fields)}")

        # Generate reports
        reporter = FailureMicroscopeReporter(run_id=self.run_id, output_dir=self.output_dir)
        report_paths = reporter.generate_reports(
            field_records=all_fields,
            total_benchmark_docs=len(tasks),
            doc_total_fields=doc_totals,
        )

        print(f"Reports generated successfully in {self.output_dir}:")
        for name, p in report_paths.items():
            print(f"  - {name}: {p.name}")

        return {
            "run_id": self.run_id,
            "total_documents": len(tasks),
            "total_fields": len(all_fields),
            "runtime_seconds": elapsed_total,
            "report_paths": {k: str(v) for k, v in report_paths.items()},
        }


def main():
    parser = argparse.ArgumentParser(description="Run Authoritative Failure Microscope V3")
    parser.add_argument("--predictions-dir", type=str, required=True, help="Directory containing prediction result JSON files")
    parser.add_argument("--data-dir", type=str, default=str(repo_root / "research" / "data" / "full"), help="ExtractBench data directory")
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "research" / "observer" / "reports"), help="Output directory for reports")
    parser.add_argument("--run-id", type=str, default="run_v3", help="Run identifier")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit number of documents")
    args = parser.parse_args()

    runner = MicroscopeRunner(
        predictions_dir=args.predictions_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        workers=args.workers,
    )
    runner.run(max_docs=args.max_docs)


if __name__ == "__main__":
    main()
