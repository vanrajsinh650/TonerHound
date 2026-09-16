#!/usr/bin/env python3
"""Automated Regression Auditor for EXP-005 Non-FTX Control Documents.

Verifies that adaptations to the long-document and FTX grounding pipeline
maintain exact baseline checkpoints and introduce ZERO regressions across
key non-FTX control documents spanning diverse domains:
  1. short/13f__sl_advisors_llc             (Form 13F Holdings)
  2. short/nport__bullfinch_fund_inc        (Form N-PORT Mutual Fund Schedule)
  3. medium/cabrera-2023                    (IRS Form 1040 Tax Return)
  4. long/real_credit_strategies_full       (Schedule of Investments)
  5. short/real_clinton_property_25_11073_corrupted (Scanned Deed & Conveyance)

Baseline checkpoints established during EXP-005 Part 3 / Part 4:
  - short/13f__sl_advisors_llc:             99.80% Word F1 | 100.00% Page F1
  - short/nport__bullfinch_fund_inc:        91.44% Word F1 | 100.00% Page F1
  - medium/cabrera-2023:                    26.67% Word F1 |  68.99% Page F1
  - long/real_credit_strategies_full:       19.29% Word F1 |  77.57% Page F1
  - short/real_clinton_property_25_11073_corrupted: 47.83% Word F1 | 94.12% Page F1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure repository paths are accessible
REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACTBENCH_SRC = REPO_ROOT / "research" / "reference" / "ExtractBench" / "src"
if EXTRACTBENCH_SRC.exists() and str(EXTRACTBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(EXTRACTBENCH_SRC))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from extract_bench.test_cases.loader import load_test_case
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.document.index import DocumentIndex


@dataclass(frozen=True)
class BaselineCheckpoint:
    doc_id: str
    domain: str
    num_pages: int
    enable_ocr: bool
    word_f1: float
    page_f1: float
    word_precision: float
    word_recall: float
    page_precision: float
    page_recall: float


# Ground-truth verified baseline checkpoints for regression testing
BASELINE_CHECKPOINTS: dict[str, BaselineCheckpoint] = {
    "short/13f__sl_advisors_llc": BaselineCheckpoint(
        doc_id="short/13f__sl_advisors_llc",
        domain="13F Holdings",
        num_pages=2,
        enable_ocr=False,
        word_f1=0.99802372,
        page_f1=1.00000000,
        word_precision=0.99802372,
        word_recall=0.99802372,
        page_precision=1.00000000,
        page_recall=1.00000000,
    ),
    "short/nport__bullfinch_fund_inc": BaselineCheckpoint(
        doc_id="short/nport__bullfinch_fund_inc",
        domain="Mutual Fund Schedule",
        num_pages=3,
        enable_ocr=False,
        word_f1=0.91438980,
        page_f1=1.00000000,
        word_precision=0.91438980,
        word_recall=0.91438980,
        page_precision=1.00000000,
        page_recall=1.00000000,
    ),
    "medium/cabrera-2023": BaselineCheckpoint(
        doc_id="medium/cabrera-2023",
        domain="IRS Form 1040 Tax",
        num_pages=28,
        enable_ocr=True,
        word_f1=0.26666667,
        page_f1=0.68985507,
        word_precision=0.29113924,
        word_recall=0.24598930,
        page_precision=0.75316456,
        page_recall=0.63636364,
    ),
    "long/real_credit_strategies_full": BaselineCheckpoint(
        doc_id="long/real_credit_strategies_full",
        domain="Investment Schedule",
        num_pages=59,
        enable_ocr=False,
        word_f1=0.19292833,
        page_f1=0.77571845,
        word_precision=0.21743782,
        word_recall=0.17338452,
        page_precision=0.86087358,
        page_recall=0.70589350,
    ),
    "short/real_clinton_property_25_11073_corrupted": BaselineCheckpoint(
        doc_id="short/real_clinton_property_25_11073_corrupted",
        domain="Deed & Conveyance (OCR)",
        num_pages=9,
        enable_ocr=True,
        word_f1=0.47826087,
        page_f1=0.94117647,
        word_precision=0.49624060,
        word_recall=0.46153846,
        page_precision=0.97297297,
        page_recall=0.91139241,
    ),
}


@dataclass
class DocumentAuditResult:
    doc_id: str
    domain: str
    baseline_word_f1: float
    current_word_f1: float
    delta_word_f1: float
    baseline_page_f1: float
    current_page_f1: float
    delta_page_f1: float
    current_word_precision: float
    current_word_recall: float
    current_page_precision: float
    current_page_recall: float
    num_citations: int
    indexing_time_sec: float
    grounding_time_sec: float
    eval_time_sec: float
    total_time_sec: float
    is_regressed: bool
    status: str


def run_single_audit(
    doc_id: str,
    baseline: BaselineCheckpoint,
    tolerance: float = 1e-4,
    verbose: bool = False,
) -> DocumentAuditResult:
    """Run ExtractBenchAdapter and evaluate metrics on a single control document."""
    pdf_path = REPO_ROOT / "research" / "data" / "full" / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Control document PDF not found at {pdf_path}")

    t_start = time.perf_counter()
    tc = load_test_case(pdf_path)

    # 1. Index document
    t0 = time.perf_counter()
    doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=baseline.enable_ocr)
    t_idx = time.perf_counter() - t0

    # 2. Ground extracted data with standard ExtractBenchAdapter
    t1 = time.perf_counter()
    adapter = ExtractBenchAdapter(
        doc_index,
        enable_structural_disambiguation=True,
        enable_verification=True,
        score_margin_threshold=0.01,
        enable_bbox_precision=True,
    )
    payload = adapter.ground_extracted_data(tc.expected_output, example_id=doc_id)
    citations = payload.get("field_citations", [])
    t_ground = time.perf_counter() - t1

    # 3. Official ExtractBench evaluation
    t2 = time.perf_counter()
    eval_res = evaluate_prediction(
        expected_output=tc.expected_output,
        extracted_data=tc.expected_output,
        field_rules=tc.test_rules,
        field_citations=citations,
        data_schema=tc.data_schema,
    )
    t_eval = time.perf_counter() - t2
    t_total = time.perf_counter() - t_start

    current_wf1 = float(eval_res.get("word_grounding_f1") or 0.0)
    current_pf1 = float(eval_res.get("page_grounding_f1") or 0.0)
    current_wp = float(eval_res.get("word_grounding_precision") or 0.0)
    current_wr = float(eval_res.get("word_grounding_recall") or 0.0)
    current_pp = float(eval_res.get("page_grounding_precision") or 0.0)
    current_pr = float(eval_res.get("page_grounding_recall") or 0.0)

    delta_wf1 = current_wf1 - baseline.word_f1
    delta_pf1 = current_pf1 - baseline.page_f1

    # Regression detected if current metric drops below baseline minus numerical tolerance
    is_regressed = (delta_wf1 < -tolerance) or (delta_pf1 < -tolerance)
    status = "REGRESSION" if is_regressed else "PASS"

    if verbose:
        print(f"  [{doc_id}] wf1={current_wf1*100:.4f}% (delta={delta_wf1*100:+.4f}pp), "
              f"pf1={current_pf1*100:.4f}% (delta={delta_pf1*100:+.4f}pp), status={status}")

    return DocumentAuditResult(
        doc_id=doc_id,
        domain=baseline.domain,
        baseline_word_f1=baseline.word_f1,
        current_word_f1=current_wf1,
        delta_word_f1=delta_wf1,
        baseline_page_f1=baseline.page_f1,
        current_page_f1=current_pf1,
        delta_page_f1=delta_pf1,
        current_word_precision=current_wp,
        current_word_recall=current_wr,
        current_page_precision=current_pp,
        current_page_recall=current_pr,
        num_citations=len(citations),
        indexing_time_sec=t_idx,
        grounding_time_sec=t_ground,
        eval_time_sec=t_eval,
        total_time_sec=t_total,
        is_regressed=is_regressed,
        status=status,
    )


def audit_control_documents(
    doc_filter: list[str] | None = None,
    tolerance: float = 1e-4,
    verbose: bool = False,
) -> list[DocumentAuditResult]:
    """Execute audit across specified or all 5 control documents."""
    targets = doc_filter or list(BASELINE_CHECKPOINTS.keys())
    results: list[DocumentAuditResult] = []

    for doc_id in targets:
        if doc_id not in BASELINE_CHECKPOINTS:
            raise KeyError(f"Unknown control document ID: {doc_id}. Valid options: {list(BASELINE_CHECKPOINTS.keys())}")
        baseline = BASELINE_CHECKPOINTS[doc_id]
        res = run_single_audit(doc_id, baseline, tolerance=tolerance, verbose=verbose)
        results.append(res)

    return results


def print_audit_report(results: list[DocumentAuditResult], tolerance: float = 1e-4) -> bool:
    """Print structured audit report table and return True if zero regressions."""
    print("\n" + "=" * 110)
    print("EXP-005 NON-FTX CONTROL DOCUMENT REGRESSION AUDIT REPORT")
    print("=" * 110)
    header = (
        f"{'Control Document':<47} | {'Domain':<20} | "
        f"{'Word F1 (Cur / Base)':<20} | {'Page F1 (Cur / Base)':<20} | {'Status':<10}"
    )
    print(header)
    print("-" * 110)

    any_regressions = False
    total_time = 0.0

    for r in results:
        total_time += r.total_time_sec
        if r.is_regressed:
            any_regressions = True

        cur_wf1_str = f"{r.current_word_f1 * 100:6.2f}%"
        base_wf1_str = f"{r.baseline_word_f1 * 100:6.2f}%"
        wf1_combined = f"{cur_wf1_str} / {base_wf1_str}"

        cur_pf1_str = f"{r.current_page_f1 * 100:6.2f}%"
        base_pf1_str = f"{r.baseline_page_f1 * 100:6.2f}%"
        pf1_combined = f"{cur_pf1_str} / {base_pf1_str}"

        print(
            f"{r.doc_id:<47} | {r.domain:<20} | "
            f"{wf1_combined:<20} | {pf1_combined:<20} | {r.status:<10}"
        )

    print("-" * 110)
    print(f"Total Audit Execution Time: {total_time:.2f}s across {len(results)} control documents")

    print("\nDetailed Metric Breakdown:")
    for r in results:
        delta_w_sign = "+" if r.delta_word_f1 >= 0 else ""
        delta_p_sign = "+" if r.delta_page_f1 >= 0 else ""
        print(f"  * {r.doc_id}:")
        print(f"      Word F1: {r.current_word_f1 * 100:6.2f}% (Baseline: {r.baseline_word_f1 * 100:6.2f}%, "
              f"Delta: {delta_w_sign}{r.delta_word_f1 * 100:.2f} pp)")
        print(f"      Page F1: {r.current_page_f1 * 100:6.2f}% (Baseline: {r.baseline_page_f1 * 100:6.2f}%, "
              f"Delta: {delta_p_sign}{r.delta_page_f1 * 100:.2f} pp)")
        print(f"      Precision / Recall: Word P={r.current_word_precision * 100:.2f}%, R={r.current_word_recall * 100:.2f}% | "
              f"Page P={r.current_page_precision * 100:.2f}%, R={r.current_page_recall * 100:.2f}%")
        print(f"      Citations: {r.num_citations} | Latency: {r.total_time_sec:.2f}s "
              f"(Idx: {r.indexing_time_sec:.2f}s, Grd: {r.grounding_time_sec:.2f}s, Eval: {r.eval_time_sec:.2f}s)")

    print("=" * 110)
    if not any_regressions:
        print("RESULT: SUCCESS - ZERO REGRESSIONS DETECTED ACROSS ALL CONTROL DOCUMENTS.")
        print("Baseline integrity verified. Adapter is safe for deployment / evaluation.")
    else:
        print("RESULT: FAILURE - REGRESSION DETECTED IN ONE OR MORE CONTROL DOCUMENTS!")
    print("=" * 110 + "\n")

    return not any_regressions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify zero regressions across EXP-005 non-FTX control documents."
    )
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        help="Run audit only on specific document ID (can specify multiple times).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Allowed drop before flagging regression (default: 1e-4 / 0.01 pp).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "experiments" / "EXP-005-control-regressions.json",
        help="Path to save audit results JSON.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose progress output during audit execution.",
    )
    args = parser.parse_args()

    print("Starting EXP-005 Non-FTX Control Document Audit...")
    results = audit_control_documents(
        doc_filter=args.docs,
        tolerance=args.tolerance,
        verbose=args.verbose,
    )

    success = print_audit_report(results, tolerance=args.tolerance)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "experiment_id": "EXP-005-control-regression-audit",
            "tolerance": args.tolerance,
            "all_passed": success,
            "documents": [asdict(r) for r in results],
        }
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved audit JSON artifact to: {args.json_output.relative_to(REPO_ROOT)}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
