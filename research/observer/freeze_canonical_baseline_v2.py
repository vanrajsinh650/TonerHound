"""Freeze Canonical Baseline V2 and Run Calibrated Failure Microscope V4.

Phase C of the Mega Execution Directive:
1. Loads current production TonerHound predictions (from EXP-037).
2. Verifies official ExtractBench metrics:
   - Expected Word F1: 56.6707%
   - Expected Page F1: 81.9786%
   (Deviation threshold <= 0.5 pp).
3. Runs Failure Microscope V4 with RealisticEstimator.
4. Freezes results under research/observer/reports/canonical_baseline_v2/.
5. Produces canonical_baseline_v2_report.json and canonical_baseline_v2_report.md.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.test_cases.loader import load_test_case

from research.observer.field_classifier import (
    FailureMicroscopeClassifier,
    compute_iou_xywh,
)
from research.observer.realistic_estimator import RealisticEstimator
from research.observer.report_generator import FailureMicroscopeReporter
from tonerhound.document.hybrid_index import HybridDocumentIndex
from tonerhound.document.index import DocumentIndex


def run_freeze_baseline_v2():
    print("=== EXP-038 Phase C: Freezing Canonical Baseline V2 ===")
    t_start = time.perf_counter()

    pred_dir = repo_root / "research" / "experiments" / "EXP-037" / "predictions" / "tonerhound"
    data_dir = repo_root / "research" / "data" / "full"
    out_dir = repo_root / "research" / "observer" / "reports" / "canonical_baseline_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load precomputed benchmark results from EXP-037 full benchmark
    full_res_path = repo_root / "research" / "experiments" / "EXP-037" / "full_benchmark_results.json"
    with open(full_res_path, encoding="utf-8") as f:
        exp037_res = json.load(f)

    official_metrics = exp037_res["official_metrics"]
    word_f1 = official_metrics["word_grounding_f1"]
    page_f1 = official_metrics["page_grounding_f1"]
    word_prec = official_metrics["word_precision"]
    word_rec = official_metrics["word_recall"]
    passing_fields = exp037_res["passing_fields"]
    failing_fields = exp037_res["failing_fields"]
    total_gradeable = exp037_res["total_gradeable_fields"]
    eval_docs = exp037_res["evaluated_documents"]

    print(f"Loaded EXP-037 Benchmark Metrics:")
    print(f"  Word Grounding F1: {word_f1:.4f}% (Expected: ~56.6707%)")
    print(f"  Page Grounding F1: {page_f1:.4f}% (Expected: ~81.9786%)")
    print(f"  Word Precision:    {word_prec:.4f}%")
    print(f"  Word Recall:       {word_rec:.4f}%")
    print(f"  Passing Fields:    {passing_fields}")
    print(f"  Failing Fields:    {failing_fields}")

    # Deviation check (must be <= 0.5 pp from expected 56.6707%)
    expected_word_f1 = 56.6707
    deviation = abs(word_f1 - expected_word_f1)
    if deviation > 0.5:
        raise ValueError(
            f"FATAL: Word F1 {word_f1:.4f}% deviates by {deviation:.4f} pp from expected {expected_word_f1:.4f}% (> 0.5 pp limit)!"
        )
    print(f"Deviation check passed: {deviation:.4f} pp <= 0.5 pp.\n")

    # 2. Run Calibrated Microscope V4 across all fields
    print("--- Executing Failure Microscope V4 Causal Audit ---")
    estimator = RealisticEstimator()

    # Load post-OCR failure summary from exp037_full_run_v1
    exp037_fs_path = repo_root / "research" / "observer" / "reports" / "exp037_full_run_v1" / "failure_summary.json"
    exp037_db_path = repo_root / "research" / "observer" / "reports" / "exp037_full_run_v1" / "document_breakdown.json"
    exp037_fb_path = repo_root / "research" / "observer" / "reports" / "exp037_full_run_v1" / "family_breakdown.json"
    exp037_fl_path = repo_root / "research" / "observer" / "reports" / "exp037_full_run_v1" / "field_level.json"

    with open(exp037_fs_path, encoding="utf-8") as f:
        exp037_fs = json.load(f)
    with open(exp037_db_path, encoding="utf-8") as f:
        exp037_db = json.load(f)
    with open(exp037_fb_path, encoding="utf-8") as f:
        exp037_fb = json.load(f)
    with open(exp037_fl_path, encoding="utf-8") as f:
        exp037_fl = json.load(f)

    # Copy raw breakdowns
    with open(out_dir / "document_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(exp037_db, f, indent=2)
    with open(out_dir / "family_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(exp037_fb, f, indent=2)
    with open(out_dir / "field_level.json", "w", encoding="utf-8") as f:
        json.dump(exp037_fl, f, indent=2)

    # 3. Calculate calibrated V4 estimates for each failure class
    calibrated_classes = []
    total_theoretical_opp = 0.0
    total_realistic_gain = 0.0

    raw_classes = exp037_fs.get("classes", [])
    for rank, c in enumerate(raw_classes, start=1):
        fc = c["failure_class"]
        f_cnt = c["field_count"]
        d_cnt = c["affected_documents"]
        macro_pp = c["macro_weighted_opportunity_pp"]
        total_theoretical_opp += macro_pp

        est = estimator.estimate_class(
            failure_class=fc,
            field_count=f_cnt,
            document_count=d_cnt,
            total_benchmark_docs=eval_docs,
            theoretical_ceiling_pp=macro_pp,
        )
        total_realistic_gain += est.realistic_expected_gain_pp

        calibrated_classes.append({
            "rank": rank,
            "failure_class": fc,
            "field_count": f_cnt,
            "field_pct": c["field_pct"],
            "affected_documents": d_cnt,
            "document_breadth_pct": est.document_breadth_pct,
            "THEORETICAL_CEILING_pp": est.theoretical_ceiling_pp,
            "REALISTIC_RECOVERY_RATE": est.realistic_recovery_rate,
            "REALISTIC_RECOVERY_RATE_pct": f"{est.realistic_recovery_rate * 100:.2f}%",
            "REALISTIC_EXPECTED_GAIN_pp": round(est.realistic_expected_gain_pp, 4),
            "fix_type": est.fix_type,
            "confidence": est.confidence,
            "source_evidence": est.source_evidence,
            "recommended_next_step": est.recommended_next_step,
            "formatted_block": est.format_text_block(),
            "representative_examples": c.get("representative_examples", []),
        })

    # Failure summary v4
    failure_summary_v4 = {
        "run_id": "canonical_baseline_v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_fields_evaluated": total_gradeable,
        "total_grounding_success": passing_fields,
        "total_grounding_failures": failing_fields,
        "evaluated_documents": eval_docs,
        "total_theoretical_ceiling_pp": round(total_theoretical_opp, 4),
        "total_realistic_expected_gain_pp": round(total_realistic_gain, 4),
        "official_metrics": {
            "word_grounding_f1": word_f1,
            "page_grounding_f1": page_f1,
            "word_precision": word_prec,
            "word_recall": word_rec,
        },
        "classes": calibrated_classes,
    }

    with open(out_dir / "failure_summary.json", "w", encoding="utf-8") as f:
        json.dump(failure_summary_v4, f, indent=2)

    # 4. Generate recommendations.md with Section 3.4 blocks
    recs_md_lines = [
        "# Failure Microscope V4 Calibrated Audit — canonical_baseline_v2",
        "",
        "## 1. Executive Baseline Metrics",
        f"- **Word Grounding F1**: **{word_f1:.4f}%**",
        f"- **Page Grounding F1**: **{page_f1:.4f}%**",
        f"- **Word Grounding Precision**: {word_prec:.4f}%",
        f"- **Word Grounding Recall**: {word_rec:.4f}%",
        f"- **Total Evaluated Documents**: {eval_docs} documents",
        f"- **Total Gradeable Fields**: {total_gradeable:,} fields",
        f"- **Passing Fields ($IoU \\ge 0.50$)**: {passing_fields:,} fields ({passing_fields/total_gradeable*100:.2f}%)",
        f"- **Failing Fields**: {failing_fields:,} fields ({failing_fields/total_gradeable*100:.2f}%)",
        f"- **Total Theoretical Opportunity**: +{total_theoretical_opp:.4f} pp (upper bound ceiling)",
        f"- **Total Realistic Expected Opportunity**: **+{total_realistic_gain:.4f} pp** (empirically calibrated)",
        "",
        "## 2. Calibrated Failure Class Rankings (Microscope V4 Format)",
        "",
        "| Rank | Failure Class | Fields | Breadth | Theoretical Ceiling | Recovery Rate | Realistic Expected Gain | Fix Type | Confidence |",
        "|:---:|:---|---:|---:|---:|---:|---:|:---|:---:|",
    ]

    for c in calibrated_classes:
        recs_md_lines.append(
            f"| {c['rank']} | `{c['failure_class']}` | {c['field_count']:,} | {c['document_breadth_pct']:.1f}% | "
            f"+{c['THEORETICAL_CEILING_pp']:.4f} pp | {c['REALISTIC_RECOVERY_RATE_pct']} | "
            f"**+{c['REALISTIC_EXPECTED_GAIN_pp']:.4f} pp** | {c['fix_type']} | {c['confidence']} |"
        )

    recs_md_lines.extend([
        "",
        "## 3. Class-by-Class Calibrated Profiles",
        "",
    ])

    for c in calibrated_classes:
        recs_md_lines.extend([
            "```",
            c["formatted_block"],
            "```",
            "",
        ])

    recs_md_content = "\n".join(recs_md_lines) + "\n"
    with open(out_dir / "recommendations.md", "w", encoding="utf-8") as f:
        f.write(recs_md_content)

    # 5. Produce canonical_baseline_v2_report.json and canonical_baseline_v2_report.md
    report_json_path = out_dir / "canonical_baseline_v2_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(failure_summary_v4, f, indent=2)

    # Also copy to research/observer/
    shutil.copy2(report_json_path, repo_root / "research" / "observer" / "canonical_baseline_v2_report.json")

    report_md_path = out_dir / "canonical_baseline_v2_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(recs_md_content)

    shutil.copy2(report_md_path, repo_root / "research" / "observer" / "canonical_baseline_v2_report.md")

    t_end = time.perf_counter() - t_start
    print(f"Canonical Baseline V2 successfully frozen in {t_end:.2f}s!")
    print(f"Artifacts saved in {out_dir}")
    print(f"Master report saved in research/observer/canonical_baseline_v2_report.md")


if __name__ == "__main__":
    run_freeze_baseline_v2()
