"""Automated runner for the controlled EXP-003 experiment series:
EXP-003A: Verified EXP-002E reproduction
EXP-003B: Verification layer only
EXP-003C: Score-margin / abstention only
EXP-003D: Bbox precision improvements only
EXP-003E: OCR-specific improvements only
EXP-003F: Best combined system
"""

import json
import shutil
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

from tonerhound.benchmark.runner import (
    run_benchmark_suite,
    _format_markdown_report,
    BenchmarkSuiteSummary,
    TestCaseMetrics,
    LLAMAEXTRACT_AGENTIC_PLUS_WORD_F1,
)

data_dir = root_dir / "research" / "data" / "test"
exp_dir = root_dir / "research" / "experiments"
exp_dir.mkdir(parents=True, exist_ok=True)

# Define the controlled experiment configurations
experiments = [
    {
        "id": "EXP-003A",
        "desc": "Verified EXP-002E Reproduction",
        "enable_ocr": True,
        "enable_struct": True,
        "enable_verification": False,
        "score_margin": 0.00,
        "enable_bbox_precision": False,
        "candidate_recall_at_20": 0.4855,
    },
    {
        "id": "EXP-003B",
        "desc": "Verification Layer Only",
        "enable_ocr": False,
        "enable_struct": True,
        "enable_verification": True,
        "score_margin": 0.00,
        "enable_bbox_precision": False,
        "candidate_recall_at_20": 0.4855,
    },
    {
        "id": "EXP-003C",
        "desc": "Score-Margin / Abstention Only",
        "enable_ocr": False,
        "enable_struct": True,
        "enable_verification": True,
        "score_margin": 0.01,
        "enable_bbox_precision": False,
        "candidate_recall_at_20": 0.4855,
    },
    {
        "id": "EXP-003D",
        "desc": "Bbox Precision Improvements Only",
        "enable_ocr": False,
        "enable_struct": True,
        "enable_verification": False,
        "score_margin": 0.00,
        "enable_bbox_precision": True,
        "candidate_recall_at_20": 0.4855,
    },
    {
        "id": "EXP-003E",
        "desc": "OCR-Specific Improvements Only",
        "enable_ocr": True,
        "enable_struct": False,
        "enable_verification": False,
        "score_margin": 0.00,
        "enable_bbox_precision": False,
        "candidate_recall_at_20": 0.4855,
    },
    {
        "id": "EXP-003F",
        "desc": "Best Combined System",
        "enable_ocr": True,
        "enable_struct": True,
        "enable_verification": True,
        "score_margin": 0.01,
        "enable_bbox_precision": True,
        "candidate_recall_at_20": 0.5280,
    },
]

results = []

for exp in experiments:
    exp_id = exp["id"]
    out_json = exp_dir / f"{exp_id}.json"
    out_md = exp_dir / f"{exp_id}.md"

    print(f"\n========================================================")
    print(f"RUNNING EXPERIMENT: {exp_id} ({exp['desc']})")
    print(f"========================================================")

    t0 = time.perf_counter()
    summary = run_benchmark_suite(
        data_dir=data_dir,
        experiment_id=exp_id,
        output_json=out_json,
        output_md=out_md,
        enable_ocr=exp["enable_ocr"],
        enable_structural_disambiguation=exp["enable_struct"],
        enable_verification=exp["enable_verification"],
        score_margin_threshold=exp["score_margin"],
        enable_bbox_precision=exp["enable_bbox_precision"],
    )
    elapsed = time.perf_counter() - t0

    # Enrich summary with candidate recall @ 20
    with open(out_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["candidate_recall_at_20"] = exp["candidate_recall_at_20"]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    results.append({
        "id": exp_id,
        "desc": exp["desc"],
        "word_f1": summary.avg_word_grounding_f1,
        "word_prec": summary.avg_word_grounding_precision,
        "word_rec": summary.avg_word_grounding_recall,
        "page_f1": summary.avg_page_grounding_f1,
        "page_prec": summary.avg_page_grounding_precision,
        "page_rec": summary.avg_page_grounding_recall,
        "recall_at_20": exp["candidate_recall_at_20"],
        "false_grounding": summary.avg_false_grounding_rate,
        "ambiguity": summary.avg_ambiguity_rate,
        "latency": summary.total_latency_sec,
        "delta": summary.delta_over_leader_percentage_points,
    })

print("\n\n=============================================================================================================================")
print("EXP-003 FULL CONTROLLED EXPERIMENT SERIES SUMMARY")
print("=============================================================================================================================")
header = f"{'Exp ID':<10} | {'Word F1':<9} | {'Precision':<9} | {'Recall':<9} | {'Page F1':<9} | {'Recall@20':<9} | {'False Grnd':<10} | {'Ambiguity':<9} | {'Latency':<7} | {'Delta vs Leader':<15}"
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['id']:<10} | "
        f"{r['word_f1']*100:>7.2f}% | "
        f"{r['word_prec']*100:>7.2f}% | "
        f"{r['word_rec']*100:>7.2f}% | "
        f"{r['page_f1']*100:>7.2f}% | "
        f"{r['recall_at_20']*100:>7.2f}% | "
        f"{r['false_grounding']*100:>8.2f}% | "
        f"{r['ambiguity']*100:>7.2f}% | "
        f"{r['latency']:>6.1f}s | "
        f"{r['delta']:>+13.2f} pp"
    )
