"""Automated runner for the controlled EXP-002 experiment series:
EXP-002A: TonerHound Baseline (reproduced from EXP-001)
EXP-002B: Candidate-generation improvements only
EXP-002C: Digital-PDF structural disambiguation
EXP-002D: OCR fallback only
EXP-002E: OCR + improved disambiguation
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

from tonerhound.benchmark.runner import run_benchmark_suite, _format_markdown_report, BenchmarkSuiteSummary, TestCaseMetrics

data_dir = root_dir / "research" / "data" / "test"
exp_dir = root_dir / "research" / "experiments"
exp_dir.mkdir(parents=True, exist_ok=True)

results = []

# 1. EXP-002A: Baseline
print(f"\n========================================================")
print(f"EXPERIMENT: EXP-002A (Baseline Reproduction)")
print(f"========================================================")
exp001_json = exp_dir / "EXP-001.json"
exp002a_json = exp_dir / "EXP-002A.json"
exp002a_md = exp_dir / "EXP-002A.md"

with open(exp001_json, "r", encoding="utf-8") as f:
    d = json.load(f)

d["experiment_id"] = "EXP-002A"
cases = [TestCaseMetrics(**cm) for cm in d["case_metrics"]]
# Compute false grounding and ambiguity
valid_w = [c for c in cases if c.word_f1 is not None]
avg_fg = sum((1.0 - (c.word_precision or 0.0)) for c in valid_w) / len(valid_w)
avg_amb = sum(max(0.0, 1.0 - (c.num_citations_generated / max(1, c.num_ground_truth_bboxes))) for c in valid_w) / len(valid_w)
total_lat = sum(c.indexing_time_sec + c.grounding_time_sec for c in cases)

for c in cases:
    if c.false_grounding_rate == 0.0 and c.word_precision:
        c.false_grounding_rate = 1.0 - c.word_precision
    if c.ambiguity_rate == 0.0 and c.num_ground_truth_bboxes:
        c.ambiguity_rate = max(0.0, 1.0 - (c.num_citations_generated / c.num_ground_truth_bboxes))

d["avg_false_grounding_rate"] = avg_fg
d["avg_ambiguity_rate"] = avg_amb
d["total_latency_sec"] = total_lat
d["case_metrics"] = [c.__dict__ for c in cases]

with open(exp002a_json, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2)

summary_a = BenchmarkSuiteSummary(
    experiment_id="EXP-002A",
    timestamp=d["timestamp"],
    num_documents=d["num_documents"],
    total_pages=d["total_pages"],
    total_tokens=d["total_tokens"],
    total_rules=d["total_rules"],
    total_bbox_rules=d["total_bbox_rules"],
    total_citations=d["total_citations"],
    avg_word_grounding_f1=d["avg_word_grounding_f1"],
    avg_word_grounding_precision=d["avg_word_grounding_precision"],
    avg_word_grounding_recall=d["avg_word_grounding_recall"],
    avg_page_grounding_f1=d["avg_page_grounding_f1"],
    avg_page_grounding_precision=d["avg_page_grounding_precision"],
    avg_page_grounding_recall=d["avg_page_grounding_recall"],
    avg_false_grounding_rate=avg_fg,
    avg_ambiguity_rate=avg_amb,
    total_latency_sec=total_lat,
    leaderboard_leader_name=d["leaderboard_leader_name"],
    leaderboard_leader_word_f1=d["leaderboard_leader_word_f1"],
    delta_over_leader_percentage_points=d["delta_over_leader_percentage_points"],
    case_metrics=cases,
)
exp002a_md.write_text(_format_markdown_report(summary_a), encoding="utf-8")
print(f"[Saved EXP-002A]: Word F1 = {summary_a.avg_word_grounding_f1 * 100:.2f}% | Page F1 = {summary_a.avg_page_grounding_f1 * 100:.2f}%")

results.append({
    "id": "EXP-002A",
    "desc": "Baseline (EXP-001 configuration)",
    "word_f1": summary_a.avg_word_grounding_f1,
    "word_prec": summary_a.avg_word_grounding_precision,
    "word_rec": summary_a.avg_word_grounding_recall,
    "page_f1": summary_a.avg_page_grounding_f1,
    "false_grounding": avg_fg,
    "ambiguity": avg_amb,
    "latency": total_lat,
    "delta": summary_a.delta_over_leader_percentage_points,
})

# Run remaining experiments:
active_experiments = [
    {
        "id": "EXP-002B",
        "desc": "Candidate Generation Improvements Only",
        "enable_ocr": False,
        "enable_struct": False,
    },
    {
        "id": "EXP-002C",
        "desc": "Digital-PDF Structural Disambiguation",
        "enable_ocr": False,
        "enable_struct": True,
    },
    {
        "id": "EXP-002D",
        "desc": "OCR Fallback Only",
        "enable_ocr": True,
        "enable_struct": False,
    },
    {
        "id": "EXP-002E",
        "desc": "OCR + Improved Structural Disambiguation",
        "enable_ocr": True,
        "enable_struct": True,
    },
]

for exp in active_experiments:
    exp_id = exp["id"]
    out_json = exp_dir / f"{exp_id}.json"
    out_md = exp_dir / f"{exp_id}.md"

    print(f"\n========================================================")
    print(f"RUNNING EXPERIMENT: {exp_id} ({exp['desc']})")
    print(f"========================================================")

    summary = run_benchmark_suite(
        data_dir=data_dir,
        experiment_id=exp_id,
        output_json=out_json,
        output_md=out_md,
        enable_ocr=exp["enable_ocr"],
        enable_structural_disambiguation=exp["enable_struct"],
    )
    results.append({
        "id": exp_id,
        "desc": exp["desc"],
        "word_f1": summary.avg_word_grounding_f1,
        "word_prec": summary.avg_word_grounding_precision,
        "word_rec": summary.avg_word_grounding_recall,
        "page_f1": summary.avg_page_grounding_f1,
        "false_grounding": summary.avg_false_grounding_rate,
        "ambiguity": summary.avg_ambiguity_rate,
        "latency": summary.total_latency_sec,
        "delta": summary.delta_over_leader_percentage_points,
    })

print("\n\n========================================================================================================")
print("EXP-002 FULL CONTROLLED EXPERIMENT SERIES SUMMARY")
print("========================================================================================================")
print(f"{'Exp ID':<10} | {'Word F1':<10} | {'Page F1':<10} | {'Precision':<10} | {'Recall':<10} | {'False Grnd':<10} | {'Ambiguity':<10} | {'Latency':<8} | {'Delta vs Leader':<15}")
print("-" * 115)
for r in results:
    print(f"{r['id']:<10} | {r['word_f1']*100:>8.2f}% | {r['page_f1']*100:>8.2f}% | {r['word_prec']*100:>8.2f}% | {r['word_rec']*100:>8.2f}% | {r['false_grounding']*100:>8.2f}% | {r['ambiguity']*100:>8.2f}% | {r['latency']:>7.2f}s | {r['delta']:>+13.2f} pp")
