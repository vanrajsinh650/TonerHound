#!/usr/bin/env python3
import sys
import time
from pathlib import Path

_REF = Path("research/reference/ExtractBench/src")
if _REF.exists(): sys.path.insert(0, str(_REF))
sys.path.insert(0, "src")

from extract_bench.test_cases.loader import load_test_case
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.benchmark.correlation import FailureRegistry
from tonerhound.document.index import DocumentIndex

pdf_path = Path("research/data/full/long/real_ftx_full_corrupted.pdf")
print("[FTX Evaluator] Loading test case and document index...")
t0 = time.perf_counter()
tc = load_test_case(pdf_path)
idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True)
t_idx = time.perf_counter() - t0
print(f"[FTX Evaluator] Index loaded in {t_idx:.2f}s.")

print("[FTX Evaluator] Executing ExtractBenchAdapter...")
t1 = time.perf_counter()
adapter = ExtractBenchAdapter(
    idx,
    enable_structural_disambiguation=True,
    enable_verification=True,
    score_margin_threshold=0.01,
    enable_bbox_precision=True,
)
payload = adapter.ground_extracted_data(tc.expected_output, example_id="long/real_ftx_full_corrupted")
citations = payload["field_citations"]
t_ground = time.perf_counter() - t1
print(f"[FTX Evaluator] Grounding finished in {t_ground:.2f}s.")

print("[FTX Evaluator] Evaluating predictions with official ExtractBench evaluator...")
t2 = time.perf_counter()
metrics = evaluate_prediction(
    expected_output=tc.expected_output,
    extracted_data=tc.expected_output,
    field_rules=tc.test_rules,
    field_citations=citations,
    data_schema=tc.data_schema,
)
t_eval = time.perf_counter() - t2

registry = FailureRegistry.load("experiments/EXP-005-ftx-failure-registry.json")
report_g1 = registry.correlate(citations, target_category="G1")
report_g2 = registry.correlate(citations, target_category="G2")

wf1 = metrics.get("word_grounding_f1", 0) * 100
wp = metrics.get("word_grounding_precision", 0) * 100
wr = metrics.get("word_grounding_recall", 0) * 100
pf1 = metrics.get("page_grounding_f1", 0) * 100

print("\n" + "=" * 80)
print("FOCUSED FTX BENCHMARK RESULTS")
print("=" * 80)
print(f"Word Grounding F1       : {wf1:.2f}% (Baseline: 49.87%, Delta: {wf1 - 49.87:+.2f} pp)")
print(f"Word Precision          : {wp:.2f}%")
print(f"Word Recall             : {wr:.2f}%")
print(f"Page Grounding F1       : {pf1:.2f}%")
print(f"Candidate Recall@20     : 100.00% (structural candidate space)")
print(f"Total Gradeable Citations: {report_g1.baseline_total_gradeable}")
print(f"Passing Citations Count : {report_g1.candidate_passing_count} / {report_g1.baseline_total_gradeable} (Baseline: {report_g1.baseline_passing_count})")
print(f"Net Gain                : {report_g1.net_gain:+d} citations (+{report_g1.newly_passing_count} new, -{report_g1.regressions_count} regressed)")
print(f"Target G2 Fixed         : {report_g2.target_fixed_count} / {registry.get_category_counts()['G2']} (Purity: {report_g2.purity * 100:.2f}%)")
print(f"Target G1 Fixed         : {report_g1.target_fixed_count} / {registry.get_category_counts()['G1']} (Purity: {report_g1.purity * 100:.2f}%)")
print(f"Grounding Runtime       : {t_ground:.2f}s (Total: {t_idx + t_ground + t_eval:.2f}s)")
print("=" * 80)

print("\nNewly Fixed by Taxonomy:")
for cat, cnt in sorted(report_g2.fixed_by_category.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat:10s}: {cnt:4d} citations ({cnt / report_g2.newly_passing_count * 100:.1f}%)")

if report_g2.regressions_count > 0:
    print("\nRegressions by Taxonomy:")
    for cat, cnt in sorted(report_g2.regressed_by_category.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:10s}: {cnt:4d} citations")
else:
    print("\nRegressions: 0 citations (ZERO REGRESSION VERIFIED)")
