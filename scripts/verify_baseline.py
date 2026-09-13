"""Verify metrics for Step 1 of EXP-003:
1. Re-run EXP-002E (or verify exact case-by-case outputs)
2. Verify evaluator formulas and aggregation
3. Check harmonic mean invariant for each doc and overall
4. Record commit and exact setup
"""

import json
import math
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

from tonerhound.benchmark.runner import run_benchmark_suite, LLAMAEXTRACT_AGENTIC_PLUS_WORD_F1

data_dir = root_dir / "research" / "data" / "test"
exp_dir = root_dir / "research" / "experiments"

print("--- Verifying EXP-002E Saved Artifact ---")
with open(exp_dir / "EXP-002E.json", "r") as f:
    exp002e = json.load(f)

print(f"Timestamp: {exp002e['timestamp']}")
print(f"Avg Word Grounding F1: {exp002e['avg_word_grounding_f1'] * 100:.4f}%")
print(f"Avg Word Precision: {exp002e['avg_word_grounding_precision'] * 100:.4f}%")
print(f"Avg Word Recall: {exp002e['avg_word_grounding_recall'] * 100:.4f}%")
print(f"Avg Page Grounding F1: {exp002e['avg_page_grounding_f1'] * 100:.4f}%")
print(f"Avg False Grounding Rate: {exp002e['avg_false_grounding_rate'] * 100:.4f}%")
print(f"Avg Ambiguity Rate: {exp002e['avg_ambiguity_rate'] * 100:.4f}%")

print("\n--- Per-Case Verification ---")
f1_list = []
prec_list = []
rec_list = []
fg_list = []
amb_list = []

for c in exp002e["case_metrics"]:
    w_f1 = c["word_f1"]
    w_p = c["word_precision"]
    w_r = c["word_recall"]
    fg = c["false_grounding_rate"]
    amb = c["ambiguity_rate"]
    if w_f1 is not None:
        expected_f1 = 2 * w_p * w_r / (w_p + w_r) if (w_p + w_r) > 0 else 0.0
        diff = abs(w_f1 - expected_f1)
        print(f"Case {c['test_id']}:")
        print(f"  P = {w_p:.6f}, R = {w_r:.6f}, F1 = {w_f1:.6f}, Computed Harmonic Mean = {expected_f1:.6f} (diff = {diff:.2e})")
        print(f"  False Grounding = {fg:.6f} (1 - P = {1.0 - w_p:.6f})")
        print(f"  Ambiguity Rate = {amb:.6f}")
        f1_list.append(w_f1)
        prec_list.append(w_p)
        rec_list.append(w_r)
        fg_list.append(fg)
        amb_list.append(amb)
    else:
        print(f"Case {c['test_id']}: No GT bounding boxes (word_f1 is None)")

macro_f1 = sum(f1_list) / len(f1_list)
macro_prec = sum(prec_list) / len(prec_list)
macro_rec = sum(rec_list) / len(rec_list)
macro_fg = sum(fg_list) / len(fg_list)
macro_amb = sum(amb_list) / len(amb_list)

print("\n--- Aggregation Confirmation ---")
print(f"Macro F1 over {len(f1_list)} valid docs: {macro_f1:.6f} (Matches reported: {abs(macro_f1 - exp002e['avg_word_grounding_f1']) < 1e-6})")
print(f"Macro Precision: {macro_prec:.6f} (Matches reported: {abs(macro_prec - exp002e['avg_word_grounding_precision']) < 1e-6})")
print(f"Macro Recall: {macro_rec:.6f} (Matches reported: {abs(macro_rec - exp002e['avg_word_grounding_recall']) < 1e-6})")
print(f"Macro False Grounding: {macro_fg:.6f} (Matches reported: {abs(macro_fg - exp002e['avg_false_grounding_rate']) < 1e-6})")
print(f"Macro Ambiguity: {macro_amb:.6f} (Matches reported: {abs(macro_amb - exp002e['avg_ambiguity_rate']) < 1e-6})")
