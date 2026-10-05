"""EXP-035: Exact Macro-Weighted Opportunity and Contribution Breakdown.

Computes the official document-macro Word Grounding F1 contribution of each
reclassified failure mechanism, using the official 236 grounded document denominator.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent

with open(repo_root / "research" / "experiments" / "EXP-035" / "failure_audit.json") as f:
    audit_cases = json.load(f)["cases"]

eval_cache_cur = repo_root / "research" / "experiments" / "EXP-032" / "eval_cache" / "mode_a"
eval_cache_exp = repo_root / "research" / "experiments" / "EXP-028B0" / "eval_cache"

# Load doc F1 deltas
doc_f1_deltas: dict[str, float] = {}
for cf in eval_cache_cur.rglob("*.eval.json"):
    rel = cf.relative_to(eval_cache_cur)
    tid = rel.as_posix().removesuffix(".eval.json")
    ef = eval_cache_exp / rel
    if ef.exists():
        with open(cf) as f1, open(ef) as f2:
            d1, d2 = json.load(f1), json.load(f2)
        f1_c = d1.get("word_f1")
        if f1_c is None:
            m1 = {m["metric_name"]: m["value"] for m in d1.get("metrics", [])}
            f1_c = m1.get("extract_unified_grounded_f1")

        f1_e = d2.get("word_f1")
        if f1_e is None:
            m2 = {m["metric_name"]: m["value"] for m in d2.get("metrics", [])}
            f1_e = m2.get("extract_unified_grounded_f1")

        if f1_c is not None and f1_e is not None:
            doc_f1_deltas[tid] = max(0.0, (f1_e - f1_c) * 100)

grounded_doc_count = 236

# Group audited cases by doc and class
doc_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
doc_total_alleged: dict[str, int] = defaultdict(int)
class_fields: dict[str, int] = defaultdict(int)
class_docs: dict[str, set[str]] = defaultdict(set)

for c in audit_cases:
    doc_id = c["document_id"]
    cls = c["failure_mechanism"]
    doc_class_counts[doc_id][cls] += 1
    doc_total_alleged[doc_id] += 1
    class_fields[cls] += 1
    class_docs[cls].add(doc_id)

total_alleged_pp = 3.4991
class_macro_pp: dict[str, float] = defaultdict(float)

# In EXP-034R, each document contributed delta_pp * (cnt / tot_gap_doc) / 236.
# Among the alleged 572 cases in that document:
# The portion of the document's contribution corresponding to each mechanism:
for doc_id, cls_counts in doc_class_counts.items():
    delta_pp = doc_f1_deltas.get(doc_id, 0.0)
    tot_alleged = doc_total_alleged[doc_id]
    for cls, cnt in cls_counts.items():
        class_macro_pp[cls] += (delta_pp / grounded_doc_count) * (cnt / max(1, tot_alleged))

tot_unnorm = sum(class_macro_pp.values())

output_data = []
for cls in sorted(class_fields.keys(), key=lambda c: class_fields[c], reverse=True):
    norm_pp = round((class_macro_pp[cls] / tot_unnorm) * total_alleged_pp, 4) if tot_unnorm > 0 else 0.0
    pct = round((norm_pp / total_alleged_pp) * 100, 2)
    output_data.append({
        "failure_mechanism": cls,
        "fields_affected": class_fields[cls],
        "documents_affected": len(class_docs[cls]),
        "macro_weighted_f1_contribution_pp": norm_pp,
        "percentage_of_alleged_gap": pct,
    })

# Save breakdown
out_file = repo_root / "research" / "experiments" / "EXP-035" / "macro_weighted_breakdown.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "total_alleged_fields": len(audit_cases),
        "total_alleged_macro_pp": total_alleged_pp,
        "grounded_benchmark_documents": grounded_doc_count,
        "mechanisms": output_data,
    }, f, indent=2)

print("=" * 80)
print("AUDITED MACRO-WEIGHTED OPPORTUNITY BREAKDOWN (EXP-035)")
print("=" * 80)
print(f"{'Failure Mechanism':<26} | {'Fields':<8} | {'Docs':<6} | {'Macro F1 (pp)':<14} | {'% of Alleged':<14}")
print("-" * 80)
for row in output_data:
    print(f"{row['failure_mechanism']:<26} | {row['fields_affected']:>8} | {row['documents_affected']:<6} | +{row['macro_weighted_f1_contribution_pp']:<13.4f} | {row['percentage_of_alleged_gap']:>13.2f}%")
print("=" * 80)
print(f"Saved to {out_file}")
