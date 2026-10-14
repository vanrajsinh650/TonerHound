"""EXP-048 Phase A: Full Document Distribution Diagnostic.

Answers:
Which documents are dragging the macro F1 down, and what specifically is failing in each?
1. Loads EXP-047 full benchmark results.
2. Extracts every document's Word F1, Page F1, field count, passing count, failing count.
3. Buckets every evaluated document by Word F1 (10 buckets).
4. Computes potential gain per bucket and sorts descending.
5. Ranks documents by lift_potential = (100 - word_f1) * gradeable_field_count.
6. Identifies tail distribution (<30%, 30-50%, 50-70%, 70-85%, 85%+).
7. Emits distribution.json and lift_ranking.json into research/experiments/EXP-048/audit/.
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "research" / "reference" / "ExtractBench" / "src"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.test_cases.loader import load_test_case

exp047_dir = repo_root / "research" / "experiments" / "EXP-047"
exp048_dir = repo_root / "research" / "experiments" / "EXP-048"
audit_dir = exp048_dir / "audit"
audit_dir.mkdir(parents=True, exist_ok=True)
data_dir = repo_root / "research" / "data" / "full"

def _get_gold_xywh(fp_name: str, ev_boxes: dict[str, Any]) -> list[float] | None:
    if fp_name not in ev_boxes or not ev_boxes[fp_name]:
        return None
    ent = ev_boxes[fp_name][0]
    if isinstance(ent, (list, tuple)):
        if len(ent) == 4 and all(isinstance(v, (int, float)) for v in ent):
            return list(ent)
        if len(ent) == 2 and isinstance(ent[1], (list, tuple)) and len(ent[1]) == 4:
            return list(ent[1])
    return None

def _check_passes(
    cit: dict[str, Any] | None,
    fp_str: str,
    ev_boxes: dict,
    ev_pages: dict,
) -> bool:
    if cit is None:
        return False
    bp = cit.get("page")
    bb = cit.get("bbox") or cit.get("bounding_box", [])
    gp = list(ev_pages.get(fp_str, set()))[0] if fp_str in ev_pages else None
    gb = _get_gold_xywh(fp_str, ev_boxes)
    if bp is None or gp is None or gb is None or not bb:
        return False
    if bp != gp:
        return False
    if len(bb) != 4 or len(gb) != 4:
        return False
    return iou_xywh(bb, gb) >= 0.5


def run_phase_a():
    print("=== EXP-048 Phase A: Starting Full Document Distribution Diagnostic ===", flush=True)
    t0 = time.time()

    # A.1 Load EXP-047 results
    exp047_res_file = exp047_dir / "results" / "full_benchmark_results.json"
    with open(exp047_res_file, encoding="utf-8") as f:
        exp047_results = json.load(f)

    per_doc_base = exp047_results["per_document"]
    eval_docs = [d for d in per_doc_base if d["word_f1"] is not None]
    print(f"Loaded {len(per_doc_base)} documents from EXP-047. Evaluated documents: {len(eval_docs)}", flush=True)

    # Load EXP-047 predictions
    exp047_preds_dir = exp047_dir / "predictions" / "tonerhound"

    # Pre-load known field totals from EXP-045 breakdown to speed up if needed
    exp045_bd_file = repo_root / "research" / "observer" / "reports" / "EXP-045" / "document_breakdown.json"
    exp045_bd = {}
    if exp045_bd_file.exists():
        with open(exp045_bd_file, encoding="utf-8") as f:
            exp045_bd = json.load(f).get("documents", {})

    doc_diagnostics = []
    total_gradeable = 0
    total_passing = 0
    total_failing = 0

    print("Auditing all 236 evaluated documents...", flush=True)
    for idx, d in enumerate(eval_docs):
        tid = d["test_id"]
        word_f1 = d["word_f1"]
        page_f1 = d["page_f1"]
        pdf_path = data_dir / f"{tid}.pdf"
        pred_path = exp047_preds_dir / f"{tid}.result.json"

        # Determine gradeable field count, passing count, failing count
        if not pdf_path.exists() or not pred_path.exists():
            # Fallback to estimates if file missing
            fields = exp045_bd.get(tid, {}).get("total_fields", 0)
            passing = int(round(fields * (word_f1 / 100.0)))
            failing = fields - passing
        else:
            try:
                tc = load_test_case(pdf_path)
                rules = tc.get_extract_field_rules()
                alt_values, ev_boxes, ev_pages, _ = build_rule_indexes(rules)

                gradeable_rules = [r for r in rules if r.field_path in ev_boxes and ev_boxes[r.field_path]]
                fields = len(gradeable_rules)

                with open(pred_path, encoding="utf-8") as pf:
                    pred_data = json.load(pf)
                cits = {
                    c["field_path"]: c
                    for c in pred_data.get("output", {}).get("field_citations", [])
                    if c.get("field_path") and c.get("page") is not None
                }

                passing = 0
                for r in gradeable_rules:
                    fp = r.field_path
                    cit = cits.get(fp)
                    if _check_passes(cit, fp, ev_boxes, ev_pages):
                        passing += 1
                failing = fields - passing
            except Exception as e:
                print(f"Warning: error auditing {tid}: {e}", flush=True)
                fields = exp045_bd.get(tid, {}).get("total_fields", 0)
                passing = int(round(fields * (word_f1 / 100.0)))
                failing = fields - passing

        total_gradeable += fields
        total_passing += passing
        total_failing += failing

        lift_potential = (100.0 - word_f1) * fields

        doc_record = {
            "test_id": tid,
            "word_f1": word_f1,
            "page_f1": page_f1,
            "precision": d.get("precision"),
            "recall": d.get("recall"),
            "field_count": fields,
            "passing_count": passing,
            "failing_count": failing,
            "lift_potential": round(lift_potential, 4),
            "macro_weight_pct": round(100.0 / len(eval_docs), 4),
            "macro_potential_gain_pp": round((100.0 - word_f1) / len(eval_docs), 4),
        }
        doc_diagnostics.append(doc_record)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(eval_docs):
            print(f"  Processed {idx + 1}/{len(eval_docs)} documents ({time.time() - t0:.1f}s elapsed)", flush=True)

    print(f"\nAudit complete in {time.time() - t0:.2f}s.", flush=True)
    print(f"Total gradeable fields audited: {total_gradeable}")
    print(f"Total passing fields: {total_passing}")
    print(f"Total failing fields: {total_failing}")

    # A.2 Produce full distribution across 10 buckets
    # Bucket definitions
    buckets_spec = [
        {"bucket": 0, "name": "Bucket 0: F1 < 10%", "min_f1": 0.0, "max_f1": 10.0},
        {"bucket": 1, "name": "Bucket 1: F1 10-20%", "min_f1": 10.0, "max_f1": 20.0},
        {"bucket": 2, "name": "Bucket 2: F1 20-30%", "min_f1": 20.0, "max_f1": 30.0},
        {"bucket": 3, "name": "Bucket 3: F1 30-40%", "min_f1": 30.0, "max_f1": 40.0},
        {"bucket": 4, "name": "Bucket 4: F1 40-50%", "min_f1": 40.0, "max_f1": 50.0},
        {"bucket": 5, "name": "Bucket 5: F1 50-60%", "min_f1": 50.0, "max_f1": 60.0},
        {"bucket": 6, "name": "Bucket 6: F1 60-70%", "min_f1": 60.0, "max_f1": 70.0},
        {"bucket": 7, "name": "Bucket 7: F1 70-80%", "min_f1": 70.0, "max_f1": 80.0},
        {"bucket": 8, "name": "Bucket 8: F1 80-90%", "min_f1": 80.0, "max_f1": 90.0},
        {"bucket": 9, "name": "Bucket 9: F1 90-100%", "min_f1": 90.0, "max_f1": 100.0001},
    ]

    bucket_data = []
    for b in buckets_spec:
        docs_in_b = [
            d for d in doc_diagnostics
            if b["min_f1"] <= d["word_f1"] < b["max_f1"]
            or (b["bucket"] == 9 and d["word_f1"] == 100.0)
        ]
        n_docs = len(docs_in_b)
        tot_fields = sum(d["field_count"] for d in docs_in_b)
        tot_passing = sum(d["passing_count"] for d in docs_in_b)
        tot_failing = sum(d["failing_count"] for d in docs_in_b)
        avg_f1 = sum(d["word_f1"] for d in docs_in_b) / n_docs if n_docs > 0 else 0.0
        potential_gain = sum((100.0 - d["word_f1"]) * d["field_count"] for d in docs_in_b)
        macro_potential_pp = sum(100.0 - d["word_f1"] for d in docs_in_b) / len(eval_docs)

        bucket_data.append({
            "bucket_id": b["bucket"],
            "bucket_name": b["name"],
            "document_count": n_docs,
            "total_gradeable_fields": tot_fields,
            "passing_fields": tot_passing,
            "failing_fields": tot_failing,
            "average_f1": round(avg_f1, 4),
            "potential_gain_field_pts": round(potential_gain, 2),
            "macro_potential_gain_pp": round(macro_potential_pp, 4),
            "document_ids": [d["test_id"] for d in docs_in_b],
        })

    # Sort buckets by potential gain descending
    bucket_data_sorted = sorted(bucket_data, key=lambda b: -b["potential_gain_field_pts"])

    distribution_payload = {
        "experiment_id": "EXP-048",
        "description": "Full document Word F1 distribution across all 236 evaluated documents",
        "total_documents_evaluated": len(eval_docs),
        "total_gradeable_fields": total_gradeable,
        "total_passing_fields": total_passing,
        "total_failing_fields": total_failing,
        "macro_word_f1": round(sum(d["word_f1"] for d in eval_docs) / len(eval_docs), 4),
        "macro_page_f1": round(sum(d["page_f1"] for d in eval_docs) / len(eval_docs), 4),
        "buckets_sorted_by_potential_gain": bucket_data_sorted,
        "buckets_natural_order": bucket_data,
    }

    dist_file = audit_dir / "distribution.json"
    with open(dist_file, "w", encoding="utf-8") as f:
        json.dump(distribution_payload, f, indent=2)
    print(f"Saved distribution to {dist_file}", flush=True)

    # A.3 Rank documents by lift potential descending
    lift_ranking = sorted(doc_diagnostics, key=lambda d: -d["lift_potential"])
    lift_ranking_payload = {
        "experiment_id": "EXP-048",
        "total_documents": len(lift_ranking),
        "top_documents_by_lift_potential": lift_ranking,
    }

    lift_file = audit_dir / "lift_ranking.json"
    with open(lift_file, "w", encoding="utf-8") as f:
        json.dump(lift_ranking_payload, f, indent=2)
    print(f"Saved lift ranking to {lift_file}", flush=True)

    # A.4 Identify the tail
    tail_lt_30 = [d for d in doc_diagnostics if d["word_f1"] < 30.0]
    tail_30_50 = [d for d in doc_diagnostics if 30.0 <= d["word_f1"] < 50.0]
    tail_50_70 = [d for d in doc_diagnostics if 50.0 <= d["word_f1"] < 70.0]
    tail_70_85 = [d for d in doc_diagnostics if 70.0 <= d["word_f1"] < 85.0]
    tail_85_plus = [d for d in doc_diagnostics if d["word_f1"] >= 85.0]

    macro_lt_30 = sum(100.0 - d["word_f1"] for d in tail_lt_30) / len(eval_docs)
    macro_30_50 = sum(100.0 - d["word_f1"] for d in tail_30_50) / len(eval_docs)
    macro_50_70 = sum(100.0 - d["word_f1"] for d in tail_50_70) / len(eval_docs)
    macro_70_85 = sum(100.0 - d["word_f1"] for d in tail_70_85) / len(eval_docs)
    macro_85_plus = sum(100.0 - d["word_f1"] for d in tail_85_plus) / len(eval_docs)

    tail_summary = {
        "documents_f1_lt_30": {
            "count": len(tail_lt_30),
            "macro_lift_potential_pp": round(macro_lt_30, 4),
            "fields": sum(d["field_count"] for d in tail_lt_30),
        },
        "documents_f1_30_50": {
            "count": len(tail_30_50),
            "macro_lift_potential_pp": round(macro_30_50, 4),
            "fields": sum(d["field_count"] for d in tail_30_50),
        },
        "documents_f1_50_70": {
            "count": len(tail_50_70),
            "macro_lift_potential_pp": round(macro_50_70, 4),
            "fields": sum(d["field_count"] for d in tail_50_70),
        },
        "documents_f1_70_85": {
            "count": len(tail_70_85),
            "macro_lift_potential_pp": round(macro_70_85, 4),
            "fields": sum(d["field_count"] for d in tail_70_85),
        },
        "documents_f1_85_plus": {
            "count": len(tail_85_plus),
            "macro_lift_potential_pp": round(macro_85_plus, 4),
            "fields": sum(d["field_count"] for d in tail_85_plus),
        },
    }

    print("\n--- Phase A Summary Report ---", flush=True)
    print(f"Total evaluated documents: {len(eval_docs)}")
    print(f"F1 < 30%: {len(tail_lt_30)} docs ({macro_lt_30:.4f} pp lift potential)")
    print(f"F1 30-50%: {len(tail_30_50)} docs ({macro_30_50:.4f} pp lift potential)")
    print(f"F1 50-70%: {len(tail_50_70)} docs ({macro_50_70:.4f} pp lift potential)")
    print(f"F1 70-85%: {len(tail_70_85)} docs ({macro_70_85:.4f} pp lift potential)")
    print(f"F1 85%+: {len(tail_85_plus)} docs ({macro_85_plus:.4f} pp lift potential)")
    print(f"Total potential of bottom 20% (< 50% F1): {macro_lt_30 + macro_30_50:.4f} pp")

    print("\nTop 10 Documents by Lift Potential:")
    for rank, doc in enumerate(lift_ranking[:10], 1):
        print(f"  {rank}. {doc['test_id']}: F1={doc['word_f1']:.2f}%, fields={doc['field_count']}, lift={doc['lift_potential']:.1f}")

    return {
        "distribution": distribution_payload,
        "lift_ranking": lift_ranking_payload,
        "tail_summary": tail_summary,
    }


if __name__ == "__main__":
    run_phase_a()
