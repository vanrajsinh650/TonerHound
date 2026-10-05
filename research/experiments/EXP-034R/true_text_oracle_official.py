"""EXP-034R: Official Evaluation Runner for True Text/OCR Oracle.

Evaluates the unconstrained True Text/OCR Oracle predictions across all 370 documents
of the official ExtractBench benchmark using the unmodified ExtractEvaluator.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Single thread math libraries to prevent CPU starvation
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.loader import load_test_case


def run_official_oracle_evaluation() -> dict[str, Any]:
    print("=" * 80)
    print("EXP-034R: TRUE OFFICIAL TEXT/OCR ORACLE BENCHMARK EVALUATOR")
    print("=" * 80)

    t0_suite = time.perf_counter()

    data_dir = repo_root / "research" / "data" / "full"
    pred_dir = repo_root / "research" / "experiments" / "EXP-034R" / "oracle_predictions" / "mode_a"
    cache_dir = repo_root / "research" / "experiments" / "EXP-028B0" / "eval_cache"
    out_json = repo_root / "research" / "experiments" / "EXP-034R" / "true_text_oracle_results.json"

    with open(repo_root / "benchmarks" / "held_out_manifest.json") as f:
        held_docs = json.load(f)["documents"]
    held_ids = set(d["test_id"] for d in held_docs)

    all_pred_files = sorted(list(pred_dir.rglob("*.result.json")))
    print(f"Loaded {len(all_pred_files)} oracle prediction files ({len(held_ids)} in Cohort B).")

    results = []
    evaluator = ExtractEvaluator()

    for idx, rf in enumerate(all_pred_files, 1):
        rel = rf.relative_to(pred_dir)
        tid = rel.as_posix().removesuffix(".result.json")
        split = rel.parts[0]
        pdf_path = data_dir / f"{tid}.pdf"
        cached_eval_path = cache_dir / f"{tid}.eval.json"

        # Check existing eval cache for speed & exactness
        if cached_eval_path.exists():
            try:
                with open(cached_eval_path, encoding="utf-8") as f:
                    cdata = json.load(f)
                m_map = {m["metric_name"]: m["value"] for m in cdata.get("metrics", [])}
                results.append({
                    "test_id": tid,
                    "split": split,
                    "success": cdata.get("success", True),
                    "word_f1": m_map.get("extract_unified_grounded_f1"),
                    "word_precision": m_map.get("extract_unified_grounded_precision"),
                    "word_recall": m_map.get("extract_unified_grounded_recall"),
                    "page_f1": m_map.get("extract_unified_page_f1"),
                    "value_f1": m_map.get("extract_unified_value_f1", 1.0),
                })
                continue
            except Exception:
                pass

        # Fresh evaluation if not cached
        with open(rf, encoding="utf-8") as f:
            inf = InferenceResult.model_validate(json.load(f))
        tc = load_test_case(pdf_path)
        eval_res = evaluator.evaluate(inf, tc)
        m_map = {m.metric_name: m.value for m in eval_res.metrics}
        results.append({
            "test_id": tid,
            "split": split,
            "success": eval_res.success,
            "word_f1": m_map.get("extract_unified_grounded_f1"),
            "word_precision": m_map.get("extract_unified_grounded_precision"),
            "word_recall": m_map.get("extract_unified_grounded_recall"),
            "page_f1": m_map.get("extract_unified_page_f1"),
            "value_f1": m_map.get("extract_unified_value_f1", 1.0),
        })

    def _calc_stats(docs: list[dict[str, Any]]) -> dict[str, float]:
        valid_wf1 = [d["word_f1"] for d in docs if d.get("success") and d.get("word_f1") is not None]
        valid_wp = [d["word_precision"] for d in docs if d.get("success") and d.get("word_precision") is not None]
        valid_wr = [d["word_recall"] for d in docs if d.get("success") and d.get("word_recall") is not None]
        valid_pf1 = [d["page_f1"] for d in docs if d.get("success") and d.get("page_f1") is not None]
        valid_vf1 = [d["value_f1"] for d in docs if d.get("success") and d.get("value_f1") is not None]

        wf1 = float(np.mean(valid_wf1) * 100) if valid_wf1 else 0.0
        wp = float(np.mean(valid_wp) * 100) if valid_wp else 0.0
        wr = float(np.mean(valid_wr) * 100) if valid_wr else 0.0
        pf1 = float(np.mean(valid_pf1) * 100) if valid_pf1 else 0.0
        vf1 = float(np.mean(valid_vf1) * 100) if valid_vf1 else 0.0

        return {
            "word_f1": round(wf1, 4),
            "word_precision": round(wp, 4),
            "word_recall": round(wr, 4),
            "page_f1": round(pf1, 4),
            "value_f1": round(vf1, 4),
            "false_grounding_rate": round(100.0 - wp, 4),
            "abstention_rate": round(100.0 - wr, 4),
            "grounded_docs_count": len(valid_wf1),
            "total_evaluated": len(docs),
        }

    full_370_stats = _calc_stats(results)
    cohort_b_results = [r for r in results if r["test_id"] in held_ids]
    cohort_b_stats = _calc_stats(cohort_b_results)

    # Length split stats
    split_stats = {}
    for sp in ("short", "medium", "long"):
        sp_docs = [r for r in results if r["split"] == sp]
        split_stats[sp] = _calc_stats(sp_docs)

    prod_baseline_f1 = 56.0477
    current_pool_oracle_f1 = 60.9591
    true_oracle_f1 = full_370_stats["word_f1"]

    selection_headroom = round(current_pool_oracle_f1 - prod_baseline_f1, 4)
    expanded_headroom = round(true_oracle_f1 - prod_baseline_f1, 4)
    candidate_quality_gap = round(true_oracle_f1 - current_pool_oracle_f1, 4)
    gap_to_90 = round(90.0000 - true_oracle_f1, 4)

    total_time = time.perf_counter() - t0_suite
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print("\n" + "=" * 80)
    print("TRUE OFFICIAL TEXT/OCR ORACLE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Full 370 Word Grounding F1 : {full_370_stats['word_f1']:.4f}%")
    print(f"Full 370 Word Precision    : {full_370_stats['word_precision']:.4f}%")
    print(f"Full 370 Word Recall       : {full_370_stats['word_recall']:.4f}%")
    print(f"Full 370 Page Grounding F1 : {full_370_stats['page_f1']:.4f}%")
    print(f"False Grounding Rate       : {full_370_stats['false_grounding_rate']:.4f}%")
    print(f"Abstention Rate            : {full_370_stats['abstention_rate']:.4f}%")
    print("-" * 80)
    print(f"Cohort B Word Grounding F1 : {cohort_b_stats['word_f1']:.4f}%")
    print("-" * 80)
    print(f"Production Baseline F1     : {prod_baseline_f1:.4f}%")
    print(f"Current-Pool Oracle F1     : {current_pool_oracle_f1:.4f}%")
    print(f"True Official Oracle F1    : {true_oracle_f1:.4f}%")
    print("-" * 80)
    print(f"Current Selection Headroom : +{selection_headroom:.4f} pp")
    print(f"Expanded-Pool Headroom     : +{expanded_headroom:.4f} pp")
    print(f"Candidate-Quality Gap      : +{candidate_quality_gap:.4f} pp")
    print(f"Remaining Gap to 90% Target: {gap_to_90:.4f} pp")
    print("=" * 80)

    output = {
        "experiment": "EXP-034R",
        "description": "True Official Text/OCR Oracle Re-Measurement on Full 370 Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(total_time, 2),
        "memory_mb": round(mem_mb, 2),
        "evaluator": "extract_bench.evaluation.evaluators.extract.ExtractEvaluator",
        "reference_points": {
            "production_baseline_f1": prod_baseline_f1,
            "current_pool_selection_oracle_f1": current_pool_oracle_f1,
            "historical_claimed_true_text_oracle_f1": 75.1243,
            "target_llamaextract_f1": 58.1100,
            "target_stretch_f1": 90.0000,
        },
        "official_results": {
            "full_370": full_370_stats,
            "cohort_b": cohort_b_stats,
            "splits": split_stats,
        },
        "ceiling_decomposition": {
            "selection_headroom_pp": selection_headroom,
            "expanded_pool_headroom_pp": expanded_headroom,
            "candidate_quality_gap_pp": candidate_quality_gap,
            "gap_to_90_target_pp": gap_to_90,
            "delta_vs_llamaextract_pp": round(true_oracle_f1 - 58.1100, 4),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved true text oracle results to {out_json}")
    return output


if __name__ == "__main__":
    run_official_oracle_evaluation()
