"""EXP-044 Phase D: Ablation Study.

Evaluates the contribution of each stage (Stages 1 through 6) by computing:
- Full pipeline: rescued count & delta Word F1 pp
- Without Stage 1 (Canonicalization)
- Without Stage 2 (Character-level Fuzzy)
- Without Stage 3 (Token Alignment Needleman-Wunsch)
- Without Stage 4 (Hungarian Token Set)
- Without Stage 5 (Multi-Region V3)
- Without Stage 6 (Ambiguity Resolver / Global Assignment)

Outputs: research/experiments/EXP-044/ablation.json
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
exp_dir = Path(__file__).resolve().parent

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.loader import load_test_case

def run_ablation() -> dict[str, dict[str, float]]:
    print("=== EXP-044 Phase D: Ablation Study ===")
    data_dir = repo_root / "research" / "data" / "full"
    base_preds_dir = repo_root / "research" / "experiments" / "EXP-042" / "predictions" / "tonerhound"
    heldout_preds_dir = exp_dir / "results" / "heldout_predictions"

    heldout_res_p = exp_dir / "results" / "heldout_results.json"
    with open(heldout_res_p, encoding="utf-8") as f:
        heldout_res = json.load(f)

    base_f1_macro = heldout_res["baseline_word_f1"]
    full_f1_macro = heldout_res["after_word_f1"]
    full_rescued = heldout_res["net_rescued"]
    full_delta_pp = round(full_f1_macro - base_f1_macro, 4)

    evaluator = ExtractEvaluator()

    # Pre-load base predictions and test cases for affected docs
    docs_info = {d["document_id"]: d for d in heldout_res["per_document"]}
    all_doc_ids = [d["document_id"] for d in heldout_res["per_document"]]

    stage_keys = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5", "stage_6"]
    
    # Direct rescues per stage as established
    stage_direct_rescues = {
        "stage_1": 19,
        "stage_2": 7550,
        "stage_3": 34,
        "stage_4": 0,
        "stage_5": 3,
        "stage_6": 796,
    }

    ablation_results = {
        "full_pipeline": {
            "rescued": full_rescued,
            "delta_pp": round(full_delta_pp, 2),
        }
    }

    # For each stage to ablate, compute rescued count and evaluated delta_pp
    for stage_idx in range(1, 7):
        stage_name = f"stage_{stage_idx}"
        ablate_key = f"without_stage_{stage_idx}"
        
        rescued_without_stage = full_rescued - stage_direct_rescues[stage_name]

        # Evaluate affected docs without this stage
        doc_f1s: list[float] = []
        for doc_id in all_doc_ids:
            doc_meta = docs_info[doc_id]
            b_f1 = doc_meta["base_f1"]
            f_f1 = doc_meta["after_f1"]

            if doc_meta["rescued"] == 0:
                doc_f1s.append(b_f1 if b_f1 is not None else 0.0)
                continue

            # Load full prediction
            fname = doc_id.replace("/", "_") + ".result.json"
            p_pred = heldout_preds_dir / fname
            full_data = json.loads(p_pred.read_text(encoding="utf-8"))
            
            # Check if this doc used the ablated stage
            cits = full_data.get("output", {}).get("field_citations", [])
            has_stage = False
            for c in cits:
                meta_st = c.get("metadata", {}).get("stage", "") if isinstance(c.get("metadata"), dict) else ""
                src = c.get("source", "")
                if stage_name in meta_st or (stage_name == "stage_6" and "global_assignment" in src):
                    has_stage = True
                    break

            if not has_stage:
                doc_f1s.append(f_f1 if f_f1 is not None else 0.0)
                continue

            # Load baseline citations to revert to
            p_base = base_preds_dir / f"{doc_id}.result.json"
            base_data = json.loads(p_base.read_text(encoding="utf-8"))
            base_cits_map = {c["field_path"]: c for c in base_data.get("output", {}).get("field_citations", [])}

            # Filter citations without this stage
            ablated_cits = []
            for c in cits:
                meta_st = c.get("metadata", {}).get("stage", "") if isinstance(c.get("metadata"), dict) else ""
                src = c.get("source", "")
                is_ablated = (stage_name in meta_st) or (stage_name == "stage_6" and "global_assignment" in src)
                if is_ablated:
                    # Revert to base citation if exists
                    fp = c.get("field_path")
                    if fp in base_cits_map:
                        ablated_cits.append(base_cits_map[fp])
                else:
                    ablated_cits.append(c)

            # Re-evaluate
            ablated_pred = copy.deepcopy(full_data)
            ablated_pred["output"]["field_citations"] = ablated_cits
            
            p_pdf = data_dir / f"{doc_id}.pdf"
            tc = load_test_case(p_pdf)
            inf = InferenceResult.model_validate(ablated_pred)
            res = evaluator.evaluate(inf, tc)
            m = {x.metric_name: x.value for x in res.metrics}
            ablated_doc_f1 = (m.get("extract_unified_grounded_f1") or 0.0) * 100.0
            doc_f1s.append(round(ablated_doc_f1, 4))

        mean_ablated_f1 = sum(doc_f1s) / len(doc_f1s)
        delta_ablated_pp = round(mean_ablated_f1 - base_f1_macro, 4)

        ablation_results[ablate_key] = {
            "rescued": rescued_without_stage,
            "delta_pp": round(delta_ablated_pp, 2),
        }
        print(f"  {ablate_key}: rescued={rescued_without_stage}, delta_pp={delta_ablated_pp:+.4f} pp")

    out_json = exp_dir / "ablation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nAblation study results written to {out_json}:")
    print(json.dumps(ablation_results, indent=2))
    return ablation_results

if __name__ == "__main__":
    run_ablation()
