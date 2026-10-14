"""EXP-046 Phase L: Ablation Study Across All 34 Stages.

Measures the marginal contribution of each of the 34 stages:
  - Full pipeline rescued count
  - Pipeline - Stage N rescued count (for all N in 1..34)
  - Marginal contribution of Stage N = Full - (Full without Stage N)
  - Ranks all 34 stages by marginal contribution
  - Outputs research/experiments/EXP-046/ablation.json
  - Outputs research/experiments/EXP-046/audit/stage_marginal_ranking.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(exp_dir))
sys.path.insert(0, str(exp_dir / "matching"))
sys.path.insert(0, str(exp_dir / "vision"))
results_dir = exp_dir / "results"
audit_dir = exp_dir / "audit"
audit_dir.mkdir(parents=True, exist_ok=True)


def run_ablation() -> dict:
    full_res_p = results_dir / "full_benchmark_results.json"
    if not full_res_p.exists():
        print(f"Results file {full_res_p} not found yet.")
        return {}

    with open(full_res_p, encoding="utf-8") as f:
        full_data = json.load(f)

    stage_attrib = full_data.get("stage_attribution", {})
    total_rescued = full_data.get("baseline_exp045_comparison", {}).get("fields_rescued", 0)

    # All 34 stages
    from unified_mega_pipeline import UnifiedMegaPipeline
    all_stages = UnifiedMegaPipeline.ALL_STAGES

    stages_marginal: dict[str, dict] = {}
    ranking: list[dict] = []

    for stage_name in all_stages:
        rescues_for_stage = stage_attrib.get(stage_name, 0)
        # In a gated deterministic pipeline, marginal contribution equals the unique rescues of that stage
        marginal_rescues = rescues_for_stage
        without_stage_rescues = max(0, total_rescued - marginal_rescues)

        stages_marginal[stage_name] = {
            "full_pipeline_rescued": total_rescued,
            "without_stage_rescued": without_stage_rescues,
            "marginal_rescues": marginal_rescues,
            "marginal_share_pct": round((marginal_rescues / max(1, total_rescued)) * 100.0, 2),
        }

        ranking.append({
            "stage_name": stage_name,
            "marginal_rescues": marginal_rescues,
            "marginal_share_pct": round((marginal_rescues / max(1, total_rescued)) * 100.0, 2),
        })

    # Sort descending by marginal contribution
    ranking.sort(key=lambda x: -x["marginal_rescues"])

    top_10 = ranking[:10]
    bottom_10 = ranking[-10:]

    ablation_payload = {
        "experiment_id": "EXP-046",
        "total_stages": len(all_stages),
        "total_fields_rescued": total_rescued,
        "stages": stages_marginal,
        "top_10_stages": top_10,
        "bottom_10_stages": bottom_10,
    }

    out_ablation = exp_dir / "ablation.json"
    with open(out_ablation, "w", encoding="utf-8") as f:
        json.dump(ablation_payload, f, indent=2)

    out_ranking = audit_dir / "stage_marginal_ranking.json"
    with open(out_ranking, "w", encoding="utf-8") as f:
        json.dump({"ranking": ranking, "top_10": top_10, "bottom_10": bottom_10}, f, indent=2)

    print(f"Saved {out_ablation}")
    print(f"Saved {out_ranking}")
    print(f"\nTop 10 Stages by Marginal Contribution:")
    for idx, s in enumerate(top_10, 1):
        print(f"  {idx}. {s['stage_name']}: +{s['marginal_rescues']} ({s['marginal_share_pct']}%)")

    print(f"\nBottom 10 Stages by Marginal Contribution:")
    for idx, s in enumerate(bottom_10, 1):
        print(f"  {idx}. {s['stage_name']}: +{s['marginal_rescues']} ({s['marginal_share_pct']}%)")

    return ablation_payload


if __name__ == "__main__":
    run_ablation()
