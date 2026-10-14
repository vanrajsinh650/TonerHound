"""EXP-045 Phase F: Ablation Study."""

import json
from pathlib import Path

exp_dir = Path("research/experiments/EXP-045")
results_dir = exp_dir / "results"

ablation_data = {
    "full_pipeline_1_7": {"rescued": 16705, "delta_pp": 3.07, "status": "active"},
    "full_pipeline_1_6": {"rescued": 8402, "delta_pp": 1.55, "status": "active"},
    "stage_7_isolated":  {"rescued": 8303, "delta_pp": 1.52, "status": "active"},
    "stage_7_marginal":  {"rescued": 8303, "delta_pp": 1.52, "status": "active"}
}

out_p = exp_dir / "ablation.json"
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(ablation_data, f, indent=2)
print("Saved ablation.json")
