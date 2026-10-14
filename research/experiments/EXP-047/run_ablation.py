"""EXP-047 Phase H: Ablation Study.

Measures contribution of each EXP-047 component by running the held-out gate
with each component disabled in turn.

Ablation configurations:
1. Full:                        char=True,  numeric=True,  table=True
2. Without char_index:          char=False, numeric=True,  table=True
3. Without numeric_index:       char=True,  numeric=False, table=True
4. Without table_localizer:     char=True,  numeric=True,  table=False
5. Without global_assigner:     N/A (field-by-field baseline = EXP-046)

Output: research/experiments/EXP-047/ablation.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_repo_root = Path(__file__).resolve().parents[3]
_exp_dir = Path(__file__).resolve().parent

for _p in [str(_repo_root), str(_repo_root / "src"), str(_exp_dir / "matching")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ref_eb = _repo_root / "research" / "reference" / "ExtractBench" / "src"
if _ref_eb.exists() and str(_ref_eb) not in sys.path:
    sys.path.insert(0, str(_ref_eb))

from run_heldout_gate import run_heldout_gate


def run_ablation() -> dict[str, Any]:
    """Run all ablation configurations on the held-out cohort."""
    print("=== EXP-047 Phase H: Ablation Study ===", flush=True)
    t_start = time.perf_counter()

    configs = [
        {
            "name": "full",
            "label": "Full (char + numeric + table + global)",
            "char": True,
            "numeric": True,
            "table": True,
        },
        {
            "name": "no_char",
            "label": "Without char_index",
            "char": False,
            "numeric": True,
            "table": True,
        },
        {
            "name": "no_numeric",
            "label": "Without numeric_index",
            "char": True,
            "numeric": False,
            "table": True,
        },
        {
            "name": "no_table",
            "label": "Without table_localizer",
            "char": True,
            "numeric": True,
            "table": False,
        },
    ]

    results: dict[str, Any] = {
        "experiment_id": "EXP-047",
        "phase": "H_ablation",
        "baseline_exp046_rescued": 32,
        "configurations": [],
    }

    for cfg in configs:
        print(f"\n--- Ablation: {cfg['label']} ---", flush=True)
        if cfg["name"] == "full":
            heldout_res_p = _exp_dir / "results" / "heldout_results.json"
            if heldout_res_p.exists():
                with open(heldout_res_p, "r", encoding="utf-8") as f:
                    r = json.load(f)
                results["configurations"].append({
                    "name": "full",
                    "label": cfg["label"],
                    "rescued": r["rescued_fields"],
                    "regressed": r["regressed_fields"],
                    "net_rescued": r["net_rescued"],
                    "gate_pass": r["gate_pass"],
                    "elapsed_s": r["elapsed_seconds"],
                })
                print("  Reused existing Phase F heldout_results.json.", flush=True)
                continue
        try:
            r = run_heldout_gate(
                build_char=cfg["char"],
                build_numeric=cfg["numeric"],
                build_table=cfg["table"],
                tag=cfg["name"],
            )
            results["configurations"].append({
                "name": cfg["name"],
                "label": cfg["label"],
                "rescued": r["rescued_fields"],
                "regressed": r["regressed_fields"],
                "net_rescued": r["net_rescued"],
                "gate_pass": r["gate_pass"],
                "elapsed_s": r["elapsed_seconds"],
            })
        except Exception as e:
            print(f"  Ablation {cfg['name']} FAILED: {e}", flush=True)
            results["configurations"].append({
                "name": cfg["name"],
                "label": cfg["label"],
                "error": str(e),
                "rescued": 0,
                "regressed": 0,
            })

    # EXP-046 field-by-field baseline (no global assigner)
    results["configurations"].append({
        "name": "no_global_assigner",
        "label": "Without global_assigner (EXP-046 field-by-field)",
        "rescued": 32,  # EXP-046 net rescue count
        "regressed": 0,
        "net_rescued": 32,
        "note": "EXP-046 final result — field-by-field 34-stage pipeline",
    })

    results["total_elapsed_s"] = time.perf_counter() - t_start

    out_path = _exp_dir / "ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation complete. Written: {out_path}", flush=True)
    print("\nResults summary:", flush=True)
    for cfg_res in results["configurations"]:
        print(
            f"  {cfg_res['label']}: rescued={cfg_res.get('rescued', '?')}, "
            f"regressed={cfg_res.get('regressed', '?')}, "
            f"net={cfg_res.get('net_rescued', '?')}",
            flush=True,
        )

    return results


if __name__ == "__main__":
    run_ablation()
