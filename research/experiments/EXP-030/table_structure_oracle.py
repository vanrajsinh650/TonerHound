"""EXP-030: Oracle Table Structure & Theoretical Upper Bound.

Tests Phase 2 & 3 hypothesis:
If the correct table structure (ground-truth row page and y-center) is known,
how much selection accuracy can joint row-aware assignment recover from the
current candidate pools?

Generates:
- research/experiments/EXP-030/oracle_comparison.json
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

curr_dir = Path(__file__).resolve().parent
repo_root = curr_dir.parent.parent.parent


def evaluate_oracle_table_structure(cohort_manifest_path: Path, cohort_name: str) -> dict[str, Any]:
    with open(cohort_manifest_path, encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    doc_ids = set(d["test_id"] for d in docs)

    parquet_path = repo_root / "research" / "observer" / "field_records.parquet"
    dataset = ds.dataset(str(parquet_path), format="parquet")
    filter_expr = ds.field("document_id").isin(doc_ids) & (ds.field("candidate_count") >= 2)
    table = dataset.to_table(
        filter=filter_expr,
        columns=[
            "document_id",
            "field_path",
            "gold_value",
            "candidate_pool",
            "gold_evidence_entries",
            "candidate_hit_at_1",
            "best_candidate_iou",
        ],
    )

    records: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for i in range(len(table)):
        doc_id = table["document_id"][i].as_py()
        fpath = table["field_path"][i].as_py()
        m = re.match(r"^(.*?)\[(\d+)\]\.(.*)$", fpath)
        if not m:
            continue
        tname, ridx, subfield = m.group(1), int(m.group(2)), m.group(3)
        cands = json.loads(table["candidate_pool"][i].as_py())
        gold_entries = json.loads(table["gold_evidence_entries"][i].as_py() or "[]")
        hit_1 = bool(table["candidate_hit_at_1"][i].as_py())
        best_iou = float(table["best_candidate_iou"][i].as_py())

        records[(doc_id, tname, ridx)].append({
            "field_path": fpath,
            "subfield": subfield,
            "candidates": cands,
            "gold_entries": gold_entries,
            "baseline_hit": hit_1,
            "best_iou": best_iou,
        })

    tot_fields = sum(len(flds) for flds in records.values())
    base_hits = sum(sum(1 for f in flds if f["baseline_hit"]) for flds in records.values())
    max_possible = sum(sum(1 for f in flds if f["best_iou"] >= 0.50) for flds in records.values())

    oracle_hits = 0
    oracle_flips_beneficial = 0
    oracle_flips_harmful = 0

    for (doc_id, tname, ridx), flds in records.items():
        gold_ys: list[float] = []
        gold_pages: list[int] = []
        for f in flds:
            for ev in f["gold_entries"]:
                p = ev.get("page")
                b = ev.get("bbox")
                if p is not None and b:
                    gold_pages.append(p)
                    gold_ys.append(b[1] + b[3] / 2.0)

        if not gold_ys:
            for f in flds:
                if f["baseline_hit"]:
                    oracle_hits += 1
            continue

        oracle_p = gold_pages[0]
        oracle_y = float(np.median(gold_ys))

        for f in flds:
            b_hit = f["baseline_hit"]
            cands = f["candidates"]
            same_page = [c for c in cands if c.get("page") == oracle_p]
            if not same_page:
                if b_hit:
                    oracle_hits += 1
                continue

            best_c = min(same_page, key=lambda c: abs((c.get("bbox")[1] + c.get("bbox")[3] / 2.0) - oracle_y))
            r_hit = bool(best_c.get("best_iou", 0.0) >= 0.50)
            if r_hit:
                oracle_hits += 1
            if not b_hit and r_hit:
                oracle_flips_beneficial += 1
            elif b_hit and not r_hit:
                oracle_flips_harmful += 1

    gap = max_possible - base_hits
    closed = oracle_hits - base_hits

    return {
        "cohort": cohort_name,
        "total_records": len(records),
        "total_multi_cand_fields": tot_fields,
        "baseline_hit_count": base_hits,
        "baseline_hit_rate": round(base_hits / tot_fields * 100, 2),
        "ceiling_a_recall_count": max_possible,
        "ceiling_a_recall_rate": round(max_possible / tot_fields * 100, 2),
        "selection_gap_pp": round((max_possible - base_hits) / tot_fields * 100, 2),
        "oracle_hit_count": oracle_hits,
        "oracle_hit_rate": round(oracle_hits / tot_fields * 100, 2),
        "oracle_gain_pp": round((oracle_hits - base_hits) / tot_fields * 100, 2),
        "selection_gap_closed_pct": round(closed / max(1, gap) * 100, 2),
        "beneficial_flips": oracle_flips_beneficial,
        "harmful_flips": oracle_flips_harmful,
        "net_flips": oracle_flips_beneficial - oracle_flips_harmful,
    }


def main() -> None:
    print("Evaluating Oracle Table Structure on Cohort A (Dev)...")
    res_a = evaluate_oracle_table_structure(
        repo_root / "benchmarks" / "exp005_local_manifest.json", "Cohort A (Development)"
    )
    print(f"Cohort A Oracle Hit@1: {res_a['oracle_hit_rate']}% (Baseline: {res_a['baseline_hit_rate']}%, Gain: {res_a['oracle_gain_pp']:+.2f}pp)")

    print("Evaluating Oracle Table Structure on Cohort B (Held-Out)...")
    res_b = evaluate_oracle_table_structure(
        repo_root / "benchmarks" / "held_out_manifest.json", "Cohort B (Held-Out)"
    )
    print(f"Cohort B Oracle Hit@1: {res_b['oracle_hit_rate']}% (Baseline: {res_b['baseline_hit_rate']}%, Gain: {res_b['oracle_gain_pp']:+.2f}pp)")

    out = {
        "experiment": "EXP-030",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Oracle Table Structure Theoretical Reachability Benchmark",
        "cohort_a_development": res_a,
        "cohort_b_held_out": res_b,
    }

    out_file = curr_dir / "oracle_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
