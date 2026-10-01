"""EXP-029: Complete Pipeline Execution & Held-Out Simulation.

Executes Phases 4, 5, 6, and Decision Gate:
1. Trains Logistic Regression candidate reranker on Cohort A (32 docs).
2. Computes ROC AUC, PR AUC, calibration curve, Brier score, coefficients, source-level errors.
3. Evaluates selective conservatism thresholds (tau) on Cohort A.
4. Freezes optimal policy and evaluates on Held-Out Cohort B (32 docs).
5. Computes multi-candidate Hit@1, beneficial/harmful flips, subgroup breakdowns.
6. Evaluates official ExtractBench Word Grounding F1, Precision, Recall, False Grounding on Cohort B.
7. Saves all artifacts: training_results.json, feature_importance.json, heldout_results.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Ensure local imports
curr_dir = Path(__file__).resolve().parent
repo_root = curr_dir.parent.parent.parent
sys.path.insert(0, str(curr_dir))
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from feature_extraction import FEATURE_NAMES, extract_candidate_features_and_labels

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.loader import load_test_case


def _eval_single_doc(task: dict[str, str]) -> dict[str, Any]:
    test_id = task["test_id"]
    pdf_path = Path(task["pdf_path"])
    res_path = Path(task["res_path"])
    try:
        with open(res_path, encoding="utf-8") as f:
            inf = InferenceResult.model_validate(json.load(f))
        tc = load_test_case(pdf_path)
        evaluator = ExtractEvaluator()
        res = evaluator.evaluate(inf, tc)
        m_map = {m.metric_name: m.value for m in res.metrics}
        return {
            "test_id": test_id,
            "success": True,
            "word_f1": m_map.get("extract_unified_grounded_f1", 0.0),
            "word_precision": m_map.get("extract_unified_grounded_precision", 0.0),
            "word_recall": m_map.get("extract_unified_grounded_recall", 0.0),
            "page_f1": m_map.get("extract_unified_page_f1", 0.0),
            "false_grounding": m_map.get("extract_unified_false_grounding_rate", 0.0),
            "abstention_rate": m_map.get("extract_unified_abstention_rate", 0.0),
        }
    except Exception as exc:
        return {"test_id": test_id, "success": False, "error": str(exc)}


def load_cohort_data(doc_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], StandardScaler | None]:
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
            "document_type",
            "split",
            "candidate_count",
            "best_candidate_iou",
            "candidate_hit_at_1",
            "selected_candidate_iou",
        ],
    )

    X_list: list[list[float]] = []
    y_list: list[int] = []
    field_records: list[dict[str, Any]] = []

    curr_idx = 0
    for i in range(len(table)):
        row = {col: table[col][i].as_py() for col in table.column_names}
        f, l, m = extract_candidate_features_and_labels(row)
        if not f:
            continue
        n = len(f)
        X_list.extend(f)
        y_list.extend(l)
        field_records.append({
            "document_id": row["document_id"],
            "field_path": row["field_path"],
            "gold_value": row["gold_value"],
            "start_idx": curr_idx,
            "end_idx": curr_idx + n,
            "candidates": m,
            "doc_type": row["document_type"],
            "split": row["split"],
            "is_table": ("[" in row["field_path"] and "]" in row["field_path"]),
            "baseline_hit": bool(m[0]["best_iou"] >= 0.50),
        })
        curr_idx += n

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, field_records


def main() -> None:
    print("=== EXP-029: SELECTION QUALITY RERANKER ===")
    t_start = time.time()

    # 1. Load Cohort Manifests
    with open(repo_root / "benchmarks" / "exp005_local_manifest.json") as f:
        dev_docs = json.load(f)["documents"]
    with open(repo_root / "benchmarks" / "held_out_manifest.json") as f:
        held_docs = json.load(f)["documents"]

    dev_ids = [d["test_id"] for d in dev_docs]
    held_ids = [d["test_id"] for d in held_docs]
    print(f"Cohort A (Dev): {len(dev_ids)} docs. Cohort B (Held-Out): {len(held_ids)} docs.")

    # 2. Extract Training Features on Cohort A
    print("\n--- PHASE 2 & 4: Extracting Features on Cohort A (Training) ---")
    t0 = time.time()
    X_train_raw, y_train, train_fields = load_cohort_data(dev_ids)
    print(f"Cohort A: {len(train_fields)} fields, {len(X_train_raw)} candidates extracted in {time.time()-t0:.2f}s")
    print(f"Positive rate (best_iou >= 0.50): {np.mean(y_train)*100:.2f}% ({np.sum(y_train)} / {len(y_train)})")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    # 3. Train Logistic Regression
    print("\n--- PHASE 4: Training Logistic Regression Model ---")
    clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)
    t_fit = time.time()
    clf.fit(X_train, y_train)
    print(f"Fit completed in {time.time()-t_fit:.2f}s (iterations: {clf.n_iter_[0]})")

    train_probs = clf.predict_proba(X_train)[:, 1]
    roc_auc = float(roc_auc_score(y_train, train_probs))
    pr_auc = float(average_precision_score(y_train, train_probs))
    brier = float(brier_score_loss(y_train, train_probs))
    loss = float(log_loss(y_train, train_probs))
    train_preds = (train_probs >= 0.50).astype(int)
    acc = float(accuracy_score(y_train, train_preds))

    print(f"ROC AUC:     {roc_auc:.4f}")
    print(f"PR AUC:      {pr_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss:    {loss:.4f}")
    print(f"Candidate Acc (threshold=0.5): {acc*100:.2f}%")

    # Calibration Curve Analysis (10 deciles)
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(train_probs, bins) - 1
    calibration_data = []
    for b in range(10):
        mask = (bin_indices == b)
        if np.sum(mask) > 0:
            mean_pred = float(np.mean(train_probs[mask]))
            true_rate = float(np.mean(y_train[mask]))
            count = int(np.sum(mask))
            calibration_data.append({
                "bin": b,
                "range": [float(bins[b]), float(bins[b+1])],
                "count": count,
                "mean_predicted_probability": round(mean_pred, 4),
                "true_positive_rate": round(true_rate, 4),
            })
    print(f"Calibration binned into {len(calibration_data)} active probability deciles.")

    # Errors by source type on Cohort A
    source_stats: dict[str, dict[str, Any]] = {}
    for i, meta in enumerate([c for f in train_fields for c in f["candidates"]]):
        src = meta["source"]
        if src not in source_stats:
            source_stats[src] = {"total": 0, "pos": 0, "pred_pos": 0, "correct": 0}
        source_stats[src]["total"] += 1
        is_true = (y_train[i] == 1)
        is_pred = (train_preds[i] == 1)
        if is_true:
            source_stats[src]["pos"] += 1
        if is_pred:
            source_stats[src]["pred_pos"] += 1
        if is_true == is_pred:
            source_stats[src]["correct"] += 1

    source_breakdown = {}
    for s, st in sorted(source_stats.items(), key=lambda x: -x[1]["total"]):
        source_breakdown[s] = {
            "total_candidates": st["total"],
            "positive_count": st["pos"],
            "positive_rate": round(st["pos"] / st["total"] * 100, 2),
            "accuracy": round(st["correct"] / st["total"] * 100, 2),
        }

    # Feature Importance & Coefficients
    coefs = clf.coef_[0]
    ranked_indices = np.argsort(np.abs(coefs))[::-1]
    feature_importance_list = []
    for rank, idx in enumerate(ranked_indices, start=1):
        feat_name = FEATURE_NAMES[idx]
        val = float(coefs[idx])
        feature_importance_list.append({
            "rank": rank,
            "feature": feat_name,
            "coefficient": round(val, 4),
            "odds_ratio": round(float(math.exp(val)), 4),
            "abs_coefficient": round(abs(val), 4),
        })

    with open(curr_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feature_importance_list, f, indent=2)
    print(f"Saved feature_importance.json with {len(feature_importance_list)} ranked features.")

    # 4. Cohort A Selection Accuracy & Threshold Sweep (Phase 6)
    print("\n--- PHASE 6: Selective Reranking Sweep on Cohort A ---")
    tau_sweep = {}
    best_tau = 0.0
    best_tau_gain = -999.0
    base_train_hits = sum(1 for f in train_fields if f["baseline_hit"])
    n_train_fields = len(train_fields)

    for tau in [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        correct = 0
        flips_pos = 0
        flips_neg = 0
        for f in train_fields:
            fp = train_probs[f["start_idx"]:f["end_idx"]]
            p_r1 = fp[0]
            challenger_idx = int(np.argmax(fp[1:])) + 1
            p_chal = fp[challenger_idx]

            sel_idx = challenger_idx if (p_chal - p_r1 > tau) else 0
            b_hit = f["baseline_hit"]
            r_hit = bool(f["candidates"][sel_idx]["best_iou"] >= 0.50)
            if r_hit:
                correct += 1
            if sel_idx != 0:
                if not b_hit and r_hit:
                    flips_pos += 1
                elif b_hit and not r_hit:
                    flips_neg += 1

        gain_pp = (correct - base_train_hits) / n_train_fields * 100
        tau_sweep[f"tau_{tau:.2f}"] = {
            "tau": tau,
            "hit_count": correct,
            "hit_rate": round(correct / n_train_fields * 100, 2),
            "gain_pp": round(gain_pp, 2),
            "beneficial_flips": flips_pos,
            "harmful_flips": flips_neg,
            "net_flips": flips_pos - flips_neg,
        }
        if gain_pp > best_tau_gain:
            best_tau_gain = gain_pp
            best_tau = tau

    print(f"Cohort A Baseline Hit@1: {base_train_hits} / {n_train_fields} ({base_train_hits/n_train_fields*100:.2f}%)")
    print(f"Optimal tau selected on Cohort A: tau = {best_tau:.2f} (Gain: {best_tau_gain:+.2f}pp, Net Flips: {tau_sweep[f'tau_{best_tau:.2f}']['net_flips']})")

    # Save training_results.json
    training_results = {
        "experiment": "EXP-029",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')",
        "training_cohort": "Cohort A (32 documents)",
        "train_documents": len(dev_ids),
        "train_fields": n_train_fields,
        "train_candidates": len(X_train_raw),
        "positive_candidate_count": int(np.sum(y_train)),
        "positive_candidate_rate": round(float(np.mean(y_train) * 100), 2),
        "candidate_level_metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(loss, 4),
            "candidate_accuracy": round(acc * 100, 2),
        },
        "top1_selection_metrics": {
            "baseline_hit_count": base_train_hits,
            "baseline_hit_rate": round(base_train_hits / n_train_fields * 100, 2),
            "unselective_rerank_hit_rate": round(tau_sweep["tau_0.00"]["hit_rate"], 2),
            "unselective_rerank_gain_pp": round(tau_sweep["tau_0.00"]["gain_pp"], 2),
            "optimal_selective_tau": best_tau,
            "optimal_selective_hit_rate": round(tau_sweep[f"tau_{best_tau:.2f}"]["hit_rate"], 2),
            "optimal_selective_gain_pp": round(best_tau_gain, 2),
            "optimal_beneficial_flips": tau_sweep[f"tau_{best_tau:.2f}"]["beneficial_flips"],
            "optimal_harmful_flips": tau_sweep[f"tau_{best_tau:.2f}"]["harmful_flips"],
            "optimal_net_flips": tau_sweep[f"tau_{best_tau:.2f}"]["net_flips"],
        },
        "tau_sweep": tau_sweep,
        "calibration_deciles": calibration_data,
        "source_type_breakdown": source_breakdown,
        "top_features": feature_importance_list[:15],
    }

    with open(curr_dir / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2)
    print("Saved training_results.json.")

    # 5. Held-Out Evaluation on Cohort B (Phase 4 & 5)
    print("\n--- PHASE 4 & 5: Held-Out Evaluation on Cohort B (32 docs) ---")
    t_held = time.time()
    X_held_raw, y_held, held_fields = load_cohort_data(held_ids)
    print(f"Cohort B: {len(held_fields)} fields, {len(X_held_raw)} candidates loaded in {time.time()-t_held:.2f}s")

    X_held = scaler.transform(X_held_raw)
    held_probs = clf.predict_proba(X_held)[:, 1]
    held_roc_auc = float(roc_auc_score(y_held, held_probs))
    held_pr_auc = float(average_precision_score(y_held, held_probs))
    held_brier = float(brier_score_loss(y_held, held_probs))

    n_held_fields = len(held_fields)
    base_held_hits = sum(1 for f in held_fields if f["baseline_hit"])

    print(f"Cohort B Candidate ROC AUC: {held_roc_auc:.4f} | PR AUC: {held_pr_auc:.4f} | Brier: {held_brier:.4f}")
    print(f"Cohort B Baseline Hit@1: {base_held_hits} / {n_held_fields} ({base_held_hits/n_held_fields*100:.2f}%)")

    # Evaluate Policies on Cohort B
    # Optimal policy selected from Cohort A: tau = best_tau (e.g. 0.02)
    # Also evaluate tau = 0.00 (unselective) and tau = 0.05
    held_policy_eval = {}
    chosen_policy_rerank_map: dict[tuple[str, str], dict[str, Any]] = {}

    for tau in [0.00, best_tau, 0.05, 0.08, 0.10]:
        correct = 0
        flips_pos = 0
        flips_neg = 0
        flips_neut = 0
        
        # Subgroup counters
        sub_all = {"tot": 0, "base": 0, "rerank": 0, "pos_flip": 0, "neg_flip": 0}
        sub_low_margin = {"tot": 0, "base": 0, "rerank": 0, "pos_flip": 0, "neg_flip": 0}
        sub_ocr_fuzzy = {"tot": 0, "base": 0, "rerank": 0, "pos_flip": 0, "neg_flip": 0}
        sub_non_table = {"tot": 0, "base": 0, "rerank": 0, "pos_flip": 0, "neg_flip": 0}
        sub_table = {"tot": 0, "base": 0, "rerank": 0, "pos_flip": 0, "neg_flip": 0}

        for f in held_fields:
            fp = held_probs[f["start_idx"]:f["end_idx"]]
            p_r1 = fp[0]
            challenger_idx = int(np.argmax(fp[1:])) + 1
            p_chal = fp[challenger_idx]

            should_switch = (p_chal - p_r1 > tau)
            sel_idx = challenger_idx if should_switch else 0

            b_hit = f["baseline_hit"]
            r_hit = bool(f["candidates"][sel_idx]["best_iou"] >= 0.50)

            if tau == best_tau and should_switch:
                chosen_candidate = f["candidates"][sel_idx]
                chosen_policy_rerank_map[(f["document_id"], f["field_path"])] = {
                    "page": chosen_candidate["page"],
                    "bbox": chosen_candidate["bbox"],
                    "source": chosen_candidate["source"],
                    "best_iou": chosen_candidate["best_iou"],
                }

            if r_hit:
                correct += 1
            if sel_idx != 0:
                if not b_hit and r_hit:
                    flips_pos += 1
                elif b_hit and not r_hit:
                    flips_neg += 1
                else:
                    flips_neut += 1

            # Subgroup bookkeeping
            # A. All
            sub_all["tot"] += 1
            if b_hit: sub_all["base"] += 1
            if r_hit: sub_all["rerank"] += 1
            if sel_idx != 0 and not b_hit and r_hit: sub_all["pos_flip"] += 1
            if sel_idx != 0 and b_hit and not r_hit: sub_all["neg_flip"] += 1

            # B. Low-margin: where top-2 candidate probabilities in pool are within 0.10
            sorted_p = sorted(fp, reverse=True)
            if len(sorted_p) >= 2 and (sorted_p[0] - sorted_p[1] <= 0.10):
                sub_low_margin["tot"] += 1
                if b_hit: sub_low_margin["base"] += 1
                if r_hit: sub_low_margin["rerank"] += 1
                if sel_idx != 0 and not b_hit and r_hit: sub_low_margin["pos_flip"] += 1
                if sel_idx != 0 and b_hit and not r_hit: sub_low_margin["neg_flip"] += 1

            # C. OCR / fuzzy
            r1_src = f["candidates"][0]["source"].lower()
            if "fuzzy" in r1_src or "ocr" in r1_src or "recovered" in r1_src:
                sub_ocr_fuzzy["tot"] += 1
                if b_hit: sub_ocr_fuzzy["base"] += 1
                if r_hit: sub_ocr_fuzzy["rerank"] += 1
                if sel_idx != 0 and not b_hit and r_hit: sub_ocr_fuzzy["pos_flip"] += 1
                if sel_idx != 0 and b_hit and not r_hit: sub_ocr_fuzzy["neg_flip"] += 1

            # D & E. Table vs Non-Table
            if f["is_table"]:
                sub_table["tot"] += 1
                if b_hit: sub_table["base"] += 1
                if r_hit: sub_table["rerank"] += 1
                if sel_idx != 0 and not b_hit and r_hit: sub_table["pos_flip"] += 1
                if sel_idx != 0 and b_hit and not r_hit: sub_table["neg_flip"] += 1
            else:
                sub_non_table["tot"] += 1
                if b_hit: sub_non_table["base"] += 1
                if r_hit: sub_non_table["rerank"] += 1
                if sel_idx != 0 and not b_hit and r_hit: sub_non_table["pos_flip"] += 1
                if sel_idx != 0 and b_hit and not r_hit: sub_non_table["neg_flip"] += 1

        held_policy_eval[f"tau_{tau:.2f}"] = {
            "tau": tau,
            "hit_count": correct,
            "hit_rate": round(correct / n_held_fields * 100, 2),
            "gain_pp": round((correct - base_held_hits) / n_held_fields * 100, 2),
            "total_flips": flips_pos + flips_neg + flips_neut,
            "beneficial_flips": flips_pos,
            "harmful_flips": flips_neg,
            "neutral_flips": flips_neut,
            "net_flips": flips_pos - flips_neg,
            "subgroups": {
                "all_multi_candidate": {
                    "total": sub_all["tot"],
                    "baseline_hit_rate": round(sub_all["base"] / max(1, sub_all["tot"]) * 100, 2),
                    "rerank_hit_rate": round(sub_all["rerank"] / max(1, sub_all["tot"]) * 100, 2),
                    "gain_pp": round((sub_all["rerank"] - sub_all["base"]) / max(1, sub_all["tot"]) * 100, 2),
                    "beneficial_flips": sub_all["pos_flip"],
                    "harmful_flips": sub_all["neg_flip"],
                },
                "low_margin": {
                    "total": sub_low_margin["tot"],
                    "baseline_hit_rate": round(sub_low_margin["base"] / max(1, sub_low_margin["tot"]) * 100, 2),
                    "rerank_hit_rate": round(sub_low_margin["rerank"] / max(1, sub_low_margin["tot"]) * 100, 2),
                    "gain_pp": round((sub_low_margin["rerank"] - sub_low_margin["base"]) / max(1, sub_low_margin["tot"]) * 100, 2),
                    "beneficial_flips": sub_low_margin["pos_flip"],
                    "harmful_flips": sub_low_margin["neg_flip"],
                },
                "ocr_fuzzy": {
                    "total": sub_ocr_fuzzy["tot"],
                    "baseline_hit_rate": round(sub_ocr_fuzzy["base"] / max(1, sub_ocr_fuzzy["tot"]) * 100, 2),
                    "rerank_hit_rate": round(sub_ocr_fuzzy["rerank"] / max(1, sub_ocr_fuzzy["tot"]) * 100, 2),
                    "gain_pp": round((sub_ocr_fuzzy["rerank"] - sub_ocr_fuzzy["base"]) / max(1, sub_ocr_fuzzy["tot"]) * 100, 2),
                    "beneficial_flips": sub_ocr_fuzzy["pos_flip"],
                    "harmful_flips": sub_ocr_fuzzy["neg_flip"],
                },
                "table_fields": {
                    "total": sub_table["tot"],
                    "baseline_hit_rate": round(sub_table["base"] / max(1, sub_table["tot"]) * 100, 2),
                    "rerank_hit_rate": round(sub_table["rerank"] / max(1, sub_table["tot"]) * 100, 2),
                    "gain_pp": round((sub_table["rerank"] - sub_table["base"]) / max(1, sub_table["tot"]) * 100, 2),
                    "beneficial_flips": sub_table["pos_flip"],
                    "harmful_flips": sub_table["neg_flip"],
                },
                "non_table_fields": {
                    "total": sub_non_table["tot"],
                    "baseline_hit_rate": round(sub_non_table["base"] / max(1, sub_non_table["tot"]) * 100, 2),
                    "rerank_hit_rate": round(sub_non_table["rerank"] / max(1, sub_non_table["tot"]) * 100, 2),
                    "gain_pp": round((sub_non_table["rerank"] - sub_non_table["base"]) / max(1, sub_non_table["tot"]) * 100, 2),
                    "beneficial_flips": sub_non_table["pos_flip"],
                    "harmful_flips": sub_non_table["neg_flip"],
                },
            },
        }

    opt_key = f"tau_{best_tau:.2f}"
    print(f"\nHeld-Out Results on Cohort B ({opt_key}):")
    print(f"  Hit@1 Rate: {held_policy_eval[opt_key]['hit_rate']}% (Baseline: {base_held_hits/n_held_fields*100:.2f}%)")
    print(f"  Delta: {held_policy_eval[opt_key]['gain_pp']:+.2f}pp")
    print(f"  Beneficial Flips: {held_policy_eval[opt_key]['beneficial_flips']}")
    print(f"  Harmful Flips:    {held_policy_eval[opt_key]['harmful_flips']}")
    print(f"  Net Flips:        {held_policy_eval[opt_key]['net_flips']:+d}")

    # 6. Official ExtractBench Evaluation on Cohort B
    print("\n--- Generating Official Prediction Files for Cohort B (32 Docs) ---")
    base_pred_dir = repo_root / "research" / "experiments" / "EXP-028E" / "predictions" / "tonerhound"
    rerank_pred_dir = curr_dir / "predictions_reranked" / "tonerhound"
    rerank_pred_dir.mkdir(parents=True, exist_ok=True)

    data_dir = repo_root / "research" / "data" / "full"

    base_eval_tasks = []
    rerank_eval_tasks = []

    for tid in held_ids:
        src_file = base_pred_dir / f"{tid}.result.json"
        dst_file = rerank_pred_dir / f"{tid}.result.json"
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        with open(src_file, encoding="utf-8") as fp:
            data = json.load(fp)

        # Update citations if modified by reranker
        modified_count = 0
        cits = data.get("output", {}).get("field_citations", [])
        for c in cits:
            k = (tid, c.get("field_path"))
            if k in chosen_policy_rerank_map:
                c["page"] = chosen_policy_rerank_map[k]["page"]
                c["bbox"] = chosen_policy_rerank_map[k]["bbox"]
                modified_count += 1

        with open(dst_file, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

        pdf_path = data_dir / f"{tid}.pdf"
        base_eval_tasks.append({
            "test_id": tid,
            "pdf_path": str(pdf_path),
            "res_path": str(src_file),
        })
        rerank_eval_tasks.append({
            "test_id": tid,
            "pdf_path": str(pdf_path),
            "res_path": str(dst_file),
        })

    print(f"Reranked predictions written. Evaluating baseline vs reranker across {len(held_ids)} docs with 4 workers...")
    
    # Official baseline evaluation
    t_ev0 = time.time()
    base_doc_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_eval_single_doc, t): t["test_id"] for t in base_eval_tasks}
        for fut in as_completed(futures):
            base_doc_results.append(fut.result())
    t_base_ev = time.time() - t_ev0

    # Official reranker evaluation
    t_ev1 = time.time()
    rerank_doc_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_eval_single_doc, t): t["test_id"] for t in rerank_eval_tasks}
        for fut in as_completed(futures):
            rerank_doc_results.append(fut.result())
    t_rerank_ev = time.time() - t_ev1

    # Macro aggregation
    base_f1s = [d["word_f1"] for d in base_doc_results if d.get("success")]
    base_precs = [d["word_precision"] for d in base_doc_results if d.get("success")]
    base_recs = [d["word_recall"] for d in base_doc_results if d.get("success")]
    base_pages = [d["page_f1"] for d in base_doc_results if d.get("success")]
    base_fg = [d["false_grounding"] for d in base_doc_results if d.get("success")]

    rerank_f1s = [d["word_f1"] for d in rerank_doc_results if d.get("success")]
    rerank_precs = [d["word_precision"] for d in rerank_doc_results if d.get("success")]
    rerank_recs = [d["word_recall"] for d in rerank_doc_results if d.get("success")]
    rerank_pages = [d["page_f1"] for d in rerank_doc_results if d.get("success")]
    rerank_fg = [d["false_grounding"] for d in rerank_doc_results if d.get("success")]

    macro_base = {
        "word_f1": float(np.mean(base_f1s) * 100),
        "word_precision": float(np.mean(base_precs) * 100),
        "word_recall": float(np.mean(base_recs) * 100),
        "page_f1": float(np.mean(base_pages) * 100),
        "false_grounding": float(np.mean(base_fg) * 100),
    }

    macro_rerank = {
        "word_f1": float(np.mean(rerank_f1s) * 100),
        "word_precision": float(np.mean(rerank_precs) * 100),
        "word_recall": float(np.mean(rerank_recs) * 100),
        "page_f1": float(np.mean(rerank_pages) * 100),
        "false_grounding": float(np.mean(rerank_fg) * 100),
    }

    word_f1_gain_pp = macro_rerank["word_f1"] - macro_base["word_f1"]
    prec_gain_pp = macro_rerank["word_precision"] - macro_base["word_precision"]
    rec_gain_pp = macro_rerank["word_recall"] - macro_base["word_recall"]
    page_gain_pp = macro_rerank["page_f1"] - macro_base["page_f1"]
    fg_delta_pp = macro_rerank["false_grounding"] - macro_base["false_grounding"]

    print("\n=======================================================")
    print("OFFICIAL EXTRACTBENCH COHORT B HELD-OUT RESULTS")
    print("=======================================================")
    print(f"Metric                  Baseline     Reranker     Delta")
    print(f"Word Grounding F1:      {macro_base['word_f1']:6.2f}%     {macro_rerank['word_f1']:6.2f}%     {word_f1_gain_pp:+6.2f}pp")
    print(f"Word Precision:         {macro_base['word_precision']:6.2f}%     {macro_rerank['word_precision']:6.2f}%     {prec_gain_pp:+6.2f}pp")
    print(f"Word Recall:            {macro_base['word_recall']:6.2f}%     {macro_rerank['word_recall']:6.2f}%     {rec_gain_pp:+6.2f}pp")
    print(f"Page Grounding F1:      {macro_base['page_f1']:6.2f}%     {macro_rerank['page_f1']:6.2f}%     {page_gain_pp:+6.2f}pp")
    print(f"False Grounding Rate:   {macro_base['false_grounding']:6.2f}%     {macro_rerank['false_grounding']:6.2f}%     {fg_delta_pp:+6.2f}pp")
    print("=======================================================")

    # Decision Gate Classification
    if word_f1_gain_pp >= 2.0:
        gate_status = "STRONG_SIGNAL"
        gate_recommendation = "Prepare feature-flagged production integration candidate."
    elif word_f1_gain_pp >= 1.0:
        if word_f1_gain_pp >= 1.5:
            gate_status = "PROMISING_PASS"
            gate_recommendation = "Prepare feature-flagged production integration candidate."
        else:
            gate_status = "PROMISING_BELOW_INTEGRATION_THRESHOLD"
            gate_recommendation = "STOP production integration. Gain is < +1.5pp. Diagnose failure families."
    elif word_f1_gain_pp > 0.0:
        gate_status = "WEAK_SIGNAL"
        gate_recommendation = "STOP production integration. Gain is < +1.0pp. Diagnose failure families."
    else:
        gate_status = "REGRESSION"
        gate_recommendation = "STOP production integration. Reranker regressed baseline performance."

    print(f"\nDECISION GATE STATUS: {gate_status}")
    print(f"RECOMMENDATION:       {gate_recommendation}")

    heldout_results = {
        "experiment": "EXP-029",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "held_out_cohort": "Cohort B (32 documents)",
        "documents_evaluated": len(held_ids),
        "total_fields": n_held_fields,
        "selected_policy": {
            "name": f"Conservatism Margin Threshold tau = {best_tau:.2f}",
            "tau": best_tau,
            "criterion": f"Switch only if P(challenger) - P(rank_1) > {best_tau:.2f}",
        },
        "official_extractbench_metrics": {
            "baseline": macro_base,
            "reranker": macro_rerank,
            "delta_pp": {
                "word_grounding_f1": round(word_f1_gain_pp, 4),
                "word_precision": round(prec_gain_pp, 4),
                "word_recall": round(rec_gain_pp, 4),
                "page_f1": round(page_gain_pp, 4),
                "false_grounding": round(fg_delta_pp, 4),
            },
        },
        "candidate_level_metrics": {
            "roc_auc": round(held_roc_auc, 4),
            "pr_auc": round(held_pr_auc, 4),
            "brier_score": round(held_brier, 4),
        },
        "selection_accuracy_metrics": held_policy_eval[opt_key],
        "all_policy_evaluations": held_policy_eval,
        "decision_gate": {
            "heldout_word_f1_gain_pp": round(word_f1_gain_pp, 4),
            "status": gate_status,
            "recommendation": gate_recommendation,
        },
    }

    with open(curr_dir / "heldout_results.json", "w", encoding="utf-8") as f:
        json.dump(heldout_results, f, indent=2)
    print("Saved heldout_results.json.")
    print(f"\nAll EXP-029 execution finished in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
