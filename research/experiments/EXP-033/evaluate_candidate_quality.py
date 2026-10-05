"""EXP-033: Candidate Quality & Quantity Evaluator across 200 Audited Fields.

Implements Phase 8:
Measures candidate quality and candidate quantity separately:
- candidate_count
- has_any_candidate
- has_valid_IoU_candidate
- best_candidate_IoU
- candidate_pool_recall

Compares across all configurations:
- A0_baseline
- A1_boolean
- A1_token_strip
- A1_global
- A1_multiline
- A2_combo
- A3_all (ENABLE_EXP033_CANDIDATE_EXPANSION)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Single thread math libs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from extract_bench.test_cases.loader import load_test_case

from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput
from tonerhound.resolution.resolver import EvidenceResolver


def run_candidate_quality_evaluation() -> dict[str, Any]:
    samples_file = repo_root / "research" / "experiments" / "EXP-033" / "reconciliation_samples.json"
    data_dir = repo_root / "research" / "data" / "full"
    out_file = repo_root / "research" / "experiments" / "EXP-033" / "candidate_quality_analysis.json"

    with open(samples_file) as fp:
        samples = json.load(fp)

    print(f"Loaded {len(samples)} audited fields from {samples_file.name}.")

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_doc[s["document_id"]].append(s)

    print(f"Spans {len(by_doc)} unique documents.")

    configs = {
        "A0_baseline": {
            "enable_exp033_candidate_expansion": False,
            "enable_boolean_expansion": False,
            "enable_token_strip_recovery": False,
            "enable_global_search_relaxation": False,
            "enable_multi_line_recovery": False,
        },
        "A1_boolean": {
            "enable_boolean_expansion": True,
            "enable_token_strip_recovery": False,
            "enable_global_search_relaxation": False,
            "enable_multi_line_recovery": False,
        },
        "A1_token_strip": {
            "enable_boolean_expansion": False,
            "enable_token_strip_recovery": True,
            "enable_global_search_relaxation": False,
            "enable_multi_line_recovery": False,
        },
        "A1_global": {
            "enable_boolean_expansion": False,
            "enable_token_strip_recovery": False,
            "enable_global_search_relaxation": True,
            "enable_multi_line_recovery": False,
        },
        "A1_multiline": {
            "enable_boolean_expansion": False,
            "enable_token_strip_recovery": False,
            "enable_global_search_relaxation": False,
            "enable_multi_line_recovery": True,
        },
        "A2_combo": {
            "enable_boolean_expansion": True,
            "enable_token_strip_recovery": True,
            "enable_global_search_relaxation": True,
            "enable_multi_line_recovery": True,
        },
        "A3_all": {
            "enable_exp033_candidate_expansion": True,
        },
    }

    eval_data: dict[str, list[dict[str, Any]]] = {c: [] for c in configs}

    t0 = time.perf_counter()

    for doc_idx_num, (doc_id, doc_samples) in enumerate(by_doc.items(), start=1):
        pdf_path = data_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            print(f"Skipping missing: {pdf_path}")
            continue

        print(f"[{doc_idx_num}/{len(by_doc)}] Processing {doc_id} ({len(doc_samples)} fields)...")
        is_ocr = ("corrupted" in doc_id) or ("short" in doc_id and any(k in doc_id.lower() for k in ("w2", "w14", "1040", "h9")))
        doc_idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=is_ocr, backend="hybrid")
        tc = load_test_case(pdf_path)
        rules_by_path = {r.field_path: r for r in tc.get_extract_field_rules()}

        # Instantiate resolvers for baseline and all-expansion
        res_baseline = EvidenceResolver(doc_idx, enable_verification=True, enable_candidate_recovery=True, enable_exp033_candidate_expansion=False)
        res_all = EvidenceResolver(doc_idx, enable_verification=True, enable_candidate_recovery=True, enable_exp033_candidate_expansion=True)

        resolvers: dict[str, EvidenceResolver] = {
            "A0_baseline": res_baseline,
            "A3_all": res_all,
        }

        for s in doc_samples:
            fpath = s["field_path"]
            rule = rules_by_path.get(fpath)
            gold_boxes = []
            if rule:
                for e in rule.evidence:
                    if e.bbox is not None and e.page is not None:
                        gold_boxes.append((e.page, (e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3])))
            elif s.get("gold_pages"):
                gold_boxes = [(p, (0.0, 0.0, 1.0, 1.0)) for p in s["gold_pages"]]

            p_hint = s["gold_pages"][0] if s.get("gold_pages") else None
            inp = ExtractionInput(field=fpath, value=s["gold_value"], page_hint=p_hint)

            cands_a0 = res_baseline.collect_candidates(inp)
            cands_a3 = res_all.collect_candidates(inp)

            cands_by_cfg = {
                "A0_baseline": cands_a0,
                "A3_all": cands_a3,
            }

            if len(cands_a3) == len(cands_a0):
                # No change under any flag
                for c_name in ("A1_boolean", "A1_token_strip", "A1_global", "A1_multiline", "A2_combo"):
                    cands_by_cfg[c_name] = cands_a0
            else:
                # Evaluate individual configs for attribution
                for c_name in ("A1_boolean", "A1_token_strip", "A1_global", "A1_multiline", "A2_combo"):
                    if c_name not in resolvers:
                        resolvers[c_name] = EvidenceResolver(
                            doc_idx,
                            enable_verification=True,
                            enable_candidate_recovery=True,
                            **configs[c_name],
                        )
                    cands_by_cfg[c_name] = resolvers[c_name].collect_candidates(inp)

            # Record metrics for each config
            for cfg_name, cands in cands_by_cfg.items():
                c_count = len(cands)
                has_any = c_count > 0
                best_iou = 0.0
                if gold_boxes:
                    for c in cands:
                        c_p = c.bbox.page if hasattr(c.bbox, "page") else c.page
                        c_box = (c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height)
                        for gp, gb in gold_boxes:
                            if gp == c_p:
                                iou = iou_xywh(c_box, gb)
                                if iou > best_iou:
                                    best_iou = iou
                has_valid_iou = best_iou >= 0.50

                eval_data[cfg_name].append({
                    "document_id": doc_id,
                    "field_path": fpath,
                    "category": s["category"],
                    "candidate_count": c_count,
                    "has_any_candidate": has_any,
                    "has_valid_iou_candidate": has_valid_iou,
                    "best_candidate_iou": round(best_iou, 4),
                })

        del doc_idx, tc, rules_by_path, resolvers
        gc.collect()

    summary: dict[str, Any] = {}
    for cfg_name, items in eval_data.items():
        counts = [it["candidate_count"] for it in items]
        has_anys = [it["has_any_candidate"] for it in items]
        has_valids = [it["has_valid_iou_candidate"] for it in items]
        best_ious = [it["best_candidate_iou"] for it in items]

        cat_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "has_any": 0, "has_valid": 0})
        for it in items:
            cat = it["category"]
            cat_stats[cat]["total"] += 1
            if it["has_any_candidate"]:
                cat_stats[cat]["has_any"] += 1
            if it["has_valid_iou_candidate"]:
                cat_stats[cat]["has_valid"] += 1

        summary[cfg_name] = {
            "total_audited_fields": len(items),
            "candidate_pool_recall": round(float(np.mean(has_valids) * 100), 2),
            "has_any_candidate_rate": round(float(np.mean(has_anys) * 100), 2),
            "has_any_candidate_count": sum(has_anys),
            "valid_iou_candidate_count": sum(has_valids),
            "avg_candidate_count": round(float(np.mean(counts)), 2),
            "max_candidate_count": int(np.max(counts)) if counts else 0,
            "avg_best_candidate_iou": round(float(np.mean(best_ious)), 4),
            "category_breakdown": {
                cat: {
                    "total": st["total"],
                    "has_any": st["has_any"],
                    "has_any_pct": round(st["has_any"] / st["total"] * 100, 1),
                    "has_valid": st["has_valid"],
                    "valid_pct": round(st["has_valid"] / st["total"] * 100, 1),
                }
                for cat, st in sorted(cat_stats.items())
            },
        }

    total_time = time.perf_counter() - t0
    output_dict = {
        "metadata": {
            "experiment": "EXP-033",
            "phase": "Phase 8: Candidate Quality & Quantity Analysis",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_fields": len(samples),
            "total_documents": len(by_doc),
            "elapsed_seconds": round(total_time, 2),
        },
        "configurations": summary,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(output_dict, fp, indent=2)

    print("\n" + "=" * 80)
    print("CANDIDATE QUALITY & QUANTITY COMPARISON (200 AUDITED FIELDS)")
    print("=" * 80)
    print(f"{'Configuration':<16} | {'Has Any (%)':<12} | {'Has Valid IoU (%)':<18} | {'Avg Cands':<10} | {'Max Cands':<10} | {'Avg Best IoU':<12}")
    print("-" * 80)
    for cfg_name, st in summary.items():
        print(f"{cfg_name:<16} | {st['has_any_candidate_rate']:>6.1f}% ({st['has_any_candidate_count']:>3}) | {st['candidate_pool_recall']:>6.1f}% ({st['valid_iou_candidate_count']:>3})        | {st['avg_candidate_count']:>8.2f} | {st['max_candidate_count']:>8}   | {st['avg_best_candidate_iou']:>10.4f}")

    return output_dict


if __name__ == "__main__":
    run_candidate_quality_evaluation()
