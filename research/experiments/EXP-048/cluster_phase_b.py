"""EXP-048 Phase B: Per-Document and Global Failure Clustering.

For the top 30 documents by lift_potential:
B.1 — Load per-field failure data (field_path, gold_value, gold_page, predicted_value, predicted_page, failure_class, IoU).
B.2 — Cluster failures within each document by failure_class and field path pattern.
B.3 — Merge into global clusters and report top 20 global clusters by total field count.
Output: research/experiments/EXP-048/audit/failure_clusters.json
"""

import json
import re
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


def _canonicalize_path(path: str) -> str:
    """Normalize array indices into [N]. e.g. holdings[12].shares -> holdings[N].shares"""
    return re.sub(r"\[\d+\]", "[N]", path)


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


def run_phase_b():
    print("=== EXP-048 Phase B: Per-Document & Global Failure Clustering ===", flush=True)
    t0 = time.time()

    # Load lift ranking to get top 30 documents
    with open(audit_dir / "lift_ranking.json", encoding="utf-8") as f:
        lift_data = json.load(f)
    top30_docs_info = lift_data["top_documents_by_lift_potential"][:30]
    top30_ids = set(d["test_id"] for d in top30_docs_info)
    print(f"Loaded top 30 documents by lift potential ({len(top30_ids)} unique IDs).", flush=True)

    # Load full field level data from EXP-045
    field_level_path = repo_root / "research" / "observer" / "reports" / "EXP-045" / "field_level.json"
    print(f"Loading {field_level_path}...", flush=True)
    with open(field_level_path, encoding="utf-8") as f:
        full_fields_data = json.load(f)
    all_fields = full_fields_data.get("fields", [])
    print(f"Loaded {len(all_fields)} fields in {time.time() - t0:.2f}s.", flush=True)

    # Check which of the top 30 docs were modified in EXP-047
    with open(exp047_dir / "results" / "full_benchmark_results.json", encoding="utf-8") as f:
        exp047_meta = json.load(f)
    exp047_mod_ids = set(
        d["test_id"] for d in exp047_meta["per_document"]
        if d.get("modified") and d["test_id"] in top30_ids
    )
    print(f"{len(exp047_mod_ids)} of top 30 documents were modified in EXP-047: {exp047_mod_ids}", flush=True)

    # For modified docs, load EXP-047 predictions and re-verify pass/fail
    exp047_preds_dir = exp047_dir / "predictions" / "tonerhound"
    exp047_passing_by_doc: dict[str, set[str]] = defaultdict(set)
    for tid in exp047_mod_ids:
        pdf_p = data_dir / f"{tid}.pdf"
        pred_p = exp047_preds_dir / f"{tid}.result.json"
        if pdf_p.exists() and pred_p.exists():
            tc = load_test_case(pdf_p)
            rules = tc.get_extract_field_rules()
            alt_values, ev_boxes, ev_pages, _ = build_rule_indexes(rules)
            with open(pred_p, encoding="utf-8") as pf:
                pred_json = json.load(pf)
            cits = {
                c["field_path"]: c
                for c in pred_json.get("output", {}).get("field_citations", [])
                if c.get("field_path") and c.get("page") is not None
            }
            for r in rules:
                fp = r.field_path
                if _check_passes(cits.get(fp), fp, ev_boxes, ev_pages):
                    exp047_passing_by_doc[tid].add(fp)

    # Filter fields to only top 30 documents and failing fields
    doc_failing_fields: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_failing_count = 0

    for fld in all_fields:
        doc_id = fld["document_id"]
        if doc_id not in top30_ids:
            continue

        fp = fld["field_path"]
        # If document was modified in EXP-047, check if field now passes
        if doc_id in exp047_mod_ids:
            if fp in exp047_passing_by_doc[doc_id]:
                continue
            is_success = False
        else:
            is_success = fld.get("grounding_success", False)

        if is_success:
            continue

        failure_class = fld.get("failure_class", "UNKNOWN")
        if failure_class in ("ALREADY_RESOLVED", "AUDIT_CLASSIFICATION_ERROR"):
            continue

        gold_val = fld.get("gold_value")
        pred_val = fld.get("predicted_value")
        gold_pg = fld.get("gold_page")
        pred_pg = fld.get("predicted_page")
        iou = fld.get("iou", 0.0)

        record = {
            "field_path": fp,
            "path_pattern": _canonicalize_path(fp),
            "gold_value": gold_val,
            "gold_page": gold_pg,
            "predicted_value": pred_val,
            "predicted_page": pred_pg,
            "failure_class": failure_class,
            "iou": iou,
        }
        doc_failing_fields[doc_id].append(record)
        total_failing_count += 1

    print(f"Total failing fields across top 30 documents: {total_failing_count}", flush=True)

    # B.2 Cluster failures within each document
    per_doc_clusters: dict[str, list[dict[str, Any]]] = {}

    for doc_id, fails in doc_failing_fields.items():
        # Group by (failure_class, path_pattern)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for fld in fails:
            key = (fld["failure_class"], fld["path_pattern"])
            groups[key].append(fld)

        doc_cluster_list = []
        for (f_cls, p_pat), f_items in groups.items():
            sample_5 = [
                {
                    "field_path": x["field_path"],
                    "gold_value": x["gold_value"],
                    "gold_page": x["gold_page"],
                    "predicted_value": x["predicted_value"],
                    "predicted_page": x["predicted_page"],
                    "iou": x["iou"],
                }
                for x in f_items[:5]
            ]
            # Analyze pattern
            pages = sorted(list(set(x["gold_page"] for x in f_items if x["gold_page"] is not None)))
            sample_vals = [str(x["gold_value"]) for x in f_items[:10] if x["gold_value"] is not None]
            all_numeric = all(
                any(c.isdigit() for c in v) and not re.search(r"[a-zA-Z]{3,}", v)
                for v in sample_vals
            ) if sample_vals else False

            common_pattern = (
                f"{len(f_items)} fields with class {f_cls} matching '{p_pat}'. "
                f"Pages involved: {pages[:5]}{'...' if len(pages) > 5 else ''}. "
                f"Numeric content: {all_numeric}."
            )

            doc_cluster_list.append({
                "document_id": doc_id,
                "failure_class": f_cls,
                "path_pattern": p_pat,
                "field_count": len(f_items),
                "pages": pages,
                "common_pattern": common_pattern,
                "sample_examples": sample_5,
            })

        # Sort clusters in document by field count descending
        doc_cluster_list.sort(key=lambda c: -c["field_count"])
        per_doc_clusters[doc_id] = doc_cluster_list

    # B.3 Global failure clustering
    # Merge per-document clusters into global clusters
    # Key: (failure_class, path_pattern)
    global_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for doc_id, c_list in per_doc_clusters.items():
        for c in c_list:
            key = (c["failure_class"], c["path_pattern"])
            global_groups[key].append(c)

    global_clusters = []
    for (f_cls, p_pat), occurrences in global_groups.items():
        total_fields = sum(occ["field_count"] for occ in occurrences)
        affected_docs = sorted(list(set(occ["document_id"] for occ in occurrences)))
        # Collect sample of examples from various documents
        combined_samples = []
        for occ in occurrences:
            for s in occ["sample_examples"]:
                combined_samples.append({
                    "document_id": occ["document_id"],
                    **s,
                })
                if len(combined_samples) >= 10:
                    break
            if len(combined_samples) >= 10:
                break

        # Describe the cluster
        doc_types = ", ".join(d.split("/")[0] for d in affected_docs[:3])
        description = (
            f"Pattern '{p_pat}' failed under {f_cls} across {len(affected_docs)} documents "
            f"({total_fields} total fields)."
        )

        global_clusters.append({
            "cluster_id": f"CLUST_{f_cls}_{p_pat}",
            "failure_class": f_cls,
            "path_pattern": p_pat,
            "total_fields": total_fields,
            "affected_document_count": len(affected_docs),
            "affected_documents": affected_docs,
            "description": description,
            "sample_examples": combined_samples[:5],
        })

    # Sort global clusters by total field count descending
    global_clusters.sort(key=lambda c: -c["total_fields"])
    top20_global = global_clusters[:20]

    output_payload = {
        "experiment_id": "EXP-048",
        "description": "Per-document and global failure clusters for top 30 documents by lift potential",
        "top_30_documents_analyzed": len(per_doc_clusters),
        "total_failing_fields_in_top_30": total_failing_count,
        "top_20_global_clusters": top20_global,
        "per_document_clusters": per_doc_clusters,
    }

    out_file = audit_dir / "failure_clusters.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved failure clusters to {out_file}", flush=True)

    print("\n--- TOP 20 GLOBAL FAILURE CLUSTERS ---", flush=True)
    for i, c in enumerate(top20_global, 1):
        print(f"{i:2d}. {c['cluster_id']:<60} Fields: {c['total_fields']:5d} | Docs: {c['affected_document_count']:2d}")

    return output_payload


if __name__ == "__main__":
    run_phase_b()
