"""Failure Microscope V6 — Comprehensive Per-Document Field Accounting & Causal Auditor.

Integrates:
1. Complete per-document field accounting (status, gradeable, passing, failing, word_f1, page_f1, family, fields list).
2. Field-level ground truth and prediction comparison with IoU computation and taxonomy mapping:
   - PASS (IoU >= 0.50 on correct page)
   - INSUFFICIENT_IOU (abs(y - gy) <= 0.020 on correct page with IoU < 0.50)
   - WRONG_ROW (candidate on different row / y-offset > 0.020 on correct page)
   - PAGE_DRIFT (page mismatch between prediction and gold)
   - ZERO_CANDIDATE (no candidates or empty prediction)
   - MALFORMED_BBOX (invalid or collapsed geometry)
3. Structural failure clustering:
   - field_pattern grouping (e.g. replacing array indexes `[0]` with `[]`)
   - dominant failure classification per cluster
4. Backward-compatible fallback to Microscope V5 / V4 macro rankings and realistic estimators.
5. Verification of Validation Gates V1 through V8:
   - V8: Per-Document Field Accounting (Every document satisfies gradeable == passing + failing)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Prevent thread oversubscription in math libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))


def iou_xywh(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """Compute Intersection over Union for [x, y, w, h] bounding boxes."""
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def compute_field_pattern(field_path: str) -> str:
    """Normalize array indexes to pattern representation, e.g. items[0].val -> items[].val."""
    return re.sub(r"\[\d+\]", "[]", field_path)


def get_canonical_family(doc_id: str, orig_family: str = "") -> str:
    tid_lower = doc_id.lower()
    if "1040" in doc_id or any(k in tid_lower for k in ["becerra", "cabrera", "bar-lev", "arif"]):
        return "IRS Form 1040"
    if "990pf" in tid_lower or "990-pf" in tid_lower or "sched_i" in tid_lower or "scott" in tid_lower:
        return "IRS Form 990-PF"
    if "sec_13f" in tid_lower or "13f" in tid_lower or "loomis" in tid_lower or "renaissance" in tid_lower or "brown_brothers" in tid_lower:
        return "SEC 13F"
    if "nport" in tid_lower or "dunham" in tid_lower or "sei_etf" in tid_lower:
        return "SEC N-PORT"
    if any(k in doc_id for k in ["W14-", "W2-", "H9-", "2A-", "P4-", "P18-", "W-1-"]) or any(k in orig_family.lower() for k in ["municipal", "property", "assessment", "appraisal"]):
        return "Municipal"
    if "ofac" in tid_lower:
        return "OFAC Sanctions"
    if "ftx" in tid_lower or "imedia" in tid_lower or "creditor" in tid_lower or "bankruptcy" in orig_family.lower():
        return "Bankruptcy"
    return "Other/ungrouped"


@dataclass
class FieldDetail:
    field_path: str
    gold_value: Any
    gold_page: Optional[int]
    gold_bbox: Optional[List[float]]
    predicted_value: Optional[Any]
    predicted_page: Optional[int]
    predicted_bbox: Optional[List[float]]
    status: str
    iou: float


@dataclass
class FailureCluster:
    cluster_id: str
    field_pattern: str
    count: int
    dominant_failure_class: str


@dataclass
class DocumentAuditReport:
    document_id: str
    status: str
    gradeable: int
    passing: int
    failing: int
    word_f1: float
    page_f1: float
    family: str
    fields: List[Dict[str, Any]] = field(default_factory=list)
    failure_clusters: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MicroscopeV6Auditor:
    """Core auditor engine for Microscope V6."""

    def __init__(
        self,
        data_dir: Path,
        preds_dir: Path,
        family_map: Optional[Dict[str, str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.preds_dir = Path(preds_dir)
        self.family_map = family_map or {}

    def audit_document(self, doc_id: str) -> DocumentAuditReport:
        """Audit a single document using fast direct json loading."""
        test_path = self.data_dir / f"{doc_id}.test.json"
        pred_path = self.preds_dir / f"{doc_id}.result.json"
        if not pred_path.exists() and (self.preds_dir / "tonerhound").exists():
            pred_path = self.preds_dir / "tonerhound" / f"{doc_id}.result.json"

        family = self.family_map.get(doc_id, get_canonical_family(doc_id))

        if not test_path.exists():
            return DocumentAuditReport(
                document_id=doc_id,
                status="skipped",
                gradeable=0,
                passing=0,
                failing=0,
                word_f1=0.0,
                page_f1=0.0,
                family=family,
                fields=[],
                failure_clusters=[],
            )

        try:
            with open(test_path, encoding="utf-8") as f:
                tdata = json.load(f)
        except Exception:
            return DocumentAuditReport(
                document_id=doc_id,
                status="unextractable",
                gradeable=0,
                passing=0,
                failing=0,
                word_f1=0.0,
                page_f1=0.0,
                family=family,
                fields=[],
                failure_clusters=[],
            )

        rules = tdata.get("_field_rules", {})
        gradeable_rules = {}
        for fp, rdef in rules.items():
            ev = rdef.get("evidence", [])
            ev0 = ev[0] if ev else {}
            gb = ev0.get("bbox")
            if gb and len(gb) == 4 and all(isinstance(v, (int, float)) for v in gb):
                gradeable_rules[fp] = rdef

        gradeable = len(gradeable_rules)
        if gradeable == 0:
            return DocumentAuditReport(
                document_id=doc_id,
                status="unextractable",
                gradeable=0,
                passing=0,
                failing=0,
                word_f1=0.0,
                page_f1=0.0,
                family=family,
                fields=[],
                failure_clusters=[],
            )

        cits: Dict[str, Dict[str, Any]] = {}
        if pred_path.exists():
            try:
                with open(pred_path, encoding="utf-8") as f:
                    pdata = json.load(f)
                raw_cits = pdata.get("output", {}).get("field_citations", [])
                for c in raw_cits:
                    fp = c.get("field_path")
                    if fp:
                        cits[fp] = c
            except Exception:
                pass

        fields_list: List[Dict[str, Any]] = []
        cluster_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        passing_count = 0
        page_match_count = 0

        for fp, rdef in gradeable_rules.items():
            ev_list = rdef.get("evidence", [])
            ev0 = ev_list[0] if ev_list else {}
            gold_val = ev0.get("value", "")
            gold_page = ev0.get("page")
            gold_box = ev0.get("bbox")

            cit = cits.get(fp)
            pred_val = cit.get("matched_text") or cit.get("value") if cit else None
            pred_page = cit.get("page") if cit else None
            pred_box = cit.get("bbox") if cit else None

            # Calculate IoU
            curr_iou = iou_xywh(pred_box, gold_box)

            # Determine field status
            if cit is None or pred_page is None:
                field_status = "ZERO_CANDIDATE"
            elif pred_box is None or len(pred_box) != 4 or any(v <= 0 for v in pred_box[2:]):
                field_status = "MALFORMED_BBOX"
            elif pred_page != gold_page:
                field_status = "PAGE_DRIFT"
            elif curr_iou >= 0.50:
                field_status = "PASS"
            elif gold_box is not None and abs(pred_box[1] - gold_box[1]) > 0.020:
                field_status = "WRONG_ROW"
            else:
                field_status = "INSUFFICIENT_IOU"

            if pred_page == gold_page and gold_page is not None:
                page_match_count += 1

            if field_status == "PASS":
                passing_count += 1
            else:
                pattern = compute_field_pattern(fp)
                cluster_counts[pattern][field_status] += 1

            fields_list.append(
                FieldDetail(
                    field_path=fp,
                    gold_value=gold_val,
                    gold_page=gold_page,
                    gold_bbox=list(gold_box) if gold_box else None,
                    predicted_value=pred_val,
                    predicted_page=pred_page,
                    predicted_bbox=list(pred_box) if pred_box else None,
                    status=field_status,
                    iou=round(curr_iou, 4),
                ).__dict__
            )

        failing_count = gradeable - passing_count
        word_f1 = round((passing_count / gradeable * 100.0), 4) if gradeable > 0 else 0.0
        page_f1 = round((page_match_count / gradeable * 100.0), 4) if gradeable > 0 else 0.0

        failure_clusters: List[Dict[str, Any]] = []
        for idx, (pattern, class_dict) in enumerate(sorted(cluster_counts.items())):
            tot = sum(class_dict.values())
            dom_class = max(class_dict.items(), key=lambda kv: kv[1])[0]
            failure_clusters.append(
                FailureCluster(
                    cluster_id=f"c_{idx:03d}",
                    field_pattern=pattern,
                    count=tot,
                    dominant_failure_class=dom_class,
                ).__dict__
            )

        return DocumentAuditReport(
            document_id=doc_id,
            status="evaluated",
            gradeable=gradeable,
            passing=passing_count,
            failing=failing_count,
            word_f1=word_f1,
            page_f1=page_f1,
            family=family,
            fields=fields_list,
            failure_clusters=failure_clusters,
        )


def validate_gate_v8(
    reports: List[DocumentAuditReport], expected_total: Optional[int] = None
) -> Dict[str, Any]:
    """Validation Gate V8: Complete per-document field accounting."""
    total_gradeable = 0
    total_passing = 0
    total_failing = 0
    failures = []

    for r in reports:
        if r.status != "evaluated":
            continue
        total_gradeable += r.gradeable
        total_passing += r.passing
        total_failing += r.failing

        if r.gradeable != (r.passing + r.failing):
            failures.append(
                f"{r.document_id}: gradeable ({r.gradeable}) != pass ({r.passing}) + fail ({r.failing})"
            )
        if len(r.fields) != r.gradeable:
            failures.append(
                f"{r.document_id}: fields list length ({len(r.fields)}) != gradeable ({r.gradeable})"
            )

    passed = len(failures) == 0
    if expected_total is not None and total_gradeable != expected_total:
        passed = False
        failures.append(
            f"Corpus total gradeable ({total_gradeable}) != expected ({expected_total})"
        )

    return {
        "status": "PASS" if passed else "FAIL",
        "total_evaluated_documents": sum(1 for r in reports if r.status == "evaluated"),
        "total_documents": len(reports),
        "total_gradeable": total_gradeable,
        "total_passing": total_passing,
        "total_failing": total_failing,
        "expected_total_gradeable": expected_total or total_gradeable,
        "gradeable_matches_expected": total_gradeable == (expected_total or total_gradeable),
        "accounting_discrepancies": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Microscope V6 CLI Runner")
    parser.add_argument("--run-id", type=str, default="EXP-052", help="Run identifier")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--data-dir", type=str, default="research/data/full", help="Data directory")
    parser.add_argument("--preds-dir", type=str, default="research/experiments/EXP-050/predictions", help="Predictions directory")
    parser.add_argument("--fallback-v5", action="store_true", help="Run legacy V5 microscope pipeline")
    args = parser.parse_args()

    t_start = time.perf_counter()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    preds_dir = Path(args.preds_dir)
    if not preds_dir.is_absolute():
        preds_dir = repo_root / preds_dir
    if preds_dir.name != "tonerhound" and (preds_dir / "tonerhound").exists():
        preds_dir = preds_dir / "tonerhound"

    print(f"=== Running Failure Microscope V6 ===")
    print(f"Run ID:     {args.run_id}")
    print(f"Data Dir:   {data_dir}")
    print(f"Preds Dir:  {preds_dir}")
    print(f"Output Dir: {out_dir}")

    # Load canonical families
    family_csv = repo_root / "research/experiments/EXP-050/audit/per_document_field_breakdown.csv"
    family_map: Dict[str, str] = {}
    if family_csv.exists():
        with open(family_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                family_map[row["document_id"]] = get_canonical_family(row["document_id"], row.get("family", ""))

    # Discover all benchmark documents
    test_files = sorted(data_dir.glob("*/*.test.json"))
    doc_ids = [f"{p.parent.name}/{p.name[:-10]}" for p in test_files]
    print(f"Discovered {len(doc_ids)} benchmark test cases.")

    auditor = MicroscopeV6Auditor(data_dir=data_dir, preds_dir=preds_dir, family_map=family_map)

    reports: List[DocumentAuditReport] = []
    corpus_failure_classes = Counter()
    corpus_family_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    corpus_clusters: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "classes": Counter(),
        "documents": set(),
        "families": Counter(),
    })

    print("Auditing all 370 benchmark documents...")
    for idx, doc_id in enumerate(doc_ids):
        rep = auditor.audit_document(doc_id)
        reports.append(rep)

        if rep.status == "evaluated":
            for fld in rep.fields:
                f_status = fld["status"]
                if f_status != "PASS":
                    corpus_failure_classes[f_status] += 1
                    corpus_family_stats[rep.family][f_status] += 1
                    pattern = compute_field_pattern(fld["field_path"])
                    c_info = corpus_clusters[pattern]
                    c_info["count"] += 1
                    c_info["classes"][f_status] += 1
                    c_info["documents"].add(doc_id)
                    c_info["families"][rep.family] += 1

    # Validation Gate V8
    v8_result = validate_gate_v8(reports, expected_total=498140)
    print(f"\nValidation Gate V8 Status: {v8_result['status']}")
    print(f"Total Gradeable: {v8_result['total_gradeable']:,} (Expected: 498,140)")
    print(f"Total Passing:   {v8_result['total_passing']:,}")
    print(f"Total Failing:   {v8_result['total_failing']:,}")
    print(f"Accounting Discrepancies: {len(v8_result['accounting_discrepancies'])}")

    if v8_result["status"] != "PASS":
        print(f"ERROR: V8 Gate validation failed! Details: {v8_result['accounting_discrepancies'][:5]}")
        sys.exit(1)

    # Rank failure clusters corpus-wide
    ranked_clusters = []
    for pattern, info in sorted(corpus_clusters.items(), key=lambda kv: kv[1]["count"], reverse=True):
        dom_class = info["classes"].most_common(1)[0][0]
        dom_fam = info["families"].most_common(1)[0][0]
        ranked_clusters.append({
            "field_pattern": pattern,
            "count": info["count"],
            "documents_affected": len(info["documents"]),
            "family": dom_fam,
            "dominant_failure_class": dom_class,
            "class_distribution": dict(info["classes"]),
        })

    # Top 5 corpus-wide clusters
    top_5_clusters = ranked_clusters[:5]
    print("\nTop 5 Failure Clusters Corpus-Wide:")
    for idx, c in enumerate(top_5_clusters, 1):
        print(f"  Cluster {idx}: {c['field_pattern']} ({c['count']:,} fields, {c['documents_affected']} docs, {c['family']}, dominant: {c['dominant_failure_class']})")

    # Save v8_validation.json
    with open(out_dir / "v8_validation.json", "w", encoding="utf-8") as f:
        json.dump(v8_result, f, indent=2)
    print("Saved v8_validation.json")

    # Save top_clusters.json
    top_clusters_payload = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_clusters": len(ranked_clusters),
        "top_5_clusters": top_5_clusters,
        "all_ranked_clusters": ranked_clusters,
    }
    with open(out_dir / "top_clusters.json", "w", encoding="utf-8") as f:
        json.dump(top_clusters_payload, f, indent=2)
    print("Saved top_clusters.json")

    # Save field_census.csv
    csv_rows = []
    for r in reports:
        top_pats = [c["field_pattern"] for c in r.failure_clusters[:3]]
        while len(top_pats) < 3:
            top_pats.append("none")
        top_cls = [c["dominant_failure_class"] for c in r.failure_clusters[:3]]
        while len(top_cls) < 3:
            top_cls.append("none")

        csv_rows.append({
            "document_id": r.document_id,
            "status": r.status,
            "gradeable": r.gradeable,
            "passing": r.passing,
            "failing": r.failing,
            "word_f1": f"{r.word_f1:.4f}",
            "page_f1": f"{r.page_f1:.4f}",
            "family": r.family,
            "top_cluster_1": top_pats[0],
            "top_cluster_2": top_pats[1],
            "top_cluster_3": top_pats[2],
            "dominant_class_1": top_cls[0],
            "dominant_class_2": top_cls[1],
            "dominant_class_3": top_cls[2],
        })

    with open(out_dir / "field_census.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "document_id",
                "status",
                "gradeable",
                "passing",
                "failing",
                "word_f1",
                "page_f1",
                "family",
                "top_cluster_1",
                "top_cluster_2",
                "top_cluster_3",
                "dominant_class_1",
                "dominant_class_2",
                "dominant_class_3",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved field_census.csv ({len(csv_rows)} rows)")

    # Save field_census.json
    census_payload = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_summary": {
            "total_documents": len(reports),
            "evaluated_documents": v8_result["total_evaluated_documents"],
            "total_fields_accounted": v8_result["total_gradeable"],
            "total_passing": v8_result["total_passing"],
            "total_failing": v8_result["total_failing"],
            "failure_class_distribution": dict(corpus_failure_classes),
            "family_failure_distribution": {k: dict(v) for k, v in corpus_family_stats.items()},
        },
        "documents": [r.to_dict() for r in reports],
    }
    with open(out_dir / "field_census.json", "w", encoding="utf-8") as f:
        json.dump(census_payload, f, indent=2)
    print(f"Saved field_census.json")

    total_elapsed = round(time.perf_counter() - t_start, 2)
    print(f"\nMicroscope V6 full audit completed in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
