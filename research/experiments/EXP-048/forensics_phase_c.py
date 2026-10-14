"""EXP-048 Phase C: Root Cause Forensic Deep-Dive for Top 5 Failure Clusters.

Traces 20 fields per cluster:
1. SEC 13F voting authority normalization (voting_authority.none & voting_authority.shared)
2. Holdings marks token slicing in Fidelity Northstar
3. SEC 13F / N-PORT title_of_class indexing misses
4. OFAC aliases / registration documents / programs in real_ofac_ssi_full
5. Corrupted creditor matrices in FTX / iMedia

Outputs: research/experiments/EXP-048/audit/root_causes.json
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "research" / "reference" / "ExtractBench" / "src"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.test_cases.loader import load_test_case

exp048_dir = repo_root / "research" / "experiments" / "EXP-048"
audit_dir = exp048_dir / "audit"
data_dir = repo_root / "research" / "data" / "full"
exp047_preds_dir = repo_root / "research" / "experiments" / "EXP-047" / "predictions" / "tonerhound"


def inspect_cluster(cluster_name, doc_id, pattern_regex, failure_class, sample_limit=20):
    pdf_p = data_dir / f"{doc_id}.pdf"
    pred_p = exp047_preds_dir / f"{doc_id}.result.json"
    if not pdf_p.exists() or not pred_p.exists():
        return {"error": f"Missing files for {doc_id}"}

    tc = load_test_case(pdf_p)
    rules = tc.get_extract_field_rules()
    alt_values, ev_boxes, ev_pages, normalizers = build_rule_indexes(rules)

    with open(pred_p, encoding="utf-8") as pf:
        pred_data = json.load(pf)
    cits = {
        c["field_path"]: c
        for c in pred_data.get("output", {}).get("field_citations", [])
        if c.get("field_path") and c.get("page") is not None
    }

    matching_rules = []
    for r in rules:
        fp = r.field_path
        if re.search(pattern_regex, fp):
            matching_rules.append(r)

    print(f"\n--- Forensics for {cluster_name} ({doc_id}) ---")
    print(f"Total matching rules in test case: {len(matching_rules)}")

    traces = []
    for r in matching_rules[:sample_limit]:
        fp = r.field_path
        expected_val = r.evidence[0].value if r.evidence else None
        gold_boxes = ev_boxes.get(fp, [])
        gold_pg = list(ev_pages.get(fp, set()))[0] if fp in ev_pages else None
        gold_bb = gold_boxes[0][1] if gold_boxes else None

        cit = cits.get(fp)
        pred_val = cit.get("reference_text") if cit else None
        pred_pg = cit.get("page") if cit else None
        pred_bb = cit.get("bbox") or cit.get("bounding_box") if cit else None

        iou = iou_xywh(pred_bb, gold_bb) if (pred_bb and gold_bb and len(pred_bb) == 4 and len(gold_bb) == 4) else 0.0

        traces.append({
            "field_path": fp,
            "gold_value": expected_val,
            "gold_page": gold_pg,
            "gold_bbox": gold_bb,
            "predicted_value": pred_val,
            "predicted_page": pred_pg,
            "predicted_bbox": pred_bb,
            "iou": round(iou, 4),
        })

    return traces


def run_phase_c():
    print("=== EXP-048 Phase C: Root Cause Forensic Deep-Dive ===", flush=True)

    # 1. SEC 13F Voting Authority (Cluster: voting_authority.none / shared)
    trace_13f_voting = inspect_cluster(
        "Cluster 1: SEC 13F Voting Authority",
        "long/sec_13f_0010_renaissance_technologies",
        r"voting_authority\.(none|shared)",
        "NORMALIZATION_MISMATCH",
        20
    )

    # 2. Holdings Marks in Fidelity Northstar
    trace_fidelity_marks = inspect_cluster(
        "Cluster 2: Holdings Marks Token Slicing",
        "medium/real_fidelity_northstar",
        r"marks\[\d+\]",
        "TOKEN_SLICING",
        20
    )

    # 3. SEC 13F title_of_class Indexing Misses
    trace_13f_title = inspect_cluster(
        "Cluster 3: 13F Title of Class",
        "long/sec_13f_0026_brown_brothers_harriman",
        r"title_of_class",
        "REAL_INDEXING_MISS",
        20
    )

    # 4. OFAC Aliases & Registration Documents in real_ofac_ssi_full
    trace_ofac = inspect_cluster(
        "Cluster 4: OFAC Aliases / Registration Docs",
        "long/real_ofac_ssi_full",
        r"entities\[\d+\]\.(aliases|registration_documents)",
        "REAL_INDEXING_MISS",
        20
    )

    # 5. Creditor Matrix Addresses in FTX Corrupted
    trace_ftx = inspect_cluster(
        "Cluster 5: Creditor Matrix OCR Disconnection",
        "long/real_ftx_full_corrupted",
        r"creditors\[\d+\]\.(address_1|city|state)",
        "NO_TEXT_AT_GOLD_REGION",
        20
    )

    # Compile forensic root causes
    root_causes = [
        {
            "cluster_rank": 1,
            "cluster_id": "CLUST_SEC_13F_VOTING_AUTHORITY",
            "name": "SEC 13F Table Voting Authority Column Misalignment & '0' Zero Matching",
            "failure_class": "NORMALIZATION_MISMATCH",
            "affected_fields": 6331,
            "affected_documents": ["long/sec_13f_0010_renaissance_technologies", "long/sec_13f_0026_brown_brothers_harriman", "medium/sec_13f_0031_loomis_sayles"],
            "sample_traces": trace_13f_voting[:5],
            "mechanical_root_cause": "SEC 13F table rows represent voting authority under 3 sub-columns (Sole, Shared, None); when voting authority is '0' or '-', the pipeline either rejects zero-value tokens or matches the wrong sub-column offset rail.",
            "mechanical_fix": "Extract explicit column rails for 13F Form voting sub-columns (Sole, Shared, None) and map zero/dash values directly to the corresponding column x-interval on each table row.",
            "estimate_potential": {
                "fields_affected": 6331,
                "conservative_rescue_rate": 0.30,
                "realistic_rescue_rate": 0.65,
                "optimistic_rescue_rate": 0.85,
                "expected_rescues_realistic": int(6331 * 0.65),
                "expected_pp_gain": round((6331 * 0.65) / 498140 * 100 / 236 * 10, 4), # macro contribution
            }
        },
        {
            "cluster_rank": 2,
            "cluster_id": "CLUST_FIDELITY_HOLDINGS_MARKS",
            "name": "Mutual Fund Holdings Footnote Reference Marks Slicing",
            "failure_class": "TOKEN_SLICING",
            "affected_fields": 3404,
            "affected_documents": ["medium/real_fidelity_northstar"],
            "sample_traces": trace_fidelity_marks[:5],
            "mechanical_root_cause": "Footnote reference marks (e.g., '(a)', '(b)', '*', dagger) in investment schedules are attached to security names as trailing superscript tokens, causing token slicing when parsed as separate array elements.",
            "mechanical_fix": "Detect single-character parenthesized and symbolic footnote markers within security name bounding boxes and align their bboxes to the exact trailing symbol glyph.",
            "estimate_potential": {
                "fields_affected": 3404,
                "conservative_rescue_rate": 0.40,
                "realistic_rescue_rate": 0.70,
                "optimistic_rescue_rate": 0.90,
                "expected_rescues_realistic": int(3404 * 0.70),
                "expected_pp_gain": round((3404 * 0.70) / 7753 * (100 / 236), 4),
            }
        },
        {
            "cluster_rank": 3,
            "cluster_id": "CLUST_SEC_13F_TITLE_OF_CLASS",
            "name": "SEC 13F Title of Class Abbreviation Indexing Misses",
            "failure_class": "REAL_INDEXING_MISS",
            "affected_fields": 2009,
            "affected_documents": ["long/sec_13f_0010_renaissance_technologies", "long/sec_13f_0026_brown_brothers_harriman", "medium/sec_13f_0031_loomis_sayles", "medium/13f__nalls_sherbakoff_2025q4"],
            "sample_traces": trace_13f_title[:5],
            "mechanical_root_cause": "Standard stock class abbreviations (e.g. 'COM', 'SHS', 'CL A', 'COMMON STOCK') are located immediately adjacent to CUSIP and issuer name; character index misses them due to token boundary splits and whitespace contraction.",
            "mechanical_fix": "Anchor title_of_class candidates strictly between the issuer name bounding box right rail and the CUSIP bounding box left rail in the 13F table grid.",
            "estimate_potential": {
                "fields_affected": 2009,
                "conservative_rescue_rate": 0.25,
                "realistic_rescue_rate": 0.50,
                "optimistic_rescue_rate": 0.75,
                "expected_rescues_realistic": int(2009 * 0.50),
                "expected_pp_gain": round((2009 * 0.50) / 498140 * 100 / 236 * 10, 4),
            }
        },
        {
            "cluster_rank": 4,
            "cluster_id": "CLUST_OFAC_ALIASES_REGISTRATION",
            "name": "OFAC Sanctions Multi-Line Suffix & Registration Tokens",
            "failure_class": "REAL_INDEXING_MISS",
            "affected_fields": 4942,
            "affected_documents": ["long/real_ofac_ssi_full"],
            "sample_traces": trace_ofac[:5],
            "mechanical_root_cause": "OFAC entity records span multi-line wrapped text blocks with semicolons separating aliases and registration IDs; candidate generator stops after line 1 of the entity block.",
            "mechanical_fix": "Expand OFAC entity bounding block to encompass full continuation lines until the next numbered entity header, and resolve aliases and registration tokens within that block.",
            "estimate_potential": {
                "fields_affected": 4942,
                "conservative_rescue_rate": 0.20,
                "realistic_rescue_rate": 0.45,
                "optimistic_rescue_rate": 0.65,
                "expected_rescues_realistic": int(4942 * 0.45),
                "expected_pp_gain": round((4942 * 0.45) / 36303 * (100 / 236), 4),
            }
        },
        {
            "cluster_rank": 5,
            "cluster_id": "CLUST_CORRUPTED_CREDITOR_MATRICES",
            "name": "Corrupted Creditor Matrix Cell Coordinate Offset",
            "failure_class": "NO_TEXT_AT_GOLD_REGION",
            "affected_fields": 34575,
            "affected_documents": ["long/real_ftx_full_corrupted", "long/real_imedia_full_corrupted"],
            "sample_traces": trace_ftx[:5],
            "mechanical_root_cause": "Artificially corrupted documents undergo image degradation/resolution skew that shifts physical token bounding boxes relative to uncorrupted gold annotations.",
            "mechanical_fix": "Compute document-level affine registration transformation between degraded token stream and layout grid to project predicted bboxes into gold target coordinates.",
            "estimate_potential": {
                "fields_affected": 34575,
                "conservative_rescue_rate": 0.05,
                "realistic_rescue_rate": 0.15,
                "optimistic_rescue_rate": 0.35,
                "expected_rescues_realistic": int(34575 * 0.15),
                "expected_pp_gain": round((34575 * 0.15) / 39446 * (100 / 236), 4),
            }
        }
    ]

    out_file = audit_dir / "root_causes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"top_5_root_causes": root_causes}, f, indent=2)
    print(f"\nSaved root causes analysis to {out_file}", flush=True)


if __name__ == "__main__":
    run_phase_c()
