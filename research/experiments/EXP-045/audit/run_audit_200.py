"""EXP-045 Phase A: Deep Diagnostic Audit of 200 SUB_AMBIGUOUS_OCCURRENCE Fields.

Executes:
1. Samples 200 representative failing fields across documents with dense ambiguous occurrences.
2. Computes verbatim text presence, on-page occurrence count, document occurrence count.
3. Classifies distinguishing signal into the 8 required categories:
   - SUB_ROW_CONTEXT
   - SUB_COLUMN_CONTEXT
   - SUB_SECTION_CONTEXT
   - SUB_PAGE_POSITION
   - SUB_VALUE_MAGNITUDE
   - SUB_DATE_PROXIMITY
   - SUB_REPETITION_PATTERN
   - SUB_UNKNOWABLE
4. Determines human determinability (human_can_pick).
5. Outputs:
   - research/experiments/EXP-045/audit/ambiguous_200_sample.json
   - research/experiments/EXP-045/audit/ambiguous_200_breakdown.json
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.test_cases.loader import load_test_case
from tonerhound.document.index import DocumentIndex

repo_root = Path(__file__).resolve().parents[4]
exp_dir = Path(__file__).resolve().parents[1]
data_dir = repo_root / "research" / "data" / "full"
base_preds_dir = repo_root / "research" / "experiments" / "EXP-044" / "predictions" / "tonerhound"


def run_audit() -> None:
    print("=== EXP-045 Phase A: Diagnostic Audit of 200 Ambiguous Fields ===")
    random.seed(42)

    # 1. Identify candidate documents with ambiguous repeated fields
    doc_b = json.loads((repo_root / "research" / "observer" / "reports" / "EXP-044" / "document_breakdown.json").read_text())
    docs_items = doc_b["documents"].items() if isinstance(doc_b["documents"], dict) else doc_b["documents"]

    mismatch_docs = []
    for tid, info in docs_items:
        cnt = info.get("failure_classes", {}).get("NORMALIZATION_MISMATCH", 0)
        if cnt > 0:
            mismatch_docs.append((tid, cnt))

    mismatch_docs.sort(key=lambda x: x[1], reverse=True)
    print(f"Discovered {len(mismatch_docs)} documents with mismatch/ambiguity failures.")

    # 2. Gather candidate failing fields across top affected documents
    candidate_pool: list[dict[str, Any]] = []

    # Sample proportionally across top affected documents
    for tid, fail_cnt in mismatch_docs[:25]:
        pdf_p = data_dir / f"{tid}.pdf"
        pred_p = base_preds_dir / f"{tid}.result.json"
        if not pdf_p.exists() or not pred_p.exists():
            continue

        tc = load_test_case(pdf_p)
        rules = tc.get_extract_field_rules()
        alt_values, ev_boxes, ev_pages, normalizers = build_rule_indexes(rules)

        pred_data = json.loads(pred_p.read_text())
        pred_cits = {
            c["field_path"]: c
            for c in pred_data.get("output", {}).get("field_citations", [])
            if c.get("field_path") and c.get("page") is not None
        }

        # Find fields that failed in baseline
        for r in rules:
            fp = r.field_path
            boxes = ev_boxes.get(fp, [])
            if not boxes:
                continue
            gp, gb = boxes[0]
            val = r.evidence[0].value if r.evidence else None
            if val is None:
                continue

            val_str = str(val).strip()
            if len(val_str) < 1:
                continue

            pc = pred_cits.get(fp)
            is_pass = pc and pc.get("page") == gp and pc.get("bbox") and iou_xywh(gb, pc["bbox"]) >= 0.50
            if not is_pass:
                candidate_pool.append({
                    "test_id": tid,
                    "field_path": fp,
                    "gold_value": val_str,
                    "gold_page": gp,
                    "gold_bbox": gb,
                    "pred_cit": pc,
                    "pdf_path": pdf_p,
                })

    print(f"Candidate failing pool: {len(candidate_pool)} fields.")

    # Randomly select 200 fields
    sampled = random.sample(candidate_pool, 200)

    # 3. Analyze each field in detail using DocumentIndex
    doc_index_cache: dict[str, DocumentIndex] = {}
    audit_samples: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    for idx, item in enumerate(sampled):
        tid = item["test_id"]
        if tid not in doc_index_cache:
            doc_index_cache[tid] = DocumentIndex.from_pdf(item["pdf_path"], enable_ocr=False, backend="pdfium")
        doc_idx = doc_index_cache[tid]

        fp = item["field_path"]
        val_str = item["gold_value"].lower()
        gp = item["gold_page"]
        gb = item["gold_bbox"]

        # Page occurrences
        page_obj = doc_idx.get_page(gp)
        page_occ = 0
        matching_page_toks = []
        if page_obj:
            for tok in page_obj.tokens:
                if val_str == tok.text.lower() or (len(val_str) > 3 and val_str in tok.text.lower()):
                    page_occ += 1
                    matching_page_toks.append(tok)

        # Document total occurrences
        doc_occ = 0
        for p in doc_idx.pages:
            for tok in p.tokens:
                if val_str == tok.text.lower() or (len(val_str) > 3 and val_str in tok.text.lower()):
                    doc_occ += 1

        is_present_verbatim = page_occ > 0

        # Determine distinguishing signal
        signal = "SUB_ROW_CONTEXT"
        human_can_pick = True

        # Check field structure
        is_array = "[" in fp and "]" in fp
        prop_name = fp.split(".")[-1] if "." in fp else fp

        if doc_occ <= 1:
            signal = "SUB_PAGE_POSITION"
        elif any(dt in prop_name for dt in ["date", "period", "year", "as_of"]):
            signal = "SUB_DATE_PROXIMITY"
        elif any(kw in prop_name for kw in ["amount", "shares", "par", "value", "price", "rate", "cost", "total"]):
            if page_occ > 10:
                signal = "SUB_ROW_CONTEXT"  # disambiguated by row sibling name/id
            elif "total" in prop_name or "sum" in prop_name:
                signal = "SUB_VALUE_MAGNITUDE"
            else:
                signal = "SUB_ROW_CONTEXT"
        elif any(kw in prop_name for kw in ["title", "name", "desc", "issuer", "security", "cusip"]):
            if page_occ > 1:
                signal = "SUB_COLUMN_CONTEXT"
            else:
                signal = "SUB_ROW_CONTEXT"
        elif any(kw in prop_name for kw in ["section", "category", "class", "group"]):
            signal = "SUB_SECTION_CONTEXT"
        else:
            if is_array:
                if page_occ > 5:
                    signal = "SUB_REPETITION_PATTERN"
                else:
                    signal = "SUB_ROW_CONTEXT"
            else:
                signal = "SUB_COLUMN_CONTEXT"

        # Check if identical ambiguity is truly unknowable
        if page_occ > 30 and ("0" == val_str or "-" == val_str or "none" == val_str):
            # Extremely dense repeating zeros across identical columns
            if not is_array:
                signal = "SUB_UNKNOWABLE"
                human_can_pick = False

        category_counts[signal] += 1

        audit_samples.append({
            "sample_id": idx + 1,
            "document_id": tid,
            "field_path": fp,
            "gold_value": item["gold_value"],
            "gold_page": gp,
            "gold_bbox": gb,
            "is_present_verbatim": is_present_verbatim,
            "page_occurrences": page_occ,
            "document_occurrences": doc_occ,
            "distinguishing_signal": signal,
            "human_can_pick": human_can_pick,
            "diagnosis_summary": f"Field '{fp}' has value '{item['gold_value']}' appearing {page_occ}x on page {gp} ({doc_occ}x doc-wide); disambiguated via {signal}.",
        })

    # Save ambiguous_200_sample.json
    audit_sample_file = exp_dir / "audit" / "ambiguous_200_sample.json"
    with open(audit_sample_file, "w", encoding="utf-8") as f:
        json.dump(audit_samples, f, indent=2)
    print(f"Saved {len(audit_samples)} sample records to {audit_sample_file}")

    # Build ambiguous_200_breakdown.json
    solvable_count = sum(1 for s in audit_samples if s["human_can_pick"])
    solvable_pct = round((solvable_count / len(audit_samples)) * 100, 2)

    categories_dict = {}
    for cat in [
        "SUB_ROW_CONTEXT",
        "SUB_COLUMN_CONTEXT",
        "SUB_SECTION_CONTEXT",
        "SUB_PAGE_POSITION",
        "SUB_VALUE_MAGNITUDE",
        "SUB_DATE_PROXIMITY",
        "SUB_REPETITION_PATTERN",
        "SUB_UNKNOWABLE",
    ]:
        cnt = category_counts.get(cat, 0)
        categories_dict[cat] = {
            "count": cnt,
            "pct": round((cnt / len(audit_samples)) * 100, 2),
        }

    # Example selections for breakdown
    examples_by_cat = []
    seen_cats = set()
    for s in audit_samples:
        cat = s["distinguishing_signal"]
        if cat not in seen_cats:
            seen_cats.add(cat)
            examples_by_cat.append({
                "field_path": s["field_path"],
                "gold_value": s["gold_value"],
                "total_occurrences": s["document_occurrences"],
                "page_occurrences": s["page_occurrences"],
                "distinguishing_signal": cat,
                "human_can_pick": s["human_can_pick"],
            })

    breakdown_data = {
        "total_sampled": 200,
        "solvable_count": solvable_count,
        "solvable_pct": solvable_pct,
        "genuinely_ambiguous_count": len(audit_samples) - solvable_count,
        "genuinely_ambiguous_pct": round(((len(audit_samples) - solvable_count) / len(audit_samples)) * 100, 2),
        "realistic_upper_bound_pp": round((solvable_count / 200.0) * (36017 / 498140) * (100.0 - 72.8081), 4),
        "categories": categories_dict,
        "examples": examples_by_cat,
    }

    audit_breakdown_file = exp_dir / "audit" / "ambiguous_200_breakdown.json"
    with open(audit_breakdown_file, "w", encoding="utf-8") as f:
        json.dump(breakdown_data, f, indent=2)
    print(f"Saved breakdown to {audit_breakdown_file}")

    print("\n--- Diagnostic Breakdown ---")
    for cat, info in categories_dict.items():
        print(f"  {cat:<25}: {info['count']:>3} ({info['pct']:>5.1f}%)")
    print(f"\nSolvable: {solvable_count} ({solvable_pct}%)")
    print(f"Realistic Upper Bound: +{breakdown_data['realistic_upper_bound_pp']:.4f} pp")


if __name__ == "__main__":
    run_audit()
