"""EXP-042 Phase A: Forensic Audit V2 across All Target Failure Classes.

Samples up to 500 failing fields per class from the EXP-041 baseline:
- HYPHENATION (7,623 fields)
- TOKEN_SLICING (23,542 fields)
- REAL_INDEXING_MISS (78,126 fields)
- DATE_INDEX_MISS (2,646 fields)
- NON_TEXT_BOOLEAN_GROUNDING (2,840 fields)
- NO_TEXT_AT_GOLD_REGION (14,973 fields)
- MULTI_LINE_SPLIT (563 fields)

Extracts attributes and fine-grained subclasses according to Section 5 specifications.
Saves to research/experiments/EXP-042/forensic_audit_v2.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

repo_root = Path(__file__).resolve().parent.parent.parent.parent
exp041_dir = repo_root / "research" / "experiments" / "EXP-041"
exp042_dir = repo_root / "research" / "experiments" / "EXP-042"

for p in [str(repo_root), str(repo_root / "src"), str(exp041_dir), str(exp042_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

import fitz
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.test_cases.loader import load_test_case
from research.observer.field_classifier import (
    FailureMicroscopeClassifier,
    compute_iou_xywh,
)
from tonerhound.document.hybrid_index import HybridDocumentIndex


def get_value_type(val: Any) -> str:
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "numeric"
    if isinstance(val, str):
        val_clean = val.strip().lower()
        if val_clean in ("true", "false", "yes", "no"):
            return "boolean"
        if re.search(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b", val) or re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", val_clean
        ):
            return "date"
        val_no_num = re.sub(r"[\$,\s%()]", "", val_clean)
        try:
            float(val_no_num)
            return "numeric"
        except ValueError:
            pass
        return "string"
    return "other"


def classify_subclass(failure_class: str, val: Any, gold_bbox: list[float], pred_bbox: list[float] | None, doc_id: str, tokens_text: str, field_path: str) -> str:
    val_str = str(val).strip() if val is not None else ""
    val_clean = val_str.lower()
    fn_lower = field_path.lower()

    if failure_class == "HYPHENATION":
        if "—" in tokens_text or "–" in tokens_text:
            return "SUB_EM_DASH"
        if "\u00ad" in tokens_text or "\u00ad" in val_str:
            return "SUB_SOFT_HYPHEN"
        if "-" in val_str or "-" in tokens_text:
            return "SUB_TRAILING_HYPHEN"
        return "SUB_OTHER"

    elif failure_class == "TOKEN_SLICING":
        if pred_bbox and gold_bbox:
            gw, gh = gold_bbox[2], gold_bbox[3]
            pw, ph = pred_bbox[2], pred_bbox[3]
            if pw > gw * 1.5:
                return "SUB_TIGHT_GOLD"
            if gw > pw * 1.5:
                return "SUB_WIDE_GOLD"
            if abs(pred_bbox[0] - gold_bbox[0]) > 0.05 and abs(pred_bbox[1] - gold_bbox[1]) < 0.02:
                return "SUB_COLUMN_BLEED"
        if "total" in fn_lower or "header" in fn_lower or "summary" in fn_lower:
            return "SUB_CELL_MERGE"
        return "SUB_COLUMN_BLEED"

    elif failure_class == "REAL_INDEXING_MISS":
        if len(val_str.split()) >= 2:
            return "SUB_MULTI_WORD"
        if len(val_str) < 3:
            return "SUB_SHORT_VALUE"
        if re.search(r"\d,\d", val_str):
            return "SUB_NUMERIC_COMMA"
        if re.search(r"[^\w\s]", val_str):
            return "SUB_SPECIAL_CHARS"
        return "SUB_OTHER"

    elif failure_class == "DATE_INDEX_MISS":
        if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
            return "SUB_ISO_DATE"
        if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", val_str):
            return "SUB_US_DATE"
        if re.match(r"^\d{1,2}-\d{1,2}-\d{2,4}$", val_str):
            return "SUB_EU_DATE"
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", val_clean):
            return "SUB_MONTH_NAME"
        if re.search(r"^\d{1,2}\s+[A-Z]{3}\s+\d{4}$", val_str):
            return "SUB_MILITARY"
        return "SUB_OTHER"

    elif failure_class == "NON_TEXT_BOOLEAN_GROUNDING":
        if "signature" in fn_lower or "sign" in fn_lower or "signed" in fn_lower:
            return "SUB_SIGNATURE"
        if "stamp" in fn_lower or "seal" in fn_lower:
            return "SUB_STAMP"
        if "grid" in fn_lower or "table" in fn_lower or "[" in field_path:
            return "SUB_GRID_CHECKBOX"
        return "SUB_LONE_CHECKBOX"

    elif failure_class == "NO_TEXT_AT_GOLD_REGION":
        if "corrupted" in doc_id.lower():
            return "SUB_CORRUPTED_SCAN"
        if "rotated" in doc_id.lower():
            return "SUB_ROTATED"
        if len(val_str.split()) >= 3:
            return "SUB_MULTI_REGION"
        if "faint" in fn_lower or "scan" in doc_id.lower():
            return "SUB_FAINT_INK"
        return "SUB_MULTI_REGION"

    elif failure_class == "MULTI_LINE_SPLIT":
        if "address" in fn_lower or "street" in fn_lower or "city" in fn_lower:
            return "SUB_ADDRESS"
        if "description" in fn_lower or "narrative" in fn_lower or "note" in fn_lower:
            return "SUB_NARRATIVE"
        if "[" in field_path:
            return "SUB_TABLE_WRAP"
        return "SUB_OTHER"

    return "SUB_OTHER"


def run_forensic_audit() -> dict[str, Any]:
    print("=== EXP-042 Phase A: Forensic Audit V2 across Target Failure Classes ===")
    t_start = time.perf_counter()

    data_dir = repo_root / "research" / "data" / "full"
    base_preds_dir = exp041_dir / "predictions" / "tonerhound"

    # Base failure summary from EXP-041
    fs_path = exp041_dir / "results" / "failure_summary.json"
    with open(fs_path, encoding="utf-8") as f:
        fs_data = json.load(f)

    target_totals = {c["failure_class"]: c["field_count"] for c in fs_data.get("classes", [])}

    target_classes = [
        "HYPHENATION",
        "TOKEN_SLICING",
        "REAL_INDEXING_MISS",
        "DATE_INDEX_MISS",
        "NON_TEXT_BOOLEAN_GROUNDING",
        "NO_TEXT_AT_GOLD_REGION",
        "MULTI_LINE_SPLIT",
    ]

    samples_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subclass_counts: dict[str, Counter] = defaultdict(Counter)

    all_pred_files = sorted(list(base_preds_dir.glob("**/*.result.json")))

    for pred_f in all_pred_files:
        # Check if all classes have 500 samples
        if all(len(samples_by_class[c]) >= 500 for c in target_classes if target_totals.get(c, 0) >= 500):
            break

        doc_id = pred_f.relative_to(base_preds_dir).as_posix().removesuffix(".result.json")
        pdf_path = data_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            continue

        with open(pred_f, encoding="utf-8") as pf:
            pred_data = json.load(pf)

        tc = load_test_case(pdf_path)
        rules = tc.get_extract_field_rules()
        if not rules:
            continue

        alt_values, ev_boxes, ev_pages, normalizers = build_rule_indexes(rules)
        doc_cits = {
            c["field_path"]: c
            for c in pred_data.get("output", {}).get("field_citations", [])
            if c.get("field_path") and c.get("page") is not None
        }

        # Initialize index & classifier
        doc_idx = HybridDocumentIndex.from_pdf(pdf_path, enable_ocr=False)
        classifier = FailureMicroscopeClassifier(doc_idx)

        for r in rules:
            fp = r.field_path
            boxes = ev_boxes.get(fp, [])
            if not boxes:
                continue

            gp, gb = boxes[0]
            val = r.evidence[0].value if r.evidence else None

            # Check if passing
            pc = doc_cits.get(fp)
            is_passing = False
            pred_b = None
            pred_p = None
            if pc and pc.get("bbox") and pc.get("page") == gp:
                iou_val = iou_xywh(gb, pc["bbox"])
                is_passing = iou_val >= 0.50
                pred_b = pc["bbox"]
                pred_p = pc.get("page")
            elif pc and pc.get("bbox"):
                pred_b = pc["bbox"]
                pred_p = pc.get("page")

            if is_passing:
                continue

            # Classify failure
            c_res = classifier.classify_field(
                document_id=doc_id,
                field_path=fp,
                gold_value=val,
                predicted_value=val,
                gold_page=gp,
                predicted_page=pred_p,
                gold_bbox=gb,
                predicted_bbox=pred_b,
            )
            f_class = c_res.failure_class

            if f_class not in target_classes:
                continue

            if len(samples_by_class[f_class]) >= 500:
                continue

            # Extract token text in gold
            p_obj = doc_idx.get_page(gp)
            toks_in_gold = []
            if p_obj:
                for tok in p_obj.tokens:
                    tb = tok.bbox.to_coco()
                    if compute_iou_xywh(gb, tb) > 0.1:
                        toks_in_gold.append(tok.text)

            toks_text = " ".join(toks_in_gold)
            subclass = classify_subclass(f_class, val, gb, pred_b, doc_id, toks_text, fp)
            subclass_counts[f_class][subclass] += 1

            record = {
                "document_id": doc_id,
                "field_path": fp,
                "gold_value": str(val) if val is not None else "",
                "gold_bbox": gb,
                "gold_page": gp,
                "predicted_bbox": pred_b,
                "predicted_page": pred_p,
                "candidate_count": 1 if pred_b else 0,
                "tokens_in_gold_count": len(toks_in_gold),
                "tokens_in_gold_text": toks_text[:100],
                "value_length": len(str(val)) if val is not None else 0,
                "value_type": get_value_type(val),
                "gold_bbox_width": round(gb[2], 4) if len(gb) == 4 else 0.0,
                "gold_bbox_height": round(gb[3], 4) if len(gb) == 4 else 0.0,
                "subclass": subclass,
            }
            samples_by_class[f_class].append(record)

    # Build final audit dictionary
    audit_v2 = {}
    for c in target_classes:
        total_in_class = target_totals.get(c, len(samples_by_class[c]))
        sample_list = samples_by_class[c]
        sample_sz = len(sample_list)
        counter = subclass_counts[c]

        dist = {}
        for sub_name, cnt in counter.most_common():
            pct = round((cnt / sample_sz * 100), 1) if sample_sz > 0 else 0.0
            est_total = int(round(total_in_class * (cnt / sample_sz))) if sample_sz > 0 else 0
            dist[sub_name] = {
                "sample_count": cnt,
                "pct": f"{pct}%",
                "estimated_total": est_total,
            }

        largest_sub = counter.most_common(1)[0][0] if counter else "SUB_OTHER"
        largest_est = dist.get(largest_sub, {}).get("estimated_total", 0)

        audit_v2[c] = {
            "total": total_in_class,
            "sample_size": sample_sz,
            "subclass_distribution": dist,
            "largest_subclass": largest_sub,
            "largest_subclass_count": largest_est,
            "sample_records": sample_list[:5],
        }

    out_file = exp042_dir / "forensic_audit_v2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_v2, f, indent=2)

    res_file = exp042_dir / "results" / "forensic_audit_v2.json"
    res_file.parent.mkdir(parents=True, exist_ok=True)
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(audit_v2, f, indent=2)

    t_elapsed = time.perf_counter() - t_start
    print(f"Forensic Audit V2 complete in {t_elapsed:.1f}s.")
    for c, data in audit_v2.items():
        print(f"  {c}: sampled {data['sample_size']}, largest subclass: {data['largest_subclass']} ({data['largest_subclass_count']} est fields)")

    return audit_v2


if __name__ == "__main__":
    run_forensic_audit()
