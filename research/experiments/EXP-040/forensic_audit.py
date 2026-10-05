"""EXP-040 Phase A: Microscope Forensic Audit on Top 3 Classes.

Collects 500 field samples each for:
- REAL_INDEXING_MISS
- NORMALIZATION_MISMATCH
- NO_TEXT_AT_GOLD_REGION

Extracts 15 detailed attributes per field and classifies into fine-grained subclasses.
Saves research/experiments/EXP-040/forensic_audit.json.
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
exp039_dir = repo_root / "research" / "experiments" / "EXP-039"
exp040_dir = repo_root / "research" / "experiments" / "EXP-040"

for p in [str(repo_root), str(repo_root / "src"), str(exp039_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

import fitz  # PyMuPDF
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
from tonerhound.document.index import DocumentIndex


def get_value_type(val: Any) -> str:
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "numeric"
    if isinstance(val, str):
        val_clean = val.strip().lower()
        if val_clean in ("true", "false", "yes", "no"):
            return "boolean"
        # Check date pattern
        if re.search(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b", val) or re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", val_clean
        ):
            return "date"
        # Check numeric string
        val_no_num = re.sub(r"[\$,\s%()]", "", val_clean)
        try:
            float(val_no_num)
            return "numeric"
        except ValueError:
            pass
        return "string"
    return "other"


def classify_real_indexing_miss_subclass(
    val: Any,
    gold_bbox: list[float] | None,
    pred_bbox: list[float] | None,
    same_page: bool,
    tokens_text: str,
) -> str:
    val_str = str(val).strip()
    val_clean = val_str.lower()

    # SUB_SHORT_VALUE (< 3 characters)
    if len(val_str) < 3:
        return "SUB_SHORT_VALUE"

    # Drift checks if prediction exists on same page
    if same_page and pred_bbox and gold_bbox and len(pred_bbox) == 4 and len(gold_bbox) == 4:
        gx, gy, gw, gh = gold_bbox
        px, py, pw, ph = pred_bbox
        y_diff = abs(py - gy)
        x_diff = abs(px - gx)
        # Column drift: same row band, horizontal mismatch
        if y_diff <= max(gh, 0.01) * 1.5 and x_diff > max(gw, 0.01) * 0.7:
            return "SUB_COLUMN_DRIFT"
        # Row drift: same column rail, vertical mismatch
        if x_diff <= max(gw, 0.01) * 1.5 and y_diff > max(gh, 0.01) * 0.7:
            return "SUB_ROW_DRIFT"

    # Numeric ambiguity (repeated zeros or common numbers)
    val_no_sym = re.sub(r"[\$,\s]", "", val_clean)
    if val_no_sym in ("0", "0.0", "0.00", "0.000", "-", "—", "none", "1", "1.0", "1.00"):
        return "SUB_NUMERIC_AMBIGUITY"

    # Multi-token (multiple words separated by whitespace)
    if len(val_str.split()) > 1:
        return "SUB_MULTI_TOKEN"

    # Special characters
    if any(c in val_str for c in ["/", "-", "(", ")", ".", ",", ":", ";", "$", "%", "&", "#", "@"]):
        return "SUB_SPECIAL_CHARS"

    # Long value (> 20 characters)
    if len(val_str) > 20:
        return "SUB_LONG_VALUE"

    return "SUB_OTHER"


def classify_normalization_mismatch_subclass(val: Any, field_path: str, tokens_text: str) -> str:
    val_str = str(val).strip()
    fp_lower = field_path.lower()

    # Currency prefix
    if re.search(r"\b(USD|EUR|GBP|JPY|CAD|AUD)\b", val_str, re.I) or any(
        c in val_str for c in ["$", "€", "£", "¥", "₹"]
    ):
        return "SUB_CURRENCY_PREFIX"

    # Dash as zero
    if val_str in ("-", "—", "–", "$-", "($-)") or (val in (0, 0.0) and any(d in tokens_text for d in ("-", "—"))):
        return "SUB_DASH_AS_ZERO"

    # Percent vs decimal
    if "%" in val_str or any(k in fp_lower for k in ("_percent", "_pct", "rate")):
        return "SUB_PERCENT_DECIMAL"

    # Tilde marker
    if val_str.startswith("~") or val_str.startswith("≈") or "~" in tokens_text:
        return "SUB_TILDE_MARKER"

    # Rounded value ($1.2M, 500K)
    if re.search(r"\b\d+(\.\d+)?[kmbKMB]\b", val_str):
        return "SUB_ROUNDED_VALUE"

    # Trailing punctuation
    if val_str.endswith((".", ",", ";", ":")):
        return "SUB_TRAILING_PUNCT"

    return "SUB_OTHER"


def classify_no_text_subclass(
    doc: fitz.Document, page_num: int, gold_bbox: list[float] | None, field_path: str
) -> str:
    if page_num < 1 or page_num > len(doc):
        return "SUB_OTHER"

    page = doc[page_num - 1]

    # Check page rotation
    if page.rotation != 0:
        return "SUB_ROTATED_TEXT"

    # Check if page has vector drawings but few text words
    drawings = page.get_drawings()
    page_text = page.get_text().strip()
    if len(drawings) > 10 and len(page_text) < 50:
        return "SUB_VECTOR_ONLY"

    # Multi-region: array or combined fields
    if "[" in field_path and "]" in field_path:
        return "SUB_MULTI_REGION"

    # Check faint scan: render small crop around gold bbox
    if gold_bbox and len(gold_bbox) == 4:
        gx, gy, gw, gh = gold_bbox
        rect = fitz.Rect(gx * page.rect.width, gy * page.rect.height, (gx + gw) * page.rect.width, (gy + gh) * page.rect.height)
        if rect.is_valid and not rect.is_empty:
            pix = page.get_pixmap(clip=rect, dpi=100)
            if pix.width > 0 and pix.height > 0:
                # If pixmap is nearly blank white
                samples = list(pix.samples)
                if samples:
                    mean_val = sum(samples) / len(samples)
                    if mean_val < 250:  # ink detected visually
                        return "SUB_FAINT_SCAN"

    return "SUB_OTHER"


def run_forensic_audit() -> dict[str, Any]:
    print("=== EXP-040 Phase A: Forensic Audit on Top 3 Classes ===")
    t0 = time.perf_counter()

    data_dir = repo_root / "research" / "data" / "full"
    exp039_preds_dir = exp039_dir / "predictions" / "tonerhound"

    pred_files = sorted(list(exp039_preds_dir.glob("**/*.result.json")))
    print(f"Loaded {len(pred_files)} prediction files from EXP-039.")

    target_classes = ["REAL_INDEXING_MISS", "NORMALIZATION_MISMATCH", "NO_TEXT_AT_GOLD_REGION"]
    samples_by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in target_classes}

    max_samples = 500

    # Iterate over files to collect 500 samples per class
    for pidx, pf in enumerate(pred_files):
        tid = pf.relative_to(exp039_preds_dir).as_posix().removesuffix(".result.json")
        pdf_p = data_dir / f"{tid}.pdf"
        if not pdf_p.exists():
            continue

        with open(pf, encoding="utf-8") as f:
            pred_data = json.load(f)

        pred_cits = {
            c["field_path"]: c
            for c in pred_data.get("output", {}).get("field_citations", [])
            if c.get("field_path") and c.get("page") is not None
        }

        tc = load_test_case(pdf_p)
        rules = tc.get_extract_field_rules()
        if not rules:
            continue

        alt_values, ev_boxes, ev_pages, normalizers = build_rule_indexes(rules)

        doc_idx: DocumentIndex | None = None
        fitz_doc: fitz.Document | None = None
        classifier: FailureMicroscopeClassifier | None = None

        doc_type = tid.split("/")[0] if "/" in tid else "unknown"

        for r in rules:
            fp = r.field_path
            boxes = ev_boxes.get(fp, [])
            if not boxes:
                continue

            gp, gb = boxes[0]
            val = r.evidence[0].value if r.evidence else None

            # Check IoU with prediction
            pc = pred_cits.get(fp)
            iou = 0.0
            same_page = False
            pred_b = None
            pred_p = None
            if pc and pc.get("page") is not None:
                pred_p = pc["page"]
                pred_b = pc.get("bbox")
                same_page = (pred_p == gp)
                if same_page and pred_b:
                    iou = compute_iou_xywh(gb, pred_b)

            if iou >= 0.50:
                continue  # Passing field, not a failure

            # Field is failing! Lazy-load classifier and fitz doc
            if doc_idx is None:
                doc_idx = DocumentIndex.from_pdf(pdf_p)
                classifier = FailureMicroscopeClassifier(doc_index=doc_idx)
                fitz_doc = fitz.open(pdf_p)

            # Classify failure
            res = classifier.classify_field(
                document_id=tid,
                field_path=fp,
                gold_value=val,
                predicted_value=None,
                gold_page=gp,
                predicted_page=pred_p,
                gold_bbox=gb,
                predicted_bbox=pred_b,
                experiment_run="exp039",
            )

            fclass = res.failure_class
            if fclass in target_classes and len(samples_by_class[fclass]) < max_samples:
                tokens_in_gold = classifier.get_tokens_in_region(doc_idx, gp, gb)
                tok_text = " ".join(t.text for t in tokens_in_gold).strip()

                val_len = len(str(val)) if val is not None else 0
                val_type = get_value_type(val)
                gb_w = round(gb[2], 4) if gb and len(gb) == 4 else 0.0
                gb_h = round(gb[3], 4) if gb and len(gb) == 4 else 0.0

                # Determine fine-grained subclass
                if fclass == "REAL_INDEXING_MISS":
                    subclass = classify_real_indexing_miss_subclass(
                        val=val,
                        gold_bbox=gb,
                        pred_bbox=pred_b,
                        same_page=same_page,
                        tokens_text=tok_text,
                    )
                elif fclass == "NORMALIZATION_MISMATCH":
                    subclass = classify_normalization_mismatch_subclass(
                        val=val, field_path=fp, tokens_text=tok_text
                    )
                else:  # NO_TEXT_AT_GOLD_REGION
                    subclass = classify_no_text_subclass(
                        doc=fitz_doc, page_num=gp, gold_bbox=gb, field_path=fp
                    )

                samples_by_class[fclass].append({
                    "document_id": tid,
                    "field_path": fp,
                    "gold_value": str(val) if val is not None else None,
                    "gold_bbox": gb,
                    "gold_page": gp,
                    "predicted_bbox": pred_b,
                    "predicted_page": pred_p,
                    "candidate_count": res.candidate_count,
                    "tokens_in_gold_count": len(tokens_in_gold),
                    "tokens_in_gold_text": tok_text,
                    "gold_bbox_width": gb_w,
                    "gold_bbox_height": gb_h,
                    "value_length": val_len,
                    "value_type": val_type,
                    "document_type": doc_type,
                    "subclass": subclass,
                })

        if fitz_doc:
            fitz_doc.close()

        # Check if all classes have reached 500 samples
        if all(len(samples_by_class[c]) >= max_samples for c in target_classes):
            print(f"Reached 500 samples for all top 3 classes at doc {pidx+1}.")
            break

    # Build subclass distribution report
    post_exp039_totals = {
        "REAL_INDEXING_MISS": 84826,
        "NORMALIZATION_MISMATCH": 46158,
        "NO_TEXT_AT_GOLD_REGION": 15382,
    }

    audit_payload: dict[str, Any] = {}

    for cname in target_classes:
        samps = samples_by_class[cname]
        sub_counts = Counter(s["subclass"] for s in samps)
        total_samps = len(samps)
        total_class_fields = post_exp039_totals.get(cname, total_samps)

        sub_dist: dict[str, Any] = {}
        for sub, cnt in sub_counts.most_common():
            pct = round((cnt / total_samps) * 100, 2)
            est_total = int(round((cnt / total_samps) * total_class_fields))
            sub_dist[sub] = {
                "sample_count": cnt,
                "sample_pct": f"{pct}%",
                "estimated_total_fields": est_total,
            }

        largest_sub = sub_counts.most_common(1)[0][0] if sub_counts else "UNKNOWN"
        largest_cnt = sub_counts[largest_sub] if sub_counts else 0
        largest_est_total = int(round((largest_cnt / max(1, total_samps)) * total_class_fields))

        audit_payload[cname] = {
            "total": total_class_fields,
            "sample_size": total_samps,
            "subclass_distribution": sub_dist,
            "largest_subclass": largest_sub,
            "largest_subclass_sample_count": largest_cnt,
            "largest_subclass_count": largest_est_total,
            "sample_records": samps[:5],  # 5 illustrative examples
        }

    out_file = exp040_dir / "forensic_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    dt = time.perf_counter() - t0
    print(f"Forensic audit complete in {dt:.1f}s.")
    print("Forensic Audit Findings:")
    for cname in target_classes:
        info = audit_payload[cname]
        print(f"  {cname} (Total: {info['total']:,}):")
        print(f"    Largest Subclass: {info['largest_subclass']} ({info['largest_subclass_count']:,} fields)")
        for sub, sinfo in info["subclass_distribution"].items():
            print(f"      - {sub}: {sinfo['sample_count']}/{info['sample_size']} ({sinfo['sample_pct']}) ~ {sinfo['estimated_total_fields']:,} fields")

    return audit_payload


if __name__ == "__main__":
    run_forensic_audit()
