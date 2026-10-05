"""EXP-035: Exhaustive Pipeline Audit & Causal Classification of Alleged INDEXING_MISS Cases.

Traces all 572 alleged cases through the real production DocumentIndex,
EvidenceMatcher, CandidateRecoveryEngine, CandidateVerifier, and EvidenceResolver.
Determines exact failure root causes according to the forensic taxonomy.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput
from tonerhound.normalization.normalizers import (
    clean_currency_and_numbers,
    normalize_unicode_and_case,
    parse_date_value,
    parse_numeric_value,
)
from tonerhound.resolution.resolver import EvidenceResolver


def run_causal_audit():
    alleged_path = repo_root / "research" / "experiments" / "EXP-035" / "alleged_indexing_miss_fields.json"
    with open(alleged_path, encoding="utf-8") as f:
        alleged_data = json.load(f)

    cases = alleged_data["cases"]
    print(f"Loaded {len(cases)} alleged INDEXING_MISS cases across {alleged_data['unique_alleged_documents']} documents.")

    # Group cases by document for efficient indexing
    doc_cases = defaultdict(list)
    for c in cases:
        doc_cases[c["document_id"]].append(c)

    data_dir = repo_root / "research" / "data" / "full"

    results = []
    class_counts = defaultdict(int)
    class_docs = defaultdict(set)

    t0 = time.perf_counter()

    for d_idx, (doc_id, d_cases) in enumerate(doc_cases.items(), 1):
        pdf_path = data_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            print(f"Warning: PDF not found: {pdf_path}")
            continue

        # Load document index (OCR enabled only for corrupted documents to match production)
        is_corrupted = "corrupted" in doc_id
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=is_corrupted)
        resolver = EvidenceResolver(doc_index)

        for c in d_cases:
            fpath = c["field_path"]
            val = c["value"]
            gp = c["gold_page"]
            gb = c["gold_bbox"]

            # 1. Check for boolean checkbox (Wrong Classification)
            is_bool_val = isinstance(val, bool)
            is_bool_field = (
                is_bool_val
                or (
                    isinstance(val, str)
                    and val.strip().lower() in ("true", "false", "yes", "no")
                )
                or any(k in fpath.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no"))
            )

            if is_bool_val or (is_bool_field and isinstance(val, (bool, int))):
                classification = "WRONG_CLASSIFICATION"
                detail = "Boolean/checkbox form field misclassified as numeric by Python isinstance(bool, int)"
                record = {
                    "document_id": doc_id,
                    "field_path": fpath,
                    "value": val,
                    "gold_page": gp,
                    "gold_bbox": gb,
                    "failure_mechanism": classification,
                    "classification_detail": detail,
                    "max_candidate_iou": 0.0,
                    "resolved_iou": 0.0,
                }
                results.append(record)
                class_counts[classification] += 1
                class_docs[classification].add(doc_id)
                continue

            # 2. Check OCR coverage in target gold box
            page_obj = doc_index.get_page(gp)
            tokens_in_gold = []
            if page_obj:
                for t in page_obj.tokens:
                    # Spatial overlap check
                    if (
                        t.bbox.x < gb[0] + gb[2]
                        and t.bbox.x + t.bbox.width > gb[0]
                        and t.bbox.y < gb[1] + gb[3]
                        and t.bbox.y + t.bbox.height > gb[1]
                    ):
                        tokens_in_gold.append(t)

            if is_corrupted and not tokens_in_gold:
                classification = "OCR_CORRUPTION"
                detail = "Scanned/corrupted document with no OCR tokens in gold bbox"
                record = {
                    "document_id": doc_id,
                    "field_path": fpath,
                    "value": val,
                    "gold_page": gp,
                    "gold_bbox": gb,
                    "failure_mechanism": classification,
                    "classification_detail": detail,
                    "max_candidate_iou": 0.0,
                    "resolved_iou": 0.0,
                }
                results.append(record)
                class_counts[classification] += 1
                class_docs[classification].add(doc_id)
                continue

            # 3. Trace candidate generation
            inp = ExtractionInput(field=fpath, value=val, page_hint=gp)
            candidates = resolver.collect_candidates(inp)

            # Check candidate IoUs
            cand_ious = [
                (iou_xywh(cand.bbox.to_coco(), gb), cand)
                for cand in candidates
                if cand.page == gp
            ]
            max_cand_iou = max((ci[0] for ci in cand_ious), default=0.0)
            best_cand = max(cand_ious, key=lambda ci: ci[0])[1] if cand_ious else None

            # Trace resolver execution
            res = resolver.resolve(inp)
            res_iou = (
                iou_xywh(res.bbox.to_coco(), gb)
                if (res.is_grounded and res.page == gp and res.bbox is not None)
                else 0.0
            )

            # Reclassify based on actual pipeline behavior
            if res_iou >= 0.50:
                classification = "ALREADY_RESOLVED"
                detail = f"Production pipeline resolves field successfully with IoU={res_iou:.4f}"
            elif max_cand_iou >= 0.50:
                # Candidate existed with IoU >= 0.50!
                if res.is_grounded:
                    classification = "SELECTION"
                    detail = f"Candidate pool contained IoU={max_cand_iou:.4f}, but resolver selected different candidate with IoU={res_iou:.4f}"
                else:
                    classification = "VERIFICATION_REJECTION"
                    detail = f"Candidate pool contained IoU={max_cand_iou:.4f}, but verifier rejected candidate: {res.explanation}"
            elif candidates:
                # Candidates existed, but none achieved IoU >= 0.50
                # Inspect why candidates failed IoU
                if any(cand.page != gp for cand in candidates) and not cand_ious:
                    classification = "PAGE_ROUTING"
                    detail = f"Candidates found on other pages, but none on gold page {gp}"
                else:
                    classification = "BBOX_RECONSTRUCTION"
                    detail = f"Candidates found on page {gp}, but max IoU={max_cand_iou:.4f} < 0.50 due to boundary/span reconstruction"
            else:
                # ZERO candidates collected! Why?
                val_str = str(val)
                norm_val = normalize_unicode_and_case(val_str).text.strip()
                page_text = page_obj.text if page_obj else ""

                if "\n" in val_str:
                    classification = "MULTI_LINE"
                    detail = "Value contains newlines spanning multiple lines"
                elif isinstance(val, (int, float)) and val < 0:
                    # Parenthesized negative numbers in tax forms
                    classification = "NORMALIZATION_MISMATCH"
                    detail = f"Negative numeric value {val} formatted parenthetically in document"
                elif isinstance(val, (int, float)) and parse_numeric_value(val) is None:
                    classification = "NUMERIC_INDEX_MISS"
                    detail = f"Numeric value {val} could not be parsed by parse_numeric_value"
                elif parse_date_value(val_str) is not None and not resolver.matcher.find_normalized_date_candidates(val_str, page_hint=gp):
                    classification = "DATE_INDEX_MISS"
                    detail = f"Date value '{val_str}' parsed but not found in date index on page {gp}"
                elif any(ch in val_str for ch in ("•", "–", "—", "’", "“", "”", "®", "™", "\xa0")):
                    classification = "CHAR_ENCODING"
                    detail = f"Special character encoding / ligature mismatch in '{val_str[:30]}'"
                elif "-" in val_str and any(w in page_text for w in val_str.split("-")):
                    classification = "HYPHENATION"
                    detail = f"Hyphenated compound token split in document text"
                elif len(tokens_in_gold) == 0:
                    classification = "OCR_CORRUPTION"
                    detail = "No text tokens extracted by PDF text layer at gold coordinates"
                else:
                    gold_tokens_text = " ".join(t.text for t in tokens_in_gold)
                    if clean_currency_and_numbers(norm_val) in clean_currency_and_numbers(normalize_unicode_and_case(gold_tokens_text).text):
                        classification = "SUBTOKEN_BOUNDARY"
                        detail = f"Target text '{norm_val}' is a subtoken inside '{gold_tokens_text}'"
                    elif any(w.strip(" ,.;:") in [t.text.strip(" ,.;:") for t in tokens_in_gold] for w in norm_val.split()):
                        classification = "NORMALIZATION_MISMATCH"
                        detail = f"Tokens present at gold box ('{gold_tokens_text}'), but failed normalization match against '{norm_val}'"
                    else:
                        classification = "REAL_INDEXING_MISS"
                        detail = f"Text '{norm_val}' present on page but indexing/retrieval engine failed to locate it"

            record = {
                "document_id": doc_id,
                "field_path": fpath,
                "value": val,
                "gold_page": gp,
                "gold_bbox": gb,
                "failure_mechanism": classification,
                "classification_detail": detail,
                "max_candidate_iou": round(max_cand_iou, 4),
                "resolved_iou": round(res_iou, 4),
                "candidate_count": len(candidates),
            }
            results.append(record)
            class_counts[classification] += 1
            class_docs[classification].add(doc_id)

        if d_idx % 20 == 0 or d_idx == len(doc_cases):
            elapsed = time.perf_counter() - t0
            print(f"[{d_idx}/{len(doc_cases)} docs] Audited {len(results)}/{len(cases)} cases ({elapsed:.1f}s)...")

    # Save complete failure audit
    out_audit = repo_root / "research" / "experiments" / "EXP-035" / "failure_audit.json"
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "EXP-035",
            "total_alleged_cases": len(cases),
            "audited_cases_count": len(results),
            "classification_summary": {
                c: {
                    "field_count": class_counts[c],
                    "doc_count": len(class_docs[c]),
                    "percentage_of_alleged": round((class_counts[c] / len(cases)) * 100, 2),
                }
                for c in sorted(class_counts.keys(), key=lambda k: class_counts[k], reverse=True)
            },
            "cases": results,
        }, f, indent=2)

    # Save failure distribution summary
    out_dist = repo_root / "research" / "experiments" / "EXP-035" / "failure_distribution.json"
    summary_list = []
    for rank, (c, count) in enumerate(sorted(class_counts.items(), key=lambda x: x[1], reverse=True), 1):
        summary_list.append({
            "rank": rank,
            "failure_mechanism": c,
            "field_count": count,
            "doc_count": len(class_docs[c]),
            "percentage": round((count / len(cases)) * 100, 2),
        })

    with open(out_dist, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)

    print("\n" + "=" * 80)
    print("EXP-035 CAUSAL AUDIT RESULTS")
    print("=" * 80)
    print(f"{'Rank':<5} | {'Failure Mechanism':<26} | {'Fields':<8} | {'Docs':<6} | {'% of Total':<10}")
    print("-" * 80)
    for row in summary_list:
        print(f"{row['rank']:<5} | {row['failure_mechanism']:<26} | {row['field_count']:>8} | {row['doc_count']:<6} | {row['percentage']:>8.2f}%")
    print("=" * 80)
    print(f"Results saved to {out_audit} and {out_dist}")


if __name__ == "__main__":
    run_causal_audit()
