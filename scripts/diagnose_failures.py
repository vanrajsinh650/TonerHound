# Diagnostic failure analysis and candidate recall measurement script for EXP-002.
import json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

from extract_bench.test_cases.loader import load_test_cases
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import compute_unified_evidence_metrics
from tonerhound.benchmark.adapter import ExtractBenchAdapter, _flatten_leaves_with_context
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import EvidenceMatcher, MatchCandidate
from tonerhound.models.types import ExtractionInput, ProvenanceStatus
from tonerhound.resolution.resolver import EvidenceResolver

def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def generate_candidates_for_field(matcher, value, context, page_hint, total_pages):
    candidates = []
    is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
    if is_num:
        candidates.extend(matcher.find_normalized_numeric_candidates(value, page_hint=page_hint))
    if isinstance(value, str):
        candidates.extend(matcher.find_normalized_date_candidates(value, page_hint=page_hint))
        candidates.extend(matcher.find_exact_candidates(value, page_hint=page_hint))
        if not candidates:
            candidates.extend(matcher.find_normalized_numeric_candidates(value, page_hint=page_hint))
    elif not candidates and isinstance(value, (int, float)):
        candidates.extend(matcher.find_exact_candidates(str(value), page_hint=page_hint))

    if not candidates and page_hint is not None:
        if is_num:
            candidates.extend(matcher.find_normalized_numeric_candidates(value, page_hint=None))
        if isinstance(value, str):
            candidates.extend(matcher.find_normalized_date_candidates(value, page_hint=None))
            candidates.extend(matcher.find_exact_candidates(value, page_hint=None))
            if not candidates:
                candidates.extend(matcher.find_normalized_numeric_candidates(value, page_hint=None))
        elif not candidates and isinstance(value, (int, float)):
            candidates.extend(matcher.find_exact_candidates(str(value), page_hint=None))

    if not candidates and isinstance(value, str) and len(value.strip()) >= 4:
        candidates.extend(matcher.find_fuzzy_candidates(value, threshold=0.80, page_hint=page_hint))

    unique_candidates = []
    for c in candidates:
        if not any(u.page == c.page and u.bbox.iou(c.bbox) >= 0.90 for u in unique_candidates):
            unique_candidates.append(c)
    return unique_candidates

def rank_candidates_with_context(resolver, candidates, context):
    if not candidates or len(candidates) == 1:
        return candidates
    scored = resolver._score_candidates_with_context(candidates, context or "")
    scored.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in scored]

def analyze_document(case):
    print(f"\n==========================================")
    print(f"Analyzing: {case.test_id} ({case.file_path.name})")
    print(f"==========================================")
    doc_index = DocumentIndex.from_pdf(case.file_path)
    total_tokens = sum(len(p.tokens) for p in doc_index.pages)
    total_pages = len(doc_index.pages)
    is_ocr_absent = (total_tokens < 25)

    adapter = ExtractBenchAdapter(doc_index)
    resolver = adapter.resolver
    matcher = resolver.matcher

    t0 = time.perf_counter()
    payload = adapter.ground_extracted_data(case.expected_output, example_id=case.test_id)
    grounding_time = time.perf_counter() - t0
    citations = payload["field_citations"]
    cit_by_field = {c["field_path"]: c for c in citations}

    eval_res = compute_unified_evidence_metrics(
        expected_output=case.expected_output,
        extracted_data=case.expected_output,
        field_rules=case.test_rules,
        field_citations=citations,
        data_schema=case.data_schema,
    )
    metric_dict = {m.metric_name: m.value for m in eval_res}

    leaves = _flatten_leaves_with_context(case.expected_output)
    leaf_info = {}
    doc_offset = adapter._detect_page_offset(leaves)
    for path, val, p_hint, ctx, rec_path in leaves:
        eff_hint = p_hint + doc_offset if p_hint is not None else None
        leaf_info[path] = {
            "value": val,
            "page_hint": eff_hint,
            "context": ctx,
            "record_path": rec_path,
        }

    rules_with_bbox = [r for r in case.test_rules if any(ev.bbox is not None for ev in r.evidence)]
    print(f"Total Rules: {len(case.test_rules)} | Rules with GT BBoxes: {len(rules_with_bbox)}")

    rec_at_1 = 0
    rec_at_5 = 0
    rec_at_10 = 0
    rec_at_20 = 0
    failure_counts = Counter()
    failure_records = []
    success_count = 0

    for rule in rules_with_bbox:
        field_path = rule.field_path
        ev = next(e for e in rule.evidence if e.bbox is not None)
        gt_page = ev.page
        gt_bbox = ev.bbox
        gt_val = ev.value

        l_data = leaf_info.get(field_path, {})
        val = l_data.get("value", gt_val)
        ctx = l_data.get("context")
        p_hint = l_data.get("page_hint")

        raw_candidates = generate_candidates_for_field(matcher, val, ctx, p_hint, total_pages)
        ranked_candidates = rank_candidates_with_context(resolver, raw_candidates, ctx)

        in_top_1 = in_top_5 = in_top_10 = in_top_20 = False
        for rank, cand in enumerate(ranked_candidates[:20]):
            cand_box = [cand.bbox.x, cand.bbox.y, cand.bbox.width, cand.bbox.height]
            iou = compute_iou(cand_box, gt_bbox)
            if cand.page == gt_page and iou >= 0.5:
                if rank < 1: in_top_1 = True
                if rank < 5: in_top_5 = True
                if rank < 10: in_top_10 = True
                if rank < 20: in_top_20 = True
                break

        if in_top_1: rec_at_1 += 1
        if in_top_5: rec_at_5 += 1
        if in_top_10: rec_at_10 += 1
        if in_top_20: rec_at_20 += 1

        pred_cit = cit_by_field.get(field_path)
        if val is None:
            cat = "extraction failure"
            failure_counts[cat] += 1
            failure_records.append({
                "field_path": field_path,
                "category": cat,
                "gt_page": gt_page,
                "gt_bbox": gt_bbox,
                "gt_val": gt_val,
                "reason": "Value is None/missing in extracted output",
            })
            continue

        if is_ocr_absent:
            cat = "OCR absence"
            failure_counts[cat] += 1
            failure_records.append({
                "field_path": field_path,
                "category": cat,
                "gt_page": gt_page,
                "gt_bbox": gt_bbox,
                "gt_val": gt_val,
                "reason": f"Page has no embedded text stream ({total_tokens} tokens across {total_pages} pages)",
            })
            continue

        if pred_cit is None:
            if not raw_candidates:
                page_obj = doc_index.get_page(gt_page)
                page_text = " ".join(t.text for t in page_obj.tokens) if page_obj else ""
                val_str = str(val)
                if val_str in page_text:
                    cat = "parser failure"
                    reason = "Text exists on page but tokenization/subsequence did not index it"
                else:
                    cat = "missing candidate"
                    reason = f"No candidate matching {val} found on page {p_hint or "all"}"
            else:
                cat = "ambiguity"
                reason = f"Resolver returned ambiguous status ({len(raw_candidates)} indistinguishable candidates)"
            failure_counts[cat] += 1
            failure_records.append({
                "field_path": field_path,
                "category": cat,
                "gt_page": gt_page,
                "gt_bbox": gt_bbox,
                "gt_val": gt_val,
                "reason": reason,
            })
            continue

        pred_page = pred_cit["page"]
        pred_bbox = pred_cit["bbox"]
        iou = compute_iou(pred_bbox, gt_bbox)

        if pred_page == gt_page and iou >= 0.5:
            success_count += 1
            continue

        if pred_page != gt_page:
            cat = "wrong page"
            reason = f"Predicted page {pred_page} != GT page {gt_page}"
        elif 0.0 < iou < 0.5:
            if (pred_bbox[3] > 2.0 * gt_bbox[3]) or (gt_bbox[3] > 2.0 * pred_bbox[3]):
                cat = "multiline failure"
                reason = f"Bounding box height mismatch under multiline text (IoU={iou:.3f})"
            else:
                cat = "wrong region"
                reason = f"Predicted bbox on correct page partially overlaps GT box (IoU={iou:.3f})"
        else:
            if raw_candidates and len(raw_candidates) > 1:
                cat = "table/structure failure"
                reason = f"Predicted different table cell/occurrence on page {gt_page} with identical value {val}"
            else:
                cat = "wrong occurrence"
                reason = f"Predicted different occurrence of {val} on page {gt_page}"

        failure_counts[cat] += 1
        failure_records.append({
            "field_path": field_path,
            "category": cat,
            "gt_page": gt_page,
            "gt_bbox": gt_bbox,
            "gt_val": gt_val,
            "pred_page": pred_page,
            "pred_bbox": pred_bbox,
            "iou": iou,
            "reason": reason,
        })

    n_rules = len(rules_with_bbox)
    r1_pct = (rec_at_1 / n_rules * 100.0) if n_rules else 0.0
    r5_pct = (rec_at_5 / n_rules * 100.0) if n_rules else 0.0
    r10_pct = (rec_at_10 / n_rules * 100.0) if n_rules else 0.0
    r20_pct = (rec_at_20 / n_rules * 100.0) if n_rules else 0.0

    print(f"Results for {case.test_id}:")
    print(f"  Official Word F1: {metric_dict.get("extract_unified_grounded_f1", 0.0) * 100:.2f}%")
    print(f"  Official Page F1: {metric_dict.get("extract_unified_page_f1", 0.0) * 100:.2f}%")
    print(f"  Success bboxes (IoU>=0.5): {success_count}/{n_rules} ({success_count/n_rules*100:.2f}%)")
    print(f"  Candidate Recall: R@1={r1_pct:.2f}%, R@5={r5_pct:.2f}%, R@10={r10_pct:.2f}%, R@20={r20_pct:.2f}%")
    print(f"  Failure Breakdown: {dict(failure_counts)}")

    return {
        "test_id": case.test_id,
        "document_name": case.file_path.name,
        "num_pages": total_pages,
        "num_tokens": total_tokens,
        "is_ocr_absent": is_ocr_absent,
        "num_bbox_rules": n_rules,
        "success_count": success_count,
        "word_f1": metric_dict.get("extract_unified_grounded_f1"),
        "page_f1": metric_dict.get("extract_unified_page_f1"),
        "recall_at_1": r1_pct,
        "recall_at_5": r5_pct,
        "recall_at_10": r10_pct,
        "recall_at_20": r20_pct,
        "failure_counts": dict(failure_counts),
        "failures": failure_records[:50],
        "total_failures_logged": len(failure_records),
    }

def main():
    cases = load_test_cases(root_dir / "research" / "data" / "test")
    target_keys = ["W14-Atascosa", "bianco-2024", "real_pueblo_oct_2025", "real_sm0801_eco_full"]

    all_analyses = []
    for key in target_keys:
        case = next(c for c in cases if key in c.test_id)
        analysis = analyze_document(case)
        all_analyses.append(analysis)

    json_path = root_dir / "research" / "failures" / "EXP-002-failures.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_analyses, f, indent=2)
    print(f"\n[Saved JSON Failures]: {json_path}")

    md_path = root_dir / "research" / "failures" / "EXP-002-failures.md"
    agg_failures = Counter()
    total_rules = sum(a["num_bbox_rules"] for a in all_analyses)
    total_success = sum(a["success_count"] for a in all_analyses)
    for a in all_analyses:
        for k, v in a["failure_counts"].items():
            agg_failures[k] += v

    md_lines = [
        "# EXP-002 Baseline Failure Analysis & Candidate Recall Report",
        "",
        f"**Date**: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}  ",
        "**Scope**: Target failure cases (`W14-Atascosa`, `bianco-2024`, `real_pueblo_oct_2025`, `real_sm0801_eco_full`)  ",
        "**Evaluator**: ExtractBench Unified Evidence Grounding  ",
        "",
        "---",
        "",
        "## 1. Executive Failure Taxonomy Summary",
        "",
        f"Total Ground-Truth BBox Rules Analyzed: **{total_rules}**  ",
        f"Total Successfully Grounded (IoU $\\ge$ 0.5): **{total_success}** ({total_success/total_rules*100:.2f}%)  ",
        f"Total Failed / Missed Groundings: **{total_rules - total_success}** ({(total_rules - total_success)/total_rules*100:.2f}%)  ",
        "",
        "| Failure Category | Total Occurrences | Share of Failures | Primary Root Cause |",
        "| :--- | :---: | :---: | :--- |",
    ]

    for cat, count in agg_failures.most_common():
        pct = (count / (total_rules - total_success)) * 100.0 if (total_rules - total_success) else 0.0
        md_lines.append(f"| `{cat}` | {count} | {pct:.1f}% | See per-document details below |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Candidate Pool Recall Diagnostic (Recall@K)",
        "",
        "Mandatory architectural gate: Does the correct physical bounding box even exist in the candidate pool prior to resolution ranking?",
        "",
        "| Document Case | GT BBoxes | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Candidate Diagnosis |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for a in all_analyses:
        diag = "Candidate pool healthy; ranking/disambiguation is bottleneck" if a["recall_at_20"] > 80.0 else (
            "Zero text layer (OCR absence)" if a["is_ocr_absent"] else "Candidate generation needs improvement"
        )
        md_lines.append(
            f"| `{a["test_id"]}` | {a["num_bbox_rules"]} | {a["recall_at_1"]:.2f}% | "
            f"{a["recall_at_5"]:.2f}% | {a["recall_at_10"]:.2f}% | {a["recall_at_20"]:.2f}% | {diag} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Per-Document Deep Dive",
        "",
    ])

    for a in all_analyses:
        md_lines.extend([
            f"### Document: `{a["test_id"]}`",
            f"- **PDF File**: `{a["document_name"]}`",
            f"- **Pages**: {a["num_pages"]} | **Extracted Tokens**: {a["num_tokens"]}",
            f"- **Official Metrics**: Word F1 = **{a["word_f1"]*100 if a["word_f1"] is not None else 0.0:.2f}%** | Page F1 = **{a["page_f1"]*100 if a["page_f1"] is not None else 0.0:.2f}%**",
            f"- **Candidate Recall**: R@1 = {a["recall_at_1"]:.2f}% | R@5 = {a["recall_at_5"]:.2f}% | R@10 = {a["recall_at_10"]:.2f}% | R@20 = {a["recall_at_20"]:.2f}%",
            f"- **Failure Taxonomy Breakdown**:",
        ])
        for cat, cnt in sorted(a["failure_counts"].items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"  - `{cat}`: {cnt}")
        md_lines.append("")

    md_text = "\n".join(md_lines)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"[Saved Markdown Failures]: {md_path}")

if __name__ == "__main__":
    main()
