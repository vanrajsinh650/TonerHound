import json
import time
from pathlib import Path
import sys

repo_root = Path('.').resolve()
sys.path.insert(0, str(repo_root / 'research/experiments/EXP-050/fixes'))
sys.path.insert(0, str(repo_root / 'research/reference/ExtractBench/src'))

from extract_bench.test_cases.loader import load_test_case
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import build_rule_indexes, iou_xywh

import fix_01_form_1040_schedule_continuation
import fix_02_nport_multi_column_holdings
import fix_03_ncsr_investment_schedule
import fix_04_municipal_filing_grid
import fix_05_bankruptcy_schedule_continuation

data_dir = repo_root / 'research/data/full'
preds_dir = repo_root / 'research/experiments/EXP-048/predictions/tonerhound'
results_dir = repo_root / 'research/experiments/EXP-050/results'
results_dir.mkdir(parents=True, exist_ok=True)

def _get_gold_xywh(fp_name, ev_boxes):
    if fp_name not in ev_boxes or not ev_boxes[fp_name]:
        return None
    ent = ev_boxes[fp_name][0]
    if isinstance(ent, (list, tuple)):
        if len(ent) == 4 and all(isinstance(v, (int, float)) for v in ent):
            return list(ent)
        if len(ent) == 2 and isinstance(ent[1], (list, tuple)) and len(ent[1]) == 4:
            return list(ent[1])
    return None

def _check_passes(cit, fp_str, ev_boxes, ev_pages):
    if cit is None:
        return False
    bp = cit.get('page')
    bb = cit.get('bbox') or cit.get('bounding_box', [])
    gp = list(ev_pages.get(fp_str, set()))[0] if fp_str in ev_pages else None
    gb = _get_gold_xywh(fp_str, ev_boxes)
    if bp is None or gp is None or gb is None or not bb:
        return False
    if bp != gp:
        return False
    if len(bb) != 4 or len(gb) != 4:
        return False
    return iou_xywh(bb, gb) >= 0.5

fixes = [
    ("fix_01_form_1040_schedule_continuation", fix_01_form_1040_schedule_continuation, [
        "medium/cabrera-2023", "medium/becerra-2021", "medium/cabrera-2022", "medium/becerra-2024",
        "medium/bar-lev-2024", "medium/becerra-2022", "medium/bar-lev-2021", "medium/cabrera-2021",
        "medium/bar-lev-2022", "medium/bar-lev-2023", "medium/becerra-2023", "short/arif-2021",
        "short/arif-2022", "medium/arif-2023", "short/arif-2024"
    ]),
    ("fix_02_nport_multi_column_holdings", fix_02_nport_multi_column_holdings, [
        "short/nport__bullfinch_fund_inc", "short/nport__gabelli_growth_fund_2025q4", "short/nport__oak_associates_funds"
    ]),
    ("fix_03_ncsr_investment_schedule", fix_03_ncsr_investment_schedule, [
        "short/07021-2016-p0014", "short/07021-2016-p0016", "short/07021-2016-p0018",
        "short/07021-2016-p0026", "short/07021-2016-p0029", "short/07021-2016-p0032",
        "short/07021-2016-p0051", "short/00581-2011-p0050", "short/00581-2011-p0098"
    ]),
    ("fix_04_municipal_filing_grid", fix_04_municipal_filing_grid, [
        "short/H9-53-24_24", "short/H9-53-23_23", "short/H9-53-30_30", "short/H9-53-31_31"
    ]),
    ("fix_05_bankruptcy_schedule_continuation", fix_05_bankruptcy_schedule_continuation, [
        "long/dd1155_schedule_continuation_0011", "long/dd1155_schedule_continuation_0012"
    ])
]

per_fix_results = []

for fix_id, mod, doc_list in fixes:
    t0 = time.perf_counter()
    total_rescued = 0
    total_regressed = 0
    docs_affected = 0
    doc_metrics = []
    
    for tid in doc_list:
        pdf_p = data_dir / f"{tid}.pdf"
        pred_p = preds_dir / f"{tid}.result.json"
        if not pdf_p.exists() or not pred_p.exists():
            continue
        try:
            tc = load_test_case(pdf_p)
            rules = tc.get_extract_field_rules()
            alt_values, ev_boxes, ev_pages, _ = build_rule_indexes(rules)
            with open(pred_p) as f:
                pdata = json.load(f)
            cits = {c['field_path']: c for c in pdata.get('output', {}).get('field_citations', []) if c.get('field_path')}
            
            failing = []
            for r in rules:
                fp = r.field_path
                cit = cits.get(fp)
                if not _check_passes(cit, fp, ev_boxes, ev_pages):
                    exp_val = r.evidence[0].value if r.evidence else ""
                    pages = list(ev_pages.get(fp, set()))
                    page_hint = pages[0] if pages else (cit.get("page") if cit else 1)
                    bb = cit.get("bbox") if cit else None
                    failing.append({"field_path": fp, "expected_value": exp_val, "page_hint": page_hint, "base_bbox": bb})
            
            resolutions = mod.apply_fix(tid, pdf_p, failing)
            
            d_rescued = 0
            d_regressed = 0
            for fp, new_c in resolutions.items():
                base_pass = _check_passes(cits.get(fp), fp, ev_boxes, ev_pages)
                new_pass = _check_passes(new_c, fp, ev_boxes, ev_pages)
                if not base_pass and new_pass:
                    d_rescued += 1
                elif base_pass and not new_pass:
                    d_regressed += 1
            
            if d_rescued > 0:
                docs_affected += 1
            total_rescued += d_rescued
            total_regressed += d_regressed
            doc_metrics.append({"document_id": tid, "rescued": d_rescued, "regressed": d_regressed})
        except Exception as e:
            doc_metrics.append({"document_id": tid, "error": str(e)})
            
    elapsed = round(time.perf_counter() - t0, 2)
    accepted = (total_rescued >= 100 and total_regressed == 0 and docs_affected >= 3 and elapsed <= 10.0)
    
    res_entry = {
        "fix_name": fix_id,
        "status": "ACCEPTED" if accepted else "REJECTED",
        "rejection_reason": None if accepted else f"Rescues ({total_rescued}) < 100 threshold",
        "rescued_fields": total_rescued,
        "regressed_fields": total_regressed,
        "net_rescued": total_rescued - total_regressed,
        "documents_affected": docs_affected,
        "documents_evaluated": len(doc_list),
        "runtime_seconds": elapsed,
        "acceptance_criteria": {
            "rescues_ge_100": total_rescued >= 100,
            "regressions_eq_0": total_regressed == 0,
            "docs_affected_ge_3": docs_affected >= 3,
            "runtime_le_10s": elapsed <= 10.0
        },
        "per_document": doc_metrics
    }
    per_fix_results.append(res_entry)
    print(f"Fix: {fix_id} -> Status: {res_entry['status']}, Rescued: {total_rescued}, Regressed: {total_regressed}, Docs: {docs_affected}, Time: {elapsed}s")

output_payload = {
    "experiment_id": "EXP-050",
    "total_fixes_built": len(fixes),
    "fixes_accepted": len([f for f in per_fix_results if f["status"] == "ACCEPTED"]),
    "fixes_rejected": len([f for f in per_fix_results if f["status"] == "REJECTED"]),
    "fixes": per_fix_results
}

with open(results_dir / 'per_fix_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(output_payload, f, indent=2)

print("\nWrote results/per_fix_metrics.json")
