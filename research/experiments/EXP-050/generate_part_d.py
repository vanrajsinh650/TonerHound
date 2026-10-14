import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import sys

repo_root = Path('.').resolve()
sys.path.insert(0, str(repo_root / 'research/reference/ExtractBench/src'))
from extract_bench.test_cases.loader import load_test_case
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import build_rule_indexes, iou_xywh

audit_dir = repo_root / 'research/experiments/EXP-050/audit'
data_dir = repo_root / 'research/data/full'
preds_dir = repo_root / 'research/experiments/EXP-048/predictions/tonerhound'

with open('research/experiments/EXP-048/results/full_benchmark_results.json') as f:
    bench = json.load(f)

per_doc_map = {d['test_id']: d for d in bench['per_document']}

with open('research/experiments/EXP-048/audit/lift_ranking.json') as f:
    lr = json.load(f)
lift_map = {d['test_id']: d for d in lr['top_documents_by_lift_potential']}

all_test_ids = [p.relative_to(preds_dir).as_posix().removesuffix('.result.json') for p in sorted(preds_dir.glob('**/*.result.json'))]
print(f"Total benchmark documents to process for Part D: {len(all_test_ids)}")

def detect_family(tid: str) -> str:
    if any(k in tid for k in ['becerra', 'bar-lev', 'cabrera', 'arif']) or '1040' in tid:
        return 'IRS Form 1040 / Individual Tax Return'
    elif any(k in tid for k in ['h9-', 'H9-']):
        return 'State Property Tax Valuations & Hearings (H9-53)'
    elif any(k in tid for k in ['w14-', 'W14-', 'w2-', 'W2-', 'W-1-']):
        return 'Municipal Real Estate Tax Assessments (W-Series)'
    elif any(k in tid for k in ['p18-', 'P18-', 'p4-', 'P4-']):
        return 'County Property Appraisal Schedules (P18/P4)'
    elif any(k in tid for k in ['2a-', '2A-']):
        return 'County Assessment Appeals & Petitions (Form 2A)'
    elif '07021-2016' in tid or '00581-2011' in tid:
        return 'SEC Form N-SAR Semi-Annual Reports'
    elif 'passcoag' in tid or 'w2' in tid.lower():
        return 'IRS Form W-2 Wage & Tax Statements'
    elif 'sec_13f' in tid.lower() or '13f' in tid.lower():
        return 'SEC Form 13F Holdings Report'
    elif 'nport' in tid.lower():
        return 'SEC Form N-PORT Portfolio Investments'
    elif 'ftx' in tid.lower() or 'imedia' in tid.lower():
        return 'Bankruptcy Creditor Schedules'
    elif 'ofac' in tid.lower():
        return 'OFAC Sanctions Entities'
    elif 'sched_i' in tid.lower():
        return 'IRS Form 990 Schedule I'
    elif 'invoice' in tid.lower() or 'receipt' in tid.lower() or 'bill' in tid.lower() or 'po' in tid.lower():
        return 'Invoices, Purchase Orders & Receipts'
    else:
        return 'ungrouped'

def get_field_pattern(fp: str) -> str:
    # Abstract index numbers [0], [1], etc.
    p = re.sub(r'\[\d+\]', '[]', fp)
    # Split by dot or line
    parts = p.split('.')
    if len(parts) > 1:
        return f"{parts[0]}.{parts[1]}"
    if p.startswith('line_'):
        return 'line_items'
    return parts[0]

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

field_breakdown_rows = []
bucket_class_counts = defaultdict(lambda: Counter())

for idx, tid in enumerate(all_test_ids):
    doc_bench = per_doc_map.get(tid, {})
    wf1 = doc_bench.get('word_f1')
    pf1 = doc_bench.get('page_f1')
    prec = doc_bench.get('precision')
    rec = doc_bench.get('recall')
    
    status = 'evaluated' if wf1 is not None else ('unextractable' if pf1 is not None else 'skipped')
    family = detect_family(tid)
    
    pdf_p = data_dir / f"{tid}.pdf"
    pred_p = preds_dir / f"{tid}.result.json"
    
    gradeable = 0
    passing = 0
    failing = 0
    
    top_patterns = ["none", "none", "none"]
    top_classes = ["none", "none", "none"]
    
    if status == 'evaluated':
        lr_entry = lift_map.get(tid)
        if lr_entry:
            gradeable = lr_entry['field_count']
            passing = lr_entry['passing_count']
            failing = lr_entry['failing_count']
        
        # Analyze failures
        try:
            tc = load_test_case(pdf_p)
            rules = tc.get_extract_field_rules()
            alt_values, ev_boxes, ev_pages, _ = build_rule_indexes(rules)
            with open(pred_p) as f:
                pdata = json.load(f)
            cits = {c['field_path']: c for c in pdata.get('output', {}).get('field_citations', []) if c.get('field_path')}
            
            pattern_counter = Counter()
            class_counter = Counter()
            
            for r in rules:
                fp = r.field_path
                cit = cits.get(fp)
                bp = cit.get('page') if cit else None
                bb = cit.get('bbox') if cit else None
                gp = list(ev_pages.get(fp, set()))[0] if fp in ev_pages else None
                gb = _get_gold_xywh(fp, ev_boxes)
                
                is_pass = False
                if bp == gp and bb and gb and len(bb) == 4 and len(gb) == 4:
                    if iou_xywh(bb, gb) >= 0.5:
                        is_pass = True
                
                if not is_pass and (gb is not None or gp is not None):
                    pat = get_field_pattern(fp)
                    pattern_counter[pat] += 1
                    
                    # Determine failure class
                    if cit is None or bp is None:
                        f_cls = 'ZERO_CANDIDATE'
                    elif bp != gp:
                        f_cls = 'PAGE_DRIFT'
                    elif not bb or len(bb) != 4:
                        f_cls = 'MALFORMED_BBOX'
                    elif gb is not None and abs(bb[1] - gb[1]) > 0.020:
                        f_cls = 'WRONG_ROW'
                    else:
                        f_cls = 'INSUFFICIENT_IOU'
                    
                    class_counter[f_cls] += 1
                    
                    # Accumulate into bucket summary
                    if wf1 is not None:
                        if wf1 < 30.0:
                            b_name = '0-30'
                        elif wf1 < 50.0:
                            b_name = '30-50'
                        elif wf1 < 70.0:
                            b_name = '50-70'
                        elif wf1 < 85.0:
                            b_name = '70-85'
                        else:
                            b_name = '85-100'
                        bucket_class_counts[b_name][f_cls] += 1
            
            p_common = [p[0] for p in pattern_counter.most_common(3)]
            for i, p in enumerate(p_common):
                top_patterns[i] = p
            
            c_common = [c[0] for c in class_counter.most_common(3)]
            for i, c in enumerate(c_common):
                top_classes[i] = c
                
        except Exception as e:
            pass
            
    elif status == 'unextractable':
        # Has rules and page_f1, but no gold word bboxes
        try:
            tc = load_test_case(pdf_p)
            gradeable = len(tc.get_extract_field_rules())
            failing = gradeable
            top_classes[0] = 'NO_GOLD_BBOX_PAGE_ONLY'
        except Exception:
            pass
            
    elif status == 'skipped':
        try:
            tc = load_test_case(pdf_p)
            gradeable = len(tc.get_extract_field_rules())
            failing = gradeable
            top_classes[0] = 'NO_GOLD_EVIDENCE_VALUE_ONLY'
        except Exception:
            pass

    field_breakdown_rows.append({
        "document_id": tid,
        "status": status,
        "gradeable": gradeable,
        "passing": passing,
        "failing": failing,
        "word_f1": f"{wf1:.4f}" if wf1 is not None else "",
        "page_f1": f"{pf1:.4f}" if pf1 is not None else "",
        "precision": f"{prec:.4f}" if prec is not None else "",
        "recall": f"{rec:.4f}" if rec is not None else "",
        "family": family,
        "top_pattern_1": top_patterns[0],
        "top_pattern_2": top_patterns[1],
        "top_pattern_3": top_patterns[2],
        "top_class_1": top_classes[0],
        "top_class_2": top_classes[1],
        "top_class_3": top_classes[2]
    })

# Write per_document_field_breakdown.csv
field_breakdown_csv = audit_dir / 'per_document_field_breakdown.csv'
headers = [
    "document_id", "status", "gradeable", "passing", "failing",
    "word_f1", "page_f1", "precision", "recall", "family",
    "top_pattern_1", "top_pattern_2", "top_pattern_3",
    "top_class_1", "top_class_2", "top_class_3"
]

with open(field_breakdown_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(field_breakdown_rows)

print(f"Wrote {len(field_breakdown_rows)} rows to per_document_field_breakdown.csv")

# Write per_field_bucket_summary.csv
bucket_summary_csv = audit_dir / 'per_field_bucket_summary.csv'
all_classes = ['PAGE_DRIFT', 'WRONG_ROW', 'ZERO_CANDIDATE', 'INSUFFICIENT_IOU', 'MALFORMED_BBOX']
buckets = ['0-30', '30-50', '50-70', '70-85', '85-100']

b_rows = []
for f_cls in all_classes:
    row = {"failure_class": f_cls}
    total_cls = 0
    for b in buckets:
        cnt = bucket_class_counts[b][f_cls]
        row[b] = cnt
        total_cls += cnt
    row["total"] = total_cls
    b_rows.append(row)

with open(bucket_summary_csv, 'w', newline='', encoding='utf-8') as f:
    b_headers = ["failure_class", "0-30", "30-50", "50-70", "70-85", "85-100", "total"]
    writer = csv.DictWriter(f, fieldnames=b_headers)
    writer.writeheader()
    writer.writerows(b_rows)

print("Wrote per_field_bucket_summary.csv")
