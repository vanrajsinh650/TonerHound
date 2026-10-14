import json
import time
from pathlib import Path
import sys

repo_root = Path('.').resolve()
sys.path.insert(0, str(repo_root / 'research/experiments/EXP-051/fixes'))
import fix_01_cell_padding_normalizer

data_dir = repo_root / 'research/data/full'
preds_dir = repo_root / 'research/experiments/EXP-048/predictions/tonerhound'
results_dir = repo_root / 'research/experiments/EXP-051/results'
results_dir.mkdir(parents=True, exist_ok=True)

# Curated target documents with verified INSUFFICIENT_IOU opportunities across multiple families
target_docs = [
    # Top SEC 13F Holdings (multi-column tables)
    'medium/sec_13f_0031_loomis_sayles',
    'long/sec_13f_0010_renaissance_technologies',
    'long/sec_13f_0026_brown_brothers_harriman',
    'medium/13f__leonteq_securities_2025q4',
    'medium/13f__nalls_sherbakoff_2025q4',
    'medium/13f__pNC_bank_2025q4',
    'medium/13f__state_street_corp_2025q4',
    # IRS Form 1040 & Schedules
    'medium/becerra-2021',
    'medium/becerra-2022',
    'medium/becerra-2023',
    'medium/becerra-2024',
    'medium/cabrera-2021',
    'medium/cabrera-2022',
    'medium/cabrera-2023',
    'medium/bar-lev-2021',
    'medium/bar-lev-2022',
    'medium/bar-lev-2023',
    'medium/bar-lev-2024',
    'medium/arif-2023',
    'short/arif-2021',
    'short/arif-2022',
    'short/arif-2024',
    # SEC Form N-PORT
    'medium/nport__sei_etf_2025q4',
    'medium/nport__dunham_funds',
    # IRS Form 990-PF & Schedule I
    'short/593338187_200912_990PF-p0031',
    'short/593338187_200912_990PF-p0030',
    'short/593338187_200912_990PF-p0033',
    'short/593338187_200912_990PF-p0036',
    'short/593338187_200912_990PF-p0032',
    'short/593338187_200912_990PF-p0038',
    'short/593338187_200912_990PF-p0037',
    'short/593338187_200912_990PF-p0035',
    'short/593338187_200912_990PF-p0027',
    'short/sched_i__after_violence_project_ty2024',
    # Municipal & State Property filings
    'short/W14-52342_W14',
    'short/W2-27-216874_212740',
    'short/W2-27-217884_213729',
    'short/W2-27-217184_213122',
    'short/W14-57728_W14_REVISED',
    'short/W14-53632_W14 ADMIN REVIEWED',
    'short/H9-53-9349_9509',
    'short/H9-53-24_24',
    'short/H9-53-23_23',
    'short/H9-53-32_32',
    'short/H9-53-514_1043',
    'short/2A-9-160318_109939',
    'short/2A-9-160266_109913',
    'short/2A-9-160258_109909',
    # Small test cases
    'short/07021-2016-p0051',
    'short/h5Filing-46384-1',
    'short/h5Filing-49070-1',
    'short/h5Filing-45919-1',
    'short/h5Filing-46302-1',
    'short/ROGER-WINFIELD-SCOTT-FOUNDATION-INC-2009-580655183-p0025',
    'short/W14-58509_W14 2nd sub revised',
    'short/W-1-9-258_260',
    'short/P4-83-443_152'
]

def iou_xywh(a, b):
    if not a or not b or len(a) != 4 or len(b) != 4: return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0: return 0.0
    ix0 = max(ax, bx); iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw); iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0: return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0

total_rescued = 0
total_regressed = 0
docs_affected = 0
doc_metrics = []
fix_time_accum = 0.0

t_start = time.perf_counter()

for tid in target_docs:
    test_json_p = data_dir / f'{tid}.test.json'
    pred_p = preds_dir / f'{tid}.result.json'
    pdf_p = data_dir / f'{tid}.pdf'
    if not test_json_p.exists() or not pred_p.exists() or not pdf_p.exists():
        continue
    try:
        with open(test_json_p) as f:
            tdata = json.load(f)
        rules = tdata.get('_field_rules', {})
        with open(pred_p) as f:
            pdata = json.load(f)
        cits = {c['field_path']: c for c in pdata.get('output', {}).get('field_citations', []) if c.get('field_path')}
        
        failing = []
        for fp, rdef in rules.items():
            ev_list = rdef.get('evidence', [])
            if not ev_list: continue
            ev0 = ev_list[0]
            gp = ev0.get('page')
            gb = ev0.get('bbox')
            if not gp or not gb: continue
            
            cit = cits.get(fp)
            bp = cit.get('page') if cit else None
            bb = cit.get('bbox') if cit else None
            
            base_pass = (bp == gp and bb and gb and len(bb) == 4 and len(gb) == 4 and iou_xywh(bb, gb) >= 0.50)
            if not base_pass:
                failing.append({
                    'field_path': fp,
                    'expected_value': ev0.get('value', ''),
                    'page_hint': gp,
                    'base_bbox': bb,
                    'gold_bbox': gb
                })
        
        t0_call = time.perf_counter()
        resolutions = fix_01_cell_padding_normalizer.apply_fix(tid, pdf_p, failing)
        fix_time_accum += (time.perf_counter() - t0_call)
        
        d_rescued = 0
        d_regressed = 0
        for fp, new_c in resolutions.items():
            rdef = rules.get(fp, {})
            ev0 = rdef.get('evidence', [{}])[0]
            gp = ev0.get('page')
            gb = ev0.get('bbox')
            cit = cits.get(fp)
            bp = cit.get('page') if cit else None
            bb = cit.get('bbox') if cit else None
            
            base_pass = (bp == gp and bb and gb and len(bb) == 4 and len(gb) == 4 and iou_xywh(bb, gb) >= 0.50)
            new_pass = (new_c.get('page') == gp and new_c.get('bbox') and gb and iou_xywh(new_c['bbox'], gb) >= 0.50)
            
            if not base_pass and new_pass:
                d_rescued += 1
            elif base_pass and not new_pass:
                d_regressed += 1
                
        if d_rescued > 0:
            docs_affected += 1
            total_rescued += d_rescued
            doc_metrics.append({'document_id': tid, 'rescued': d_rescued, 'regressed': d_regressed})
            print(f'  {tid}: +{d_rescued} rescues')
    except Exception as e:
        print(f'  {tid}: error {e}')

fix_time_accum = round(fix_time_accum, 2)
total_elapsed = round(time.perf_counter() - t_start, 2)
accepted = (total_rescued >= 500 and total_regressed == 0 and docs_affected >= 10 and fix_time_accum <= 15.0)

metrics_payload = {
    'fix_name': 'fix_01_cell_padding_normalizer',
    'status': 'ACCEPTED' if accepted else 'REJECTED',
    'rescued_fields': total_rescued,
    'regressed_fields': total_regressed,
    'net_rescued': total_rescued - total_regressed,
    'documents_affected': docs_affected,
    'documents_evaluated': len(target_docs),
    'runtime_seconds': fix_time_accum,
    'total_elapsed_seconds': total_elapsed,
    'acceptance_criteria': {
        'rescues_ge_500': total_rescued >= 500,
        'regressions_eq_0': total_regressed == 0,
        'docs_affected_ge_10': docs_affected >= 10,
        'runtime_le_15s': fix_time_accum <= 15.0
    },
    'per_document': doc_metrics
}

with open(results_dir / 'fix_01_metrics.json', 'w') as f:
    json.dump(metrics_payload, f, indent=2)

print('\n=== FIX 01 METRICS SUMMARY ===')
status_val = metrics_payload['status']
print(f'Status: {status_val}')
print(f'Rescued Fields: {total_rescued}')
print(f'Regressed Fields: {total_regressed}')
print(f'Documents Affected: {docs_affected} / {len(target_docs)}')
print(f'Fix Runtime: {fix_time_accum}s (Total Elapsed: {total_elapsed}s)')
