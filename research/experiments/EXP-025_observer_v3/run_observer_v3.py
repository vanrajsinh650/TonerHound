import os
import json
import pandas as pd
import numpy as np

def iou(b1, b2):
    if b1 is None or b2 is None: return 0.0
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter/union if union > 0 else 0.0

def process_observer_v3():
    input_path = '/home/vanrajsinh/Projects/TonerHound/research/observer/field_records.parquet'
    output_dir = '/home/vanrajsinh/Projects/TonerHound/research/experiments/EXP-025_observer_v3'
    
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} records.")
    
    # We will compute the new columns
    candidate_rank1_bbox_list = []
    candidate_rank1_iou_list = []
    citation_bbox_list = []
    citation_vs_candidate_iou_list = []
    bbox_transformation_type_list = []
    failure_class_v3_list = []
    
    for idx, row in df.iterrows():
        fail_class = row.get('failure_class', 'SUCCESS')
        cand_str = row.get('candidate_pool', '[]')
        cit_str = row.get('selected_candidate', '{}')
        
        cand_bbox = None
        cand_iou = 0.0
        try:
            pool = json.loads(cand_str) if isinstance(cand_str, str) else cand_str
            if pool and len(pool) > 0:
                rank1 = pool[0]
                cand_bbox = rank1.get('bbox')
                cand_iou = rank1.get('best_iou', 0.0)
        except Exception:
            pass
            
        cit_bbox = None
        try:
            cit = json.loads(cit_str) if isinstance(cit_str, str) else cit_str
            if cit:
                cit_bbox = cit.get('bbox')
        except Exception:
            pass
            
        c_v_c_iou = 0.0
        transform = "NO_CANDIDATE"
        if cand_bbox is not None and cit_bbox is not None:
            c_v_c_iou = iou(cand_bbox, cit_bbox)
            if c_v_c_iou < 0.1:
                transform = "BBOX_REPLACED"
            elif c_v_c_iou > 0.99:
                transform = "BBOX_PRESERVED"
            else:
                # Let's check narrowed vs expanded by area
                area_cand = cand_bbox[2]*cand_bbox[3]
                area_cit = cit_bbox[2]*cit_bbox[3]
                if area_cit < area_cand:
                    transform = "BBOX_NARROWED"
                else:
                    transform = "BBOX_EXPANDED"
        elif cand_bbox is not None:
            transform = "BBOX_REPLACED" # citation has no bbox
            
        candidate_rank1_bbox_list.append(json.dumps(cand_bbox) if cand_bbox else None)
        candidate_rank1_iou_list.append(cand_iou)
        citation_bbox_list.append(json.dumps(cit_bbox) if cit_bbox else None)
        citation_vs_candidate_iou_list.append(c_v_c_iou)
        bbox_transformation_type_list.append(transform)
        failure_class_v3_list.append(fail_class)
        
    df['candidate_rank1_bbox'] = candidate_rank1_bbox_list
    df['candidate_rank1_iou'] = candidate_rank1_iou_list
    df['citation_bbox'] = citation_bbox_list
    df['citation_vs_candidate_iou'] = citation_vs_candidate_iou_list
    df['bbox_transformation_type'] = bbox_transformation_type_list
    df['failure_class_v3'] = failure_class_v3_list
    
    # Save v3 parquet
    out_parquet = os.path.join(output_dir, 'field_records_v3.parquet')
    df.to_parquet(out_parquet)
    print(f"Saved {out_parquet}")
    
    # Metrics
    total_gradeable = len(df)
    scgf_mask = df['failure_class_v3'] == 'SELECTED_CITATION_GEOMETRY_FAILURE'
    scgf_df = df[scgf_mask]
    total_scgf = len(scgf_df)
    
    scgf_replaced = scgf_df[scgf_df['citation_vs_candidate_iou'] < 0.1]
    pct_replaced = len(scgf_replaced) / total_scgf if total_scgf > 0 else 0
    
    scgf_recoverable = scgf_df[scgf_df['candidate_rank1_iou'] >= 0.5]
    pct_recoverable = len(scgf_recoverable) / total_scgf if total_scgf > 0 else 0
    
    # Current WF1 matches production: 45.31%
    current_wf1 = 45.31 
    # Approx 445950 total fields. 
    # WF1 gain approx = (len(scgf_recoverable) / 445950) * 100
    expected_gain = (len(scgf_recoverable) / total_gradeable) * 100 if total_gradeable > 0 else 0
    
    stats = {
        'total_gradeable_fields': total_gradeable,
        'total_scgf': total_scgf,
        'scgf_pct_replaced_bbox': pct_replaced * 100,
        'scgf_pct_recoverable_rank1': pct_recoverable * 100,
        'scgf_recoverable_count': len(scgf_recoverable),
        'expected_wf1_gain_pct': expected_gain,
        'transform_dist': df['bbox_transformation_type'].value_counts().to_dict(),
        'scgf_transform_dist': scgf_df['bbox_transformation_type'].value_counts().to_dict()
    }
    
    out_json = os.path.join(output_dir, 'observer_v3_reconciliation.json')
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=2)
        
    out_md = os.path.join(output_dir, 'observer_v3_report.md')
    md_content = f"""# Observer V3 Reconciliation Report

## Overview
- Total gradeable fields: {total_gradeable}
- Word Grounding F1 (baseline): {current_wf1}%
- Expected Gain if recoverable SCGF fixed: {expected_gain:.2f}%
- Total SCGF fields: {total_scgf}

## Bounding Box Transformations in SCGF
- % Completely Replaced (<0.1 IoU): {pct_replaced*100:.2f}%
- % Recoverable (Rank-1 IoU >= 0.5): {pct_recoverable*100:.2f}% ({len(scgf_recoverable)} fields)

### Transformation Distribution (SCGF Only)
```json
{json.dumps(stats['scgf_transform_dist'], indent=2)}
```

### Transformation Distribution (All Fields)
```json
{json.dumps(stats['transform_dist'], indent=2)}
```
"""
    with open(out_md, 'w') as f:
        f.write(md_content)
        
    print(json.dumps(stats))

if __name__ == '__main__':
    process_observer_v3()
