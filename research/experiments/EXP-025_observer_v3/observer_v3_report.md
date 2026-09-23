# Observer V3 Reconciliation Report

## Overview
- Total gradeable fields: 445950
- Word Grounding F1 (baseline): 45.31%
- Expected Gain if recoverable SCGF fixed: 12.97%
- Total SCGF fields: 57856

## Bounding Box Transformations in SCGF
- % Completely Replaced (<0.1 IoU): 96.16%
- % Recoverable (Rank-1 IoU >= 0.5): 100.00% (57856 fields)

### Transformation Distribution (SCGF Only)
```json
{
  "BBOX_REPLACED": 55637,
  "BBOX_EXPANDED": 1879,
  "BBOX_NARROWED": 321,
  "BBOX_PRESERVED": 19
}
```

### Transformation Distribution (All Fields)
```json
{
  "BBOX_REPLACED": 223673,
  "BBOX_EXPANDED": 139167,
  "NO_CANDIDATE": 66962,
  "BBOX_PRESERVED": 11172,
  "BBOX_NARROWED": 4976
}
```
