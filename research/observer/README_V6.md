# Failure Microscope V6 — Comprehensive Per-Document Field Accounting & Causal Auditor

## Overview
Failure Microscope V6 builds directly upon Microscope V5, introducing granular per-document field accounting, automatic structural failure clustering, and complete causal audit traceability across all documents in the benchmark corpus.

> **CRITICAL DIRECTIVE CONSTRAINT**: Microscope V6 is **WRITE ONLY** in EXP-051. Execution on the full benchmark corpus is strictly reserved for **EXP-052**.

---

## What V6 Adds Over V5

### 1. Granular Per-Document Field Accounting Schema
For every document audited, V6 produces full ground-truth vs prediction accounting matching the exact specification:
```json
{
  "document_id": "short/sched_i__after_violence_project_ty2024",
  "status": "evaluated",
  "gradeable": 34,
  "passing": 33,
  "failing": 1,
  "word_f1": 97.0588,
  "page_f1": 100.0,
  "family": "IRS Form 990 Schedule I",
  "fields": [
    {
      "field_path": "recipient_table[0].grant_amount",
      "gold_value": "15000",
      "gold_page": 1,
      "gold_bbox": [0.65, 0.42, 0.08, 0.02],
      "predicted_value": "15000",
      "predicted_page": 1,
      "predicted_bbox": [0.65, 0.42, 0.08, 0.02],
      "status": "PASS",
      "iou": 1.0000
    }
  ],
  "failure_clusters": [
    {
      "cluster_id": "c_001",
      "field_pattern": "recipient_table[].grant_amount",
      "count": 1,
      "dominant_failure_class": "INSUFFICIENT_IOU"
    }
  ]
}
```

### 2. Standardized Taxonomy Mapping
Every field is assigned an unambiguous, mutually exclusive status:
- `PASS`: IoU $\ge 0.50$ on correct page.
- `INSUFFICIENT_IOU`: $0.05 \le \text{IoU} < 0.50$ on correct page (tight bounding boxes or padding truncations).
- `WRONG_ROW`: Valid text candidate located on an incorrect line ($\text{IoU} < 0.05$).
- `PAGE_DRIFT`: Candidate located on a different page than the gold citation.
- `ZERO_CANDIDATE`: No candidate extracted or field citation omitted entirely.
- `MALFORMED_BBOX`: Extracted bounding box geometry is collapsed, negative, or invalid.

### 3. Structural Failure Clustering
Fields are normalized using regex array pattern compression (`re.sub(r'\[\d+\]', '[]', field_path)`), enabling automated grouping of repeated table row failures into coherent root-cause clusters.

### 4. Validation Gate V8: Total Field Accounting
Microscope V6 introduces and validates Gate V8:
- Mathematical invariant: For every evaluated document, $\text{gradeable} \equiv \text{passing} + \text{failing}$.
- Completeness invariant: $\text{len}(\text{fields}) \equiv \text{gradeable}$.
- Zero orphaned or unaccounted fields allowed across the entire corpus.

---

## Verification & Testing
Microscope V6 is fully unit-tested via `tests/test_microscope_v6.py`, covering:
- Array pattern normalization (`compute_field_pattern`)
- Exact intersection-over-union calculation (`iou_xywh`)
- Synthetic multi-failure document auditing
- Gate V8 consistency and discrepancy rejection
