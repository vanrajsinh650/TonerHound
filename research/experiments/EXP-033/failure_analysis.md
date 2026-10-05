# EXP-033: Failure & Root Cause Analysis

**Experiment**: EXP-033 (Candidate Generation Reconciliation Audit + Productionization)  
**Date**: October 2026  
**Final Production F1**: 56.0548% (Baseline: 56.0477%, Delta: +0.0071 pp)  
**Target Gate**: GATE D — FAILURE (< 60.00% Word Grounding F1)

---

## 1. Executive Failure Summary

EXP-033 performed the first rigorous reconciliation audit of the 48,744 zero-candidate field population and evaluated 4 targeted candidate-generation recovery mechanisms across the official 370-document ExtractBench benchmark.

While the candidate-generation fixes successfully upgraded 34 fields across 24 documents (e.g., restoring multi-line service lists in `real_bbb_service_list`, oil/gas depths in Wyoming filings, and OFAC entities), the macro Word Grounding F1 moved by only **+0.0071 pp** (from 56.0477% to 56.0548%), falling short of the 58.11% LlamaExtract target (-2.0552 pp).

This failure decisively falsifies the hypothesis that candidate-generation truncation or rigid routing on text fields is the primary ceiling-limiting factor in production.

---

## 2. Root Cause Breakdown

### A. The 48,744 Population is Heavily Skewed by Corrupted Table Noise (76.5%)
The reconciliation audit of 200 random fields from the 48,744 zero-candidate population revealed:
- **76.5% (~37,289 fields)** reside inside two long, corrupted creditor matrices: `real_imedia_full_corrupted` (8,000+ rows) and `real_ftx_full_corrupted` (7,000+ rows).
- In these documents, character-level OCR degradation prevents exact text tokenization entirely (e.g. `'1701 N GAFFEY ST'` transcribed as `'7700 GARTH B ROOK!'`).
- Because ExtractBench scores macro-averaged Word F1 across documents, these two documents account for only **0.84%** of the benchmark metric. Therefore, resolving or failing on these 37,289 fields has virtually no impact on macro F1.

### B. Table Cell Domination (>90% of Benchmark Mass)
- Out of 445,950 total gradeable benchmark fields, over 400,000 are table cells.
- Table cells cannot be grounded by independent single-query candidate generation. In dense tabular filings, numbers like `0.00`, `100.00`, and states like `CA`, `NY` appear dozens of times per page.
- As established in EXP-031A and EXP-032, independent candidate retrieval without joint Hungarian table row assignment creates severe row-assignment collisions that the official evaluator penalizes.

### C. Precision Penalty on Borderline Candidates
- In the official ExtractBench evaluator, emitting a candidate citation that achieves $\text{IoU} < 0.50$ incurs a strict precision penalty:
  $$g\_claims \leftarrow g\_claims + 1, \quad g\_correct \leftarrow g\_correct + 0$$
- In documents with dense form layouts (such as `short/W14-Atascosa SWD Well No. 4` and `short/W2-27-217884_213729`), expanding boolean and token-stripped candidates successfully generated candidate bounding boxes, but imperfect spatial alignment (IoU between 0.35 and 0.48) caused precision reductions that partially offset recall gains.

### D. The Disconnect Between the 75.12% Oracle and Production
- The 75.12% EXP-028B0 True Text Oracle was an unconstrained ceiling experiment: for every ground-truth field, it searched all text tokens across all pages and selected *any* token with $\text{IoU} \ge 0.50$.
- In production, candidate selection does not know ground-truth coordinates. Without joint structural assignment and spatial cell geometry, the production selector cannot reliably distinguish between identical candidates on a page.

---

## 3. Reconciled Architectural Bottleneck Map

```text
Theoretical Maximum: 100.00%
      │
      ├── Visual / Non-Text / Severe OCR Ceiling: ~75.12% (EXP-028B0 Oracle)
      │     └── 24.88% ungroundable without OCR repair & visual grounding
      │
      ├── Selection Ceiling on Current Pool: 60.9591% (EXP-032 Oracle)
      │     └── 14.16 pp lost to candidate pool deficiency & table collision
      │
      ├── LlamaExtract Target: 58.1100%
      │
      ├── Production Reality (EXP-033): 56.0548%
      │     └── Selection & Table Assignment Gap: ~4.90 pp
      └── Production Baseline (EXP-032): 56.0477%
```

---

## 4. Key Takeaways

1. **Candidate generation fixes on scalar text are sound but insufficient**: All 4 fixes (boolean expansion, token strip fallback, global search relaxation, multi-line recovery) work correctly and are 100% test-verified with zero regressions, but scalar fields represent a minor fraction of benchmark loss.
2. **Table assignment is the actual structural bottleneck**: The +4.91 pp selection headroom measured in EXP-032 and the multi-thousand cell dropout in tables can only be recovered by joint row-level table assignment.
3. **Target 58.11% is reachable within the existing 60.96% selection ceiling**: We do NOT need visual grounding to beat 58.11%. Closing just 2.06 pp of the 4.91 pp selection headroom through joint record-level table resolution will achieve $\ge 58.11\%$.
