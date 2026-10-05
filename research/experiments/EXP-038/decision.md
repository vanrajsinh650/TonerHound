# EXP-038 Decision Document

## Status: SUPPORTED — VALIDATED FOR PRODUCTION INTEGRATION

**Date:** 2026-10-04  
**Git Commit:** `178f81c`  
**Run ID:** `exp038_full_run_v1`  
**Target Gate:** Production Integration Gate (Zero-Regression + Net F1 Gain + Microscope Calibration)

---

## 1. Executive Summary

EXP-038 is **SUPPORTED** for integration into production `src/tonerhound/`. It delivers a **+1.4411 pp Word Grounding F1** improvement and **+0.2964 pp Page Grounding F1** improvement across the full 370-document ExtractBench benchmark with **zero field regressions** across 498,140 evaluated fields.

Failure Microscope V4 has been officially calibrated and validated across 8 formal gates (V1–V8), with backtesting demonstrating that empirical realization rates match actual benchmark gains within 2x.

---

## 2. Gate Verification Matrix

| Validation Gate | Requirement | Measured Result | Verdict |
| :--- | :--- | :--- | :---: |
| **V1: Internal Consistency** | 100% reconciliation of failure counts | 100% (309/309 sample check) | **PASS** |
| **V2: Manual Audit** | $\ge 95\%$ agreement on 30 random failures | 100.0% (30/30) | **PASS** |
| **V3: Adversarial Injection** | 100% detection of corrupted test fixtures | 100.0% (12/12) | **PASS** |
| **V4: Cross-Reference** | $\ge 85\%$ agreement with EXP-035 labels | 88.64% (507/572) | **PASS** |
| **V5: Cardinality Sanity** | Exact match on known failure subsets | 286 bool, 38 date, 22 norm, 25 real index | **PASS** |
| **V6: Golden Set** | $\ge 98\%$ agreement on 100 gold cases | 100.0% (100/100) | **PASS** |
| **V7: Reproducibility** | Bit-for-bit SHA256 identical outputs | Verified across all 4 report JSONs | **PASS** |
| **V8: Backtesting** | Expected gain within 2x of actual gain | EXP-035: 1.01x, EXP-036D: 1.00x, EXP-037: 1.20x, EXP-038: 1.92x | **PASS** |
| **Phase E: Held-Out Target**| Net rescued $\ge 200$, regressed $== 0$ | +215 rescued, 0 regressed | **PASS** |
| **Phase F: Full Benchmark** | Net Word F1 delta $> 0$, regressed $== 0$ | **+1.4411 pp**, 0 regressed | **PASS** |

---

## 3. Evidence-Based Integration Plan

The following components from EXP-038 are approved for production porting into `src/tonerhound/`:

1. **Accounting Negative Normalizer (`normalization_v2.py`)**:
   - Port parenthesized negative value generator `generate_numeric_query_variants()` into `src/tonerhound/matching/normalizer.py`.
   - Adds negligible CPU overhead; responsible for 494 rescued fields (+0.59 pp Word F1).
2. **OCR-Noise Inverted Indexing (`ocr_noise_index.py`)**:
   - Integrate bounded 3-gram candidate retrieval into `src/tonerhound/document/hybrid_index.py` strictly conditioned on `page_mode == 'ocr'`.
   - Responsible for 16 rescued fields on degraded scans without digital text regression.
3. **Date Normalizer Regex Expansion (`date_normalizer.py`)**:
   - Port `parse_extended_date()` to `src/tonerhound/matching/matcher.py`.
4. **Visual Checkbox Provider V2 (`visual_provider_v2.py`)**:
   - Integrate morphological wireframe provider with page-level contour caching into `src/tonerhound/vision/checkbox.py` behind strict boolean field gating (`is_boolean_target`).

---

## 4. Arithmetic Gap to 90%

- **Current Benchmark Score:** 58.1118% Word F1
- **Target Score:** 90.0000% Word F1
- **Remaining Arithmetic Gap:** 31.8882 pp
- **Failure Microscope V4 Realistic Ceiling:** +3.4249 pp across remaining deterministic classes.
- **Structural Limitation:** Reaching 90% solely through local string-matching and morphological heuristics is mathematically bounded. Closing the remaining 31.9 pp gap requires structural table grid graph reconstruction and layout-aware multi-column alignment.
