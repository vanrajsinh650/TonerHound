# EXP-044 Decision Document

## Status: REFINE

### 1. Decision Summary
We recommend **REFINE** for the EXP-044 multi-stage hybrid matching pipeline prior to unconditional merge into `src/tonerhound/`.

While the pipeline succeeded in rescuing **+8,464 fields** with **strictly zero regressions** across the official 370-document ExtractBench benchmark and achieved a positive macro delta (**+0.1902 pp Word Grounding F1**, **+0.0536 pp Page Grounding F1**), the causal audit revealed a fundamental structural insight:

1. **Failure Diagnosis**: The ~46,000 `NORMALIZATION_MISMATCH` failure bucket was primarily an artifact of diagnostic conflation in Microscope V4. Over 76.8% of these failures (`SUB_AMBIGUOUS_OCCURRENCE`, 36,017 fields) are **not format variations**, but repeated values in dense tabular environments (e.g., repeating corporate bond holding descriptions, zeros, ratings).
2. **Deterministic Efficacy**: Bounded fuzzy matching (Stage 2) and layout ambiguity resolution / global Hungarian matching (Stage 6) proved to be the most impactful stages, recovering 7,572 and 797 fields respectively.
3. **Production Safety**: Zero regressions were verified across all 370 documents. However, before replacing production core adapters, the layout ambiguity resolver must be coupled natively with TonerHound's table row segmenter to eliminate multi-line wrap ambiguities in complex financial reports.

---

### 2. Validation Gate Checklist

| Gate | Requirement | Actual Result | Status |
| :--- | :--- | :---: | :---: |
| **Microscope V5 Audit (V1-V7)** | All sanity and consistency checks pass | Passed 100% | **PASS** |
| **Microscope V5 Backtesting (V8)** | Actual/expected ratio $\in [0.3x, 3.0x]$ on 4 prior runs | All 4 runs within range | **PASS** |
| **Held-Out Evaluation Cohort** | Rescued $\ge 500$ net, 0 regressed | +8,402 net, 0 regressed | **PASS** |
| **Full 370 Benchmark Regressions** | Strictly 0 regressed fields | 0 regressed fields | **PASS** |
| **Full 370 Benchmark Rescues** | Measurable net positive recovery | +8,464 fields rescued | **PASS** |
| **Production Code Isolation** | `src/tonerhound/` completely unmodified | Verified untouched | **PASS** |

---

### 3. Next Strategic Priority
**TABULAR ROW-STRUCTURE INTEGRATION**: Combine Stage 2 bounded fuzzy matching and Stage 6 Hungarian table assignment with TonerHound's visual row-clustering pipeline (from EXP-028/039) to address the remaining 36,000+ `SUB_AMBIGUOUS_OCCURRENCE` repeated field occurrences.
