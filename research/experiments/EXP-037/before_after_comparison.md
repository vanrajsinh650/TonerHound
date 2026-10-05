# EXP-037: Before vs After Benchmark Comparison

## Baseline vs After Execution

- **Baseline Run ID:** `canonical_370_v1` (Git commit `178f81c`)
- **After Run ID:** `exp037_full_run_v1` (Git commit `178f81c`)
- **Evaluator:** Official ExtractBench `ExtractEvaluator`

---

## 1. Aggregate Headline Metrics

| Metric | Baseline (`canonical_370_v1`) | After (`exp037_full_run_v1`) | Measured Delta |
| :--- | :---: | :---: | :---: |
| **Word Grounding F1** | **56.0477%** | **56.6707%** | **+0.6230 pp** |
| **Page Grounding F1** | **81.6639%** | **81.9786%** | **+0.3147 pp** |
| **Word Grounding Precision** | **61.7275%** | **62.4511%** | **+0.7236 pp** |
| **Word Grounding Recall** | **52.5686%** | **53.1233%** | **+0.5547 pp** |
| **Passing Fields ($IoU \ge 0.50$)** | 307,373 | 307,514 | **+141 fields** |
| **Failing Fields** | 190,767 | 190,626 | **-141 fields** |
| **Fields Rescued** | 0 | 141 | **+141 fields** |
| **Fields Regressed** | 0 | 0 | **0 fields** |
| **Net Field Change** | 0 | +141 | **+141 fields** |

---

## 2. Document-Level Outcomes

- **Total Documents in Benchmark:** 370
- **Total Evaluated Documents with Grounding Rules:** 236
- **Documents Modified by OCR Routing:** 130
- **Documents with Word F1 Improved:** 45 documents (34.62% of modified cohort)
- **Documents with Word F1 Regressed:** 3 documents (slight precision denominator effect from generating candidates with $IoU < 0.50$)
- **Documents with Word F1 Unchanged:** 82 documents (63.08% of modified cohort)

---

## 3. Top Rescued Documents

| Document ID | Family | Before Word F1 | After Word F1 | Delta (pp) | Fields Rescued |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `short/P4-83-631_246` | RRC_OIL_GAS_FORMS | 8.33% | 27.78% | +19.44 pp | 7 |
| `short/2A-9-160258_109909` | RRC_OIL_GAS_FORMS | 31.19% | 42.20% | +11.01 pp | 7 |
| `short/P4-27-51022_74198` | RRC_OIL_GAS_FORMS | 16.84% | 27.37% | +10.53 pp | 5 |
| `short/2A-9-160282_109921` | RRC_OIL_GAS_FORMS | 32.73% | 41.82% | +9.09 pp | 5 |
| `short/passcoag-2020-w2-p0003-r4` | IRS_TAX_FORMS | 54.90% | 62.75% | +7.84 pp | 2 |
| `short/2A-9-160266_109913` | RRC_OIL_GAS_FORMS | 29.41% | 35.29% | +5.88 pp | 3 |
| `short/2A-9-160318_109939` | RRC_OIL_GAS_FORMS | 31.07% | 36.89% | +5.83 pp | 3 |
| `short/2A-9-160298_109929` | RRC_OIL_GAS_FORMS | 46.15% | 51.92% | +5.77 pp | 3 |
| `medium/becerra-2022` | IRS_TAX_FORMS | 35.76% | 37.81% | +2.05 pp | 10 |
| `medium/becerra-2021` | IRS_TAX_FORMS | 28.51% | 30.48% | +1.97 pp | 8 |
| `medium/bar-lev-2022` | IRS_TAX_FORMS | 33.15% | 34.73% | +1.58 pp | 4 |
| `medium/bar-lev-2023` | IRS_TAX_FORMS | 30.55% | 31.83% | +1.28 pp | 3 |
| `medium/arif-2023` | IRS_TAX_FORMS | 28.85% | 30.30% | +1.45 pp | 3 |
| `medium/real_enotes_deviations_corrupted` | OTHER | 69.12% | 71.84% | +2.72 pp | 29 |

---

## 4. Operational Cost and Latency Comparison

| Stage | Baseline Time (`canonical_370_v1`) | EXP-037 After Time (`exp037_full_run_v1`) | Overhead |
| :--- | :---: | :---: | :---: |
| **Phase C Targeted Test (130 docs)** | N/A | 777.23s (12.95 min) | First-time OCR + Resolution |
| **Phase D Official Evaluation (370 docs)** | 2.5s (cached) | 379.7s (6.33 min) | Re-evaluating 130 modified docs |
| **Phase E Failure Microscope Audit (370 docs)** | 145.2s (2.42 min) | 217.0s (3.62 min) | Causal tracing across all fields |
| **Total Full Run Latency** | 147.7s | 596.7s (9.94 min) | Safe, single-threaded |
