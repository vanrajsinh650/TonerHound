# EXP-028B0: True Text-Only Grounding Ceiling Report

**Date:** 2026-09-24 10:57:27 UTC  
**Status:** COMPLETE & FROZEN  
**Benchmark:** Official ExtractBench Evaluator (Completely Unchanged)  
**Denominators:** N=236 Grounded Documents (Official Macro Denominator), N=445,950 Gradeable Fields  

---

## 1. Executive Summary & Comparison

| Metric / Configuration | Word Grounding F1 | Word Precision | Word Recall | Page Grounding F1 | Delta vs Old Ceiling B | Delta vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Production Baseline (EXP-026)** | **55.98%** | 62.11% | 52.26% | 81.28% | -8.60pp | — |
| **Old Ceiling B (EXP-027)** | **64.58%** | 70.95% | 60.66% | 82.90% | — | +8.60pp |
| **TRUE TEXT ORACLE (EXP-028B0)** | **75.12%** | **81.23%** | **71.26%** | **83.85%** | **+10.55pp** | **+19.15pp** |

---

## 2. Cohort Performance

### A. Cohort B: Held-Out (32 Documents, 1,324 Pages — Never Tuned)
| Metric | Production Baseline | Old Ceiling B | TRUE TEXT ORACLE | Delta vs Old Ceiling B |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 59.27% | 70.05% | **77.05%** | **+7.00pp** |
| **Word Precision** | 63.99% | 74.47% | **80.49%** | **+6.02pp** |
| **Word Recall** | 56.19% | 67.20% | **74.68%** | **+7.48pp** |
| **Page Grounding F1** | 86.61% | 90.46% | **91.38%** | **+0.92pp** |

### B. Cohort A: Development (32 Documents, 881 Pages)
| Metric | Production Baseline | Old Ceiling B | TRUE TEXT ORACLE | Delta vs Old Ceiling B |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 65.45% | 76.09% | **83.25%** | **+7.16pp** |
| **Word Precision** | 67.35% | 78.01% | **84.93%** | **+6.92pp** |
| **Word Recall** | 64.00% | 74.53% | **81.86%** | **+7.33pp** |
| **Page Grounding F1** | 92.69% | 94.68% | **95.35%** | **+0.67pp** |

---

## 3. Length Splits Performance
- **Short Documents:** 71.54% Word F1 (85.83% Page F1)
- **Medium Documents:** 84.75% Word F1 (79.22% Page F1)
- **Long Documents:** 87.75% Word F1 (81.20% Page F1)

---

## 4. Key Findings & Strategic Implications

1. **Was 64.58% a Real Ceiling or an Artificially Truncated Ceiling?**
   It was an **artificially truncated ceiling**. Old Ceiling B failed to recover thousands of legitimate text fields because of top-25 candidate pool limits, rigid page pruning, and multi-line token search failures.

2. **The True Reach of Text-Only Grounding:**
   The True Text Oracle achieves **75.12% Word Grounding F1**.
   This demonstrates that a massive portion of the grounding gap can be closed purely through untruncated candidate generation and improved text spatial alignment, without requiring a visual model for these fields.

3. **Next Steps:**
   - If True Text Ceiling < 90%: Advance to Table/Cell Oracle, followed by Visual/Non-Text Oracle for remaining visual-only fields (checkboxes, degraded scans).
