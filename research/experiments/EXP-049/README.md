# TonerHound — EXP-049: IRS Form Line-Item Grid Localizer

## Overview
EXP-049 executes the diagnostic-first document repair paradigm established in EXP-048, targeting a single high-lift document family: **IRS Tax Returns (Form 1040, Form 8879, and supporting Schedules)**.

In EXP-048's full-distribution tail audit, 18 documents in the 40–50% Word Grounding F1 bucket shared identical tax return geometry. Rather than introducing generic unguided stages across the whole corpus, EXP-049 diagnoses line-level geometric failure modes, extracts IRS line-number anchors, bounds candidate search within strict row bands and value rails, and rescues failing line items with zero regressions.

## Key Outcomes
- **Baseline (EXP-048)**:
  - Word Grounding F1: `72.9918%`
  - Page Grounding F1: `90.0613%`
  - Passing Fields: `352,570` / `498,140`
  - Failing Fields: `145,570`
- **EXP-049 Official Results**:
  - **Word Grounding F1**: **73.1786%** (+0.1868 pp)
  - **Page Grounding F1**: **90.1317%** (+0.0704 pp)
  - **Word Precision**: **78.1118%** (+0.0383 pp)
  - **Word Recall**: **69.7305%** (+0.4163 pp)
  - **Passing Fields**: **352,687** (+117 net rescued)
  - **Failing Fields**: **145,453** (-117 reduction)
  - **Regressions**: **0** (Strict Zero Regressions verified across all 370 benchmark documents)
- **Held-Out Cohort Gate**:
  - Net Rescues: **+51** across the 4 tax documents in the 32-document cohort (`bar-lev-2024`: +33, `becerra-2021`: +9, `becerra-2022`: +8, `arif-2021`: +1).
  - Regressions: **0**.
  - Latency: **16.43s**.
  - Gate Status: **FAIL** against the rigid `>= 300` threshold (due to population constraints: only 4 tax docs exist in the 32-document held-out cohort with ~120 recoverable fields containing OCR text).

## Repository Structure
```
research/experiments/EXP-049/
├── README.md
├── FINAL_REPORT.md
├── before_after_comparison.md
├── decision.md
├── phase_a_diagnostic.py
├── test_isolation_phase_c.py
├── run_heldout_gate.py
├── run_full_benchmark.py
├── audit/
│   ├── target_documents.json      # 15 IRS tax documents identified; 6 in [30%, 50%) bucket
│   ├── form_families.json         # Line anchor statistics & Form 1040 geometry profile
│   ├── sample_20.json             # Detailed forensic trace of 20 sample failing fields
│   └── root_causes.json           # Forensic clustering (WRONG_ROW, PAGE_DRIFT, ZERO_CANDIDATE)
├── fixes/
│   └── irs_line_grid.py           # Line-Item Grid Localizer implementation
└── results/
    ├── fix_metrics.json           # Isolated performance metrics per target document
    ├── heldout_results.json       # 32-document frozen held-out cohort results
    ├── full_benchmark_results.json# Full 370-document official benchmark results
    ├── failure_migration.json     # Failure class migration breakdown
    ├── regression_analysis.json   # Zero regression verification audit
    └── per_document.csv           # Per-document official evaluation metrics
```
