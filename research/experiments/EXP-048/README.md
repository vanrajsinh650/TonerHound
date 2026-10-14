# TonerHound — EXP-048: Diagnostic-First Document Repair

## Overview
EXP-048 implements a diagnostic-first document repair methodology for TonerHound. Rather than introducing generic unguided stages across the corpus, EXP-048 performs comprehensive full-distribution tail auditing, identifies the exact document-level and field-level failure clusters dragging the macro F1 down, conducts forensic root-cause analysis, and applies targeted zero-regression fixes.

## Key Outcomes
- **Baseline (EXP-047)**: Word F1: 72.8448%, Page F1: 90.0061%, Passing: 345,461
- **EXP-048 Results**:
  - **Word Grounding F1**: **72.9918%** (+0.1470 pp)
  - **Page Grounding F1**: **90.0613%** (+0.0552 pp)
  - **Passing Fields**: **352,570** (+7,109 net rescued)
  - **Failing Fields**: **145,570** (-7,109 reduction)
  - **Regressions**: **0** (Strict Zero Regressions across all 370 benchmark documents)
- **Held-Out Cohort Gate**: **PASS** (+1,349 net rescued, 0 regressions, 23.44s latency)

## Repository Structure
```
research/experiments/EXP-048/
├── README.md
├── FINAL_REPORT.md
├── before_after_comparison.md
├── decision.md
├── audit_phase_a.py
├── cluster_phase_b.py
├── forensics_phase_c.py
├── test_per_fix_phase_e.py
├── run_heldout_gate.py
├── run_full_benchmark.py
├── audit/
│   ├── distribution.json              # Full 236-doc Word F1 distribution (10 buckets)
│   ├── lift_ranking.json              # All documents ranked by lift potential
│   ├── failure_clusters.json          # Global and per-document failure clusters
│   └── root_causes.json               # Forensic deep-dive on top 5 failure clusters
├── fixes/
│   ├── fix_01_sec13f_voting_authority.py        # [ACCEPTED] SEC 13F voting authority rails
│   ├── fix_02_holdings_marks_precision.py       # [ACCEPTED] Footnote marks character precision
│   ├── fix_03_fund_security_name_single_line.py # [ACCEPTED] Fund security name single-line clamp
│   ├── fix_04_sec13f_title_of_class.py          # [REJECTED] Insufficient multi-doc reach (<2 docs)
│   └── fix_05_creditor_matrix_parser.py         # [REJECTED] Insufficient rescues (<100 fields)
└── results/
    ├── per_fix_metrics.json           # Isolated performance metrics per fix
    ├── heldout_results.json           # 32-document held-out cohort results
    ├── full_benchmark_results.json    # Official 370-document benchmark results
    ├── failure_migration.json         # Failure class migration breakdown
    ├── regression_analysis.json       # Regression verification audit
    └── per_document.csv               # Per-document official evaluation metrics
```
