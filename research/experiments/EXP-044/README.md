# EXP-044: Microscope Audit + Normalization Mismatch Assault

## Overview

EXP-044 executes a rigorous, empirical research cycle focused on the single largest remaining failure bucket in TonerHound's ExtractBench benchmark: `NORMALIZATION_MISMATCH` (~46,158 fields in the EXP-042 canonical baseline).

This experiment had two foundational objectives:
1. **PART A (Microscope Audit)**: Audit and validate the Failure Microscope (V5) through human manual labeling of 50 sample fields, adversarial verification, and historical backtesting across 4 prior benchmark cycles.
2. **PART B (Hybrid Matching Pipeline)**: Decompose `NORMALIZATION_MISMATCH` into empirical sub-classes, build a 6-stage gated matching pipeline, and measure real field recovery with strict zero regressions across the official 370-document ExtractBench benchmark.

---

## Benchmark Results Summary

- **Word Grounding F1**: **72.8081%** (Baseline EXP-042: 72.6179%, **+0.1902 pp**)
- **Page Grounding F1**: **83.9026%** (Baseline EXP-042: 83.8490%, **+0.0536 pp**)
- **Word Precision**: **77.9620%**
- **Word Recall**: **69.1778%**
- **Passing Fields**: **336,135 / 498,140** (+8,464 net rescued)
- **Regressed Fields**: **0** (Strict zero-regression guarantee verified)
- **Top Document Gain**: `real_ishares_iboxx_bond_etfs` jumped from **70.34% to 92.28%** (+21.94 pp)

---

## Directory Structure

```
research/experiments/EXP-044/
├── README.md                          # This file
├── FINAL_REPORT.md                    # Comprehensive 20-question final report
├── before_after_comparison.md         # Detailed delta metrics & top improved docs
├── decision.md                        # Formal gate decision (REFINE)
├── implementation_notes.md            # Architecture of the 6-stage pipeline
├── ablation.json                      # Phase D ablation metrics across all stages
├── run_ablation.py                    # Ablation runner script
├── run_targeted.py                    # Phase E held-out cohort runner
├── run_full_benchmark.py              # Phase F full 370-document benchmark runner
├── finalize_results.py                # Results reconciliation & artifact generation
├── audit/
│   ├── mismatch_50_sample.json        # 50 manually audited mismatch failure fields
│   └── subclass_breakdown.json        # Sub-class decomposition with counts & examples
├── microscope/
│   ├── validation_report_v5.md        # Microscope V5 validation report (Gates V1-V8)
│   ├── mismatch_subclass_breakdown.json
│   └── recovery_rate_table_v5.json    # Calibrated empirical recovery rates
├── matching/
│   ├── canonicalize_v2.py             # Stage 1: Rule-based scalar canonicalization
│   ├── fuzzy_matcher.py               # Stage 2: Character-level bounded fuzzy matcher
│   ├── token_aligner.py               # Stage 3: Needleman-Wunsch token alignment
│   ├── token_set_matcher.py           # Stage 4: Hungarian token set permutation
│   ├── multi_region_v3.py             # Stage 5: Multi-region evidence assembly
│   ├── ambiguity_resolver.py          # Stage 6: Layout ambiguity resolver
│   ├── global_assignment.py           # Stage 6: Tabular bipartite matching
│   └── unified_pipeline.py            # Gated orchestrator
└── results/
    ├── full_benchmark_results.json    # Official 370-document benchmark metrics
    ├── heldout_results.json           # Phase E held-out cohort evaluation
    ├── failure_migration.json         # Before vs after failure class migration
    ├── per_document.csv               # Complete 370-document per-doc score table
    ├── regression_analysis.json       # Regression verification audit
    └── fix_attribution.json           # Per-stage and per-document rescue attribution
```

---

## Reproduction Commands

To reproduce the full pipeline and audits:

```bash
# 1. Run Phase D Ablation Study
uv run python research/experiments/EXP-044/run_ablation.py

# 2. Run Phase F Full 370-Document Benchmark
uv run python research/experiments/EXP-044/run_full_benchmark.py

# 3. Run Microscope V4 Audit on EXP-044
uv run python research/observer/run_microscope_v4.py --run-id EXP-044 --baseline EXP-042
```
