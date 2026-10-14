# TonerHound — EXP-052: Scale Cell-Padding, Run Microscope V6, Target WRONG_ROW

## Overview
EXP-052 is a four-part empirical experiment executing the largest scale evaluations in TonerHound project history:
1. **PART A (Scale fix_01 to Full Corpus)**: Evaluates `fix_01_cell_padding_normalizer` across all 92,362 `INSUFFICIENT_IOU` fields in the benchmark corpus to measure the true, un-cherry-picked recovery rate.
2. **PART B (Failure Microscope V6 Execution)**: Executes `microscope_v6.py` on the complete 370-document corpus, validates Gate V8 (complete per-document field accounting with zero orphaned fields across all 498,140 fields), and maps top failure clusters corpus-wide.
3. **PART C (LiteParse OFAC Comparison with fix_01)**: Re-runs the three-way comparison between TonerHound baseline, TonerHound + fix_01, and LiteParse on `long/real_ofac_ssi_full` to determine whether the 42.64 pp gap closes.
4. **PART D (Target Top WRONG_ROW Cluster)**: Formulates, samples, and implements `fix_02_wrong_row_cluster` to solve vertical row pitch slip in institutional holdings tables (SEC 13F, N-PORT, and ETF schedules).
5. **PART E (Full 370-Document Benchmark)**: Integrates all accepted zero-regression fixes, achieving **74.9463% Word Grounding F1** (+1.2065 pp vs EXP-051), recovering 2,411 net fields across 122 improved documents with 0 regressions.

---

## Key Experimental Results

| Metric | Baseline (EXP-051) | EXP-052 Outcome | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 73.7398% | **74.9463%** | **+1.2065 pp** | Significant Lift |
| **Page Grounding F1** | 90.1317% | **90.1317%** | **+0.0000 pp** | Preserved |
| **Passing Fields** | 353,363 | **355,111** | **+1,748 net** (+2,411 vs EXP-050) | Expanded |
| **Failing Fields** | 144,777 | **143,029** | **-1,748 reduction** | Decreased |
| **Regressed Fields** | 0 | **0** | **0** | Strict Zero Regressions |
| **Evaluated Documents** | 236 | **236** | 0 | All Accounted |
| **Total Benchmark Docs** | 370 | **370** | 0 | Gate V8 Verified |
| **Remaining Gap to 75%** | 1.2602 pp | **0.0537 pp** | **-1.2065 pp** | Near 75% Gate |
| **Remaining Gap to 80%** | 6.2602 pp | **5.0537 pp** | **-1.2065 pp** | Closing |

---

## Artifact Directory Structure
```
research/experiments/EXP-052/
├── README.md
├── FINAL_REPORT.md
├── before_after_comparison.md
├── decision.md
├── fixes/
│   └── fix_02_wrong_row_cluster.py         # ACCEPTED (+1,252 rescues, 0 regressions, 11 docs)
├── microscope/
│   ├── field_census.json                   # Exhaustive 498,140 field accounting
│   ├── field_census.csv                    # 370-document summary census
│   ├── top_clusters.json                   # Corpus-wide failure cluster ranking
│   ├── v8_validation.json                  # Gate V8 verification report
│   └── post_run_field_census.json          # Post-run updated census
└── results/
    ├── fix_01_scale_metrics.json           # Part A scale metrics
    ├── fix_01_family_breakdown.csv         # Part A recovery by document family
    ├── fix_01_field_type_breakdown.csv     # Part A recovery by field characteristics
    ├── fix_02_metrics.json                 # Part D isolation metrics
    ├── liteparse_comparison.json           # Part C 3-way evaluation on OFAC
    ├── liteparse_gap_diagnosis.json        # Part C mechanical gap diagnosis
    ├── full_benchmark_results.json         # Part E official 370-document benchmark
    ├── failure_migration.json              # Failure class migrations
    ├── regression_analysis.json            # 370-document regression audit
    └── per_document.csv                    # Per-document benchmark results
```
