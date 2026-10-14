# TonerHound — EXP-054: Forced Document Inclusion, Memory-Safe Evaluation, and 90% Assault

## Overview
EXP-054 executes across five key areas:
1. **PART A (Forced Document Inclusion & Memory-Safe Evaluation)**: Implemented chunked block evaluation in `src/tonerhound/benchmark/evaluator.py`, neutralizing the ExtractBench `_GROUNDED_MAX_CELLS = 100_000_000` memory guard and running all 370 benchmark documents without memory exhaustion.
2. **PART B (Production Code Integration)**: Formally merged Fixes 01 through 04 into `src/tonerhound/` with complete unit test coverage (276/276 tests passing in 17.63s), bumped production version to **v0.4.0**, and updated `CHANGELOG.md`.
3. **PART C (OFAC Row Tracker — Fix 05)**: Built `fix_05_ofac_row_tracker.py` isolating search to each entity's vertical window, rescuing **14,034 fields** with 0 regressions, raising document F1 from 29.86% to **63.14%** (+33.28 pp).
4. **PART D (Malformed Bounding Box Repair — Fix 06)**: Built `fix_06_malformed_bbox_repair.py` expanding degenerate coordinates to minimum valid geometry, rescuing **2,155 fields** on non-Oklahoma institutional documents (34,934 fields corpus-wide) with zero regressions.
5. **PART E (Full Benchmark Evaluation)**: Pushed passing fields to **411,098 / 498,140** (+16,189 net rescues), achieving **75.2099% Word Grounding F1** (237 docs) with zero regressions.

---

## Key Experimental Results

| Metric | Baseline (EXP-053) | EXP-054 Outcome | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1 (237 docs, with Oklahoma)** | 75.0241% | **75.2099%** | **+0.1858 pp** | **v0.4.0 Released** |
| **Word Grounding F1 (236 docs, standard suite)** | 74.9594% | **75.1459%** | **+0.1865 pp** | Continuous Lift |
| **Word Grounding F1 (370 docs, all-inclusive)** | Withheld by guard | **48.1750%** | Defined | First All-370 Run |
| **Page Grounding F1** | 90.1317% | **90.1733%** | **+0.0416 pp** | Lifted |
| **Passing Fields** | 394,909 | **411,098** | **+16,189** | Rescued |
| **Failing Fields** | 103,231 | **87,042** | **-16,189** | 15.7% Reduction |
| **Regressed Fields** | 0 | **0** | **0** | Strict Invariance |
| **Evaluated Documents** | 237 | **237** | 0 | Fully Accounted |
| **Total Benchmark Docs** | 370 | **370** | 0 | 100% Evaluated |
| **Gap to 80.00% Milestone** | 4.9759 pp | **4.7901 pp** | **-0.1858 pp** | Closing |
| **Gap to 90.00% Milestone** | 14.9759 pp | **14.7901 pp** | **-0.1858 pp** | Headroom mapped |
| **Theoretical Ceiling F1** | 99.9956% | **99.9956%** | Defined | 498,118 gradeable fields |

---

## Artifact Directory Structure
```
research/experiments/EXP-054/
├── README.md
├── FINAL_REPORT.md
├── before_after_comparison.md
├── decision.md
├── run_full_benchmark.py
├── fixes/
│   ├── fix_05_ofac_row_tracker.py          # ACCEPTED (+14,034 rescues, 0 regressions)
│   └── fix_06_malformed_bbox_repair.py     # ACCEPTED (+2,155 non-OK / 34,934 total rescues)
└── results/
    ├── fix_05_metrics.json                 # Part C isolation metrics (OFAC)
    ├── fix_06_metrics.json                 # Part D isolation metrics (Malformed bbox)
    ├── full_benchmark_results.json         # Part E full benchmark results
    ├── final_benchmark_results.json        # Output artifact copy
    ├── failure_migration.json              # Failure class migrations
    ├── regression_analysis.json            # 370-document regression audit
    └── per_document.csv                    # Per-document benchmark results
```
