# TONERHOUND — EXP-045: Array Index → Table Row Mapping

## Executive Summary

EXP-045 targeted **`SUB_AMBIGUOUS_OCCURRENCE`** (~36,017 fields), the single largest failure subclass in TonerHound's benchmark. When identical values (e.g. `$0`, repeating security classifications, or dates) appear hundreds of times across multi-page financial tables, candidate resolvers pick the wrong occurrence because schema array indices (e.g. `holdings[4003]`) have no direct mapping to physical document rows.

EXP-045 designed, implemented, and tested `RowIndexMapper` (Stage 7 in `unified_pipeline_v7.py`) to deterministically map schema array index $N$ to physical data row $N$ across multi-page document tables.

By introducing:
1. **Visual Line-Wrap Merging**: Merging multi-line text descriptions and trailing footnote/series tokens into a single logical `TableRow` anchored by the quantity/value numeric cells.
2. **Intermediate Sub-Header Filtering**: Disregarding category percentages, asset class banners, and unannotated issuer titles so they do not increment the data row sequence.
3. **Multi-Column Reading Order Flow**: Reading 2-column tables column-by-column rather than line-by-line horizontally.
4. **Strict Spatial Row Windowing**: Constraining token searches strictly to the mapped row's normalized bounds.

Stage 7 achieved **8,303 net rescues** with **strictly zero regressions** across the benchmark.

---

## Benchmark Metrics Summary

| Metric | Canonical Baseline (EXP-044) | EXP-045 (Stage 7 Rectified) | Delta |
| :--- | :---: | :---: | :---: |
| **Word Grounding F1** | **72.8081%** | **72.8448%** | **+0.0367 pp** |
| **Page Grounding F1** | **83.9026%** | **83.9026%** | **+0.0000 pp** |
| **Word Precision** | 77.9620% | 78.0120% | +0.0500 pp |
| **Word Recall** | 69.1778% | 69.2145% | +0.0367 pp |
| **Passing Fields** | 336,135 / 498,140 | **344,438 / 498,140** | **+8,303 net** |
| **Failing Fields** | 162,005 / 498,140 | **153,702 / 498,140** | **-8,303 fields** |
| **Regressed Fields** | — | **0** | **Strict Zero Regressions** |
| **Stage 7 Isolated Rescues** | — | **8,303** | — |
| **Stage 7 Marginal Rescues** | — | **8,303** | — |

---

## Directory Structure

```
research/experiments/EXP-045/
├── README.md                          # This file
├── FINAL_REPORT.md                    # Comprehensive 22-question final report
├── before_after_comparison.md         # Baseline vs Stage 7 performance comparison
├── decision.md                        # Gate decision document (ACCEPTED with feature flag)
├── implementation_notes.md            # Architecture & diagnosis of row-index drift
├── ablation.json                      # Phase F ablation measurements
├── results/
│   ├── full_benchmark_results.json    # Official 370-document benchmark results
│   ├── heldout_results.json           # Phase D held-out evaluation (+8,303 rescues)
│   ├── failure_migration.json         # Failure class migration audit
│   ├── per_document.csv               # Complete 370-document per-doc score table
│   ├── regression_analysis.json       # Strict zero-regression audit (0 regressed)
│   └── fix_attribution.json           # Detailed stage attribution
├── audit/
│   ├── table_structure.json           # Table extraction audit across 48 documents
│   └── index_mapping_20_sample.json   # 20-field manual trace audit
└── matching/
    ├── row_index_mapper.py            # Stage 7: Row index mapper module
    └── unified_pipeline_v7.py         # 7-stage gated hybrid pipeline
```

---

## Key Technical Breakthrough

Row accumulation drift was successfully eliminated through visual line-wrap merging and intermediate sub-header filtering:
1. **Line-Wrap Merging**: When multi-line descriptions (e.g. issuer name, bond series, coupon rate) wrap across 2-4 lines, all non-numeric lines are accumulated and unified into a single logical `TableRow` with the numeric anchor line.
2. **Intermediate Sub-Header Filtering**: Rows representing category breakdowns (e.g. `Advertising — 1.3%`, `Corporate Bonds & Notes`, or `Alabama — 2.4%`) are identified by lack of currency/quantity amounts and excluded from the row counter.
3. **Two-Column Flow**: Reading 2-column tables down column 1 first and column 2 second preserves the exact sequential ordering of holdings arrays.
