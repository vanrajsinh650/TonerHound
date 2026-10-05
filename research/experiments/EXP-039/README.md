# EXP-039: Maximum Grounding Integration

This directory contains the code, evaluation adapters, test runners, and benchmark outputs for **TonerHound EXP-039**.

---

## Benchmark Highlights

- **Evaluator**: Official `ExtractBench` `ExtractEvaluator` harness.
- **Word Grounding F1**: **69.1327%** ($\mathbf{+11.0209\text{ pp}}$ vs EXP-038 baseline $58.1118\%$)
- **Page Grounding F1**: **83.2886%** ($\mathbf{+1.0136\text{ pp}}$ vs EXP-038 baseline $82.2750\%$)
- **Word Precision**: **74.7762%** ($\mathbf{+10.8029\text{ pp}}$)
- **Word Recall**: **65.1596%** ($\mathbf{+10.6510\text{ pp}}$)
- **Net Fields Rescued**: **+6,586 fields**
- **Field Regressions**: **0** (Zero regressions verified)

---

## Directory Structure

```
research/experiments/EXP-039/
├── README.md                      # This reproduction guide
├── FINAL_REPORT.md                # Comprehensive final experiment report
├── before_after_comparison.md     # Document-by-document forensic analysis
├── decision.md                    # Ship / revise / abandon decision memorandum
├── implementation_notes.md        # Technical implementation & engineering journal
├── full_benchmark_results.json    # Complete official 370-document benchmark results
├── fix_attribution.json           # Per-phase rescue attribution breakdown
├── failure_migration.json         # Failure class migration matrix
├── regression_analysis.json       # Zero-regression verification report
├── success_analysis.json          # Rescued field samples and statistics
├── per_document.csv               # CSV of per-document scores across 370 files
├── official_evaluator_adapter.py  # Official ExtractBench adapter
├── table_cell_grounding.py        # Phase B: Table cell bounding box grounder
├── ocr_noise_index.py             # Phase C: 3-gram bounded Levenshtein index
├── multiline_assembler.py         # Phase D: Multi-line bounding box union assembler
├── visual_fallback.py             # Phase E: Deterministic pixel-statistics classifier
├── global_assignment.py           # Phase F: Hungarian bipartite table assigner
├── unified_harness_v2.py          # Unified multi-technique cascading resolver
├── run_targeted.py                # Phase G.1 Held-out 32-document test runner
├── run_full_benchmark.py          # Phase G.2 Full 370-document benchmark runner
└── results/
    ├── heldout_results.json       # Phase G.1 held-out metrics
    ├── heldout_predictions/       # Held-out prediction JSONs
    └── heldout_eval/              # Held-out cached evaluation JSONs
```

---

## Reproduction Instructions

### 1. Verify Official Evaluator Baseline
```bash
./.venv/bin/python research/experiments/EXP-039/official_evaluator_adapter.py
```
*Expected: Confirms EXP-038 baseline at exact 58.1118% Word F1 ($0.0000$ pp delta).*

### 2. Run Targeted Held-Out Evaluation (32 documents)
```bash
./.venv/bin/python research/experiments/EXP-039/run_targeted.py
```
*Expected: Rescues +832 fields with 0 regressions ($61.9085\% \rightarrow 69.7217\%$).*

### 3. Run Full 370-Document Official Benchmark Run
```bash
./.venv/bin/python research/experiments/EXP-039/run_full_benchmark.py
```
*Expected: Produces 69.1327% Word F1, +6,586 fields rescued, 0 regressions, and outputs Failure Microscope V4 report.*
