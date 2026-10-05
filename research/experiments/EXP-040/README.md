# EXP-040: Direct Assault on 90% Word Grounding F1

This directory contains the code, forensic audit artifacts, test runners, and official benchmark outputs for **TonerHound EXP-040**.

---

## Benchmark Highlights

- **Evaluator**: Official `ExtractBench` `ExtractEvaluator` harness.
- **Word Grounding F1**: **70.3894%** ($\mathbf{+1.2567\text{ pp}}$ vs EXP-039 baseline $69.1327\%$; $\mathbf{+14.3417\text{ pp}}$ vs Canonical V1)
- **Page Grounding F1**: **83.5835%** ($\mathbf{+0.2949\text{ pp}}$ vs EXP-039 baseline $83.2886\%$)
- **Word Precision**: **75.9234%** ($\mathbf{+1.1472\text{ pp}}$)
- **Word Recall**: **66.5126%** ($\mathbf{+1.3530\text{ pp}}$)
- **Net Fields Rescued**: **+4,817 fields**
- **Field Regressions**: **0** (Zero regressions verified across 498,140 fields)
- **Passing Fields**: **320,159** (out of 498,140)
- **Decision Gate**: **REFINE AND RETRY** (Score: 70.3894%, threshold: 70.00–74.00%)

---

## Directory Structure

```
research/experiments/EXP-040/
├── README.md                      # This reproduction guide
├── FINAL_REPORT.md                # Comprehensive final experiment report
├── before_after_comparison.md     # Document-by-document forensic analysis
├── decision.md                    # Ship / revise / abandon decision memorandum
├── implementation_notes.md        # Technical implementation & engineering journal
├── forensic_audit.json            # 500-sample failure microscope forensic audit
├── full_benchmark_results.json    # Complete official 370-document benchmark results
├── fix_attribution.json           # Per-phase rescue attribution breakdown
├── failure_migration.json         # Failure class migration matrix
├── regression_analysis.json       # Zero-regression verification report
├── success_analysis.json          # Rescued field samples and statistics
├── per_document.csv               # CSV of per-document scores across 370 files
├── forensic_audit.py              # Phase A forensic audit runner
├── multi_token_matcher.py         # Phase B: Multi-token sequence matcher
├── normalization_variants.py      # Phase C & E.2: Normalization variants generator
├── multi_region_assembler.py      # Phase D: Multi-region proximity clustering
├── hyphen_joiner.py               # Phase E.1: Broken line trailing hyphen joiner
├── unified_harness_v3.py          # Phase F: Unified multi-technique cascading resolver
├── run_targeted.py                # Phase G.1 Held-out 32-document test runner
├── run_full_benchmark.py          # Phase G.2 Full 370-document benchmark runner
└── results/
    ├── heldout_results.json       # Phase G.1 held-out metrics
    ├── heldout_predictions/       # Held-out prediction JSONs
    └── heldout_eval/              # Held-out cached evaluation JSONs
```

---

## Reproduction Instructions

### 1. Run Forensic Audit (Phase A)
```bash
./.venv/bin/python research/experiments/EXP-040/forensic_audit.py
```
*Expected: Samples 500 fields each for REAL_INDEXING_MISS, NORMALIZATION_MISMATCH, and NO_TEXT_AT_GOLD_REGION, generating `forensic_audit.json`.*

### 2. Run Targeted Held-Out Evaluation (32 documents)
```bash
./.venv/bin/python research/experiments/EXP-040/run_targeted.py
```
*Expected: Rescues +759 fields with 0 regressions ($69.7217\% \rightarrow 71.9185\%$).*

### 3. Run Full 370-Document Official Benchmark Run
```bash
./.venv/bin/python research/experiments/EXP-040/run_full_benchmark.py
```
*Expected: Produces 70.3894% Word Grounding F1, +4,817 fields rescued, 0 regressions, and outputs Failure Microscope V4 report.*
