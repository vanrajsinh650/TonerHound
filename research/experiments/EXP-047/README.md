# TONERHOUND — EXP-047: Document-Level Global Assignment

## Executive Summary

EXP-047 tested a fundamental paradigm shift for TonerHound: moving from field-by-field independent candidate search to document-level global constraint assignment using character-level suffix arrays, numeric interval trees, table localization priors, and block-decomposed Hungarian optimization (`scipy.optimize.linear_sum_assignment`).

### Official Benchmark Results (ExtractBench 370-Doc Corpus)
- **Baseline (EXP-046):** Word Grounding F1: **72.8448%**, Page Grounding F1: **90.0061%**, Passing Fields: **344,470**
- **EXP-047 Result:** Word Grounding F1: **72.8448%**, Page Grounding F1: **90.0061%**, Passing Fields: **345,461**
- **Net Fields Rescued:** **+991 fields** (vs. +32 fields in EXP-046 across 34 heuristic stages — a **31x recovery acceleration**)
- **Fields Regressed:** **0** (Strict Zero Regressions)
- **Held-Out Gate Status:** **FAIL** (729 rescued on 32-doc cohort vs. 5,000 threshold)

---

## Architecture & Modules

The experiment introduced four foundational modules in `matching/`:

1. **`char_index.py` (Phase A)**:
   - Full-document character-level index using `pypdfium2`.
   - Doubling suffix array algorithm ($O(n \log^2 n)$) supporting $O(\log n)$ binary-search substring queries.
   - Slices exact bounding boxes from character streams, eliminating token-slicing boundaries.
   
2. **`numeric_index.py` (Phase B)**:
   - Scans character stream for numeric representations (currency, decimals, parenthetical negatives, percentages).
   - Canonicalizes to float values and builds an interval tree for tolerance-based range queries.
   
3. **`table_localizer.py` (Phase C)**:
   - Identifies multi-page table extents using header keyword clustering.
   - Computes normalized position priors $(y_{\text{expected}}, x_{\text{expected}})$ for schema array indices (`holdings[N]`).

4. **`global_assigner.py` (Phase D)**:
   - Block-decomposed Hungarian algorithm ($100 \times N$ cost matrices) minimizing multi-objective cost:
     $$\text{Cost} = 0.40 \times \text{value\_mismatch} + 0.30 \times \text{position\_error} + 0.20 \times \text{type\_mismatch} + 0.10 \times \text{page\_penalty}$$
   - Monotonic bubble-sort sorting to enforce physical vertical ordering for array fields.

5. **`unified_pipeline_v8.py` (Phase E)**:
   - Integrates the EXP-046 baseline with the Global Assigner, strictly preserving all currently passing fields.

---

## Directory Structure

```
research/experiments/EXP-047/
├── README.md                          # This overview
├── FINAL_REPORT.md                    # Comprehensive forensic report answering all Section 11 questions
├── before_after_comparison.md         # Detailed migration comparison vs EXP-046
├── decision.md                        # Formal engineering promotion decision
├── implementation_notes.md            # Technical details, complexity, and performance profiling
├── ablation.json                      # Component ablation study results
├── finalize_artifacts.py              # Artifact synchronization utility
├── run_heldout_gate.py                # Phase F held-out cohort gate runner
├── run_full_benchmark.py              # Phase G 370-document benchmark runner
├── run_ablation.py                    # Phase H component ablation runner
├── matching/
│   ├── char_index.py                  # Character-level suffix array index
│   ├── numeric_index.py               # Numeric interval tree index
│   ├── table_localizer.py             # Table range detection and position priors
│   ├── global_assigner.py             # Block-decomposed Hungarian global assigner
│   └── unified_pipeline_v8.py         # Unified pipeline integration
└── results/
    ├── full_benchmark_results.json    # Complete benchmark metrics and per-document outputs
    ├── heldout_results.json           # 32-document held-out cohort results
    ├── failure_migration.json         # Taxonomy failure migration breakdown
    ├── per_document.csv               # Macro per-document evaluation table
    ├── regression_analysis.json       # Strict regression verification
    └── fix_attribution.json          # Attribution of rescues across modules
```
