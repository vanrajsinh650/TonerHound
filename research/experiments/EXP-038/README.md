# EXP-038: Multi-Problem Deterministic Resolution & Failure Microscope V4 Calibration

## Overview

EXP-038 is the multi-problem deterministic research cycle executed end-to-end to address the failure populations mapped by Failure Microscope V4.

It deployed 8 deterministic resolution modules under strict zero-regression constraints, validated performance on a 32-document held-out cohort, evaluated across the full official ExtractBench 370-document benchmark, and confirmed calibration through the 8-gate validation suite.

---

## Key Results

- **Official Word Grounding F1:** **58.1118%**
  - **+1.4411 pp** vs `canonical_baseline_v2` (56.6707%)
  - **+2.0641 pp** vs `canonical_370_v1` (56.0477%)
- **Official Page Grounding F1:** **82.2750%**
  - **+0.2964 pp** vs `canonical_baseline_v2` (81.9786%)
  - **+0.6111 pp** vs `canonical_370_v1` (81.6639%)
- **Net Fields Rescued:** **+1,242 fields** ($IoU < 0.50 \rightarrow IoU \ge 0.50$) across 108 documents
- **Fields Regressed:** **0 fields** ($IoU \ge 0.50 \rightarrow IoU < 0.50$)
- **Evaluator:** Official ExtractBench `ExtractEvaluator` (`ExtractAssociationF1Metric`)
- **Microscope V4 Backtesting:** PASSED across EXP-035, EXP-036D, EXP-037, and EXP-038 (1.92x actual/expected ratio within 2x requirement)
- **Integration Verdict:** **SUPPORTED FOR PRODUCTION INTEGRATION**

---

## Directory Contents

| File / Folder | Purpose |
| :--- | :--- |
| `FINAL_REPORT.md` | Comprehensive master report answering all 11 directive questions |
| `before_after_comparison.md` | Detailed before/after metric comparison and top improved documents |
| `decision.md` | Formal evidence-based decision and production integration plan |
| `implementation_notes.md` | Architectural design, module mechanics, and performance characteristics |
| `run_targeted.py` | Phase E 32-document held-out evaluation runner |
| `run_full_benchmark.py` | Phase F full 370-document benchmark runner |
| `unified_harness.py` | Multi-problem deterministic resolver priority cascade |
| `normalization_v2.py` | Parenthesized negative and currency normalization |
| `ocr_noise_index.py` | Secondary character 3-gram index for OCR pages |
| `visual_provider_v2.py` | Morphological checkbox wireframe detector with caching |
| `date_normalizer.py` | Regex date normalizer for non-standard formats |
| `multiline_assembler.py` | Bounded multi-line bounding box union assembler |
| `token_slicer.py` | Proportional character-span sub-box slicer |
| `hyphenation_joiner.py` | Terminal line hyphen joiner |
| `unresolved_ocr_retry.py` | High-DPI adaptive CLAHE/Otsu OCR retry |
| `full_benchmark_results.json` | Complete benchmark metrics and per-document table |
| `per_document.csv` | CSV summary of all 370 documents |
| `fix_attribution.json` | Exact rescue count per fix mechanism |
| `failure_migration.json` | Migration matrix from baseline class to post-run class |
| `regression_analysis.json` | Zero-regression proof audit |
| `success_analysis.json` | Sample rescued field records with before/after IoU |
| `results/` | Phase E held-out results and intermediate predictions |
| `predictions/tonerhound/` | Post-EXP-038 prediction files for all 370 documents |
| `eval_cache/` | Official ExtractBench evaluation JSONs for all modified documents |
