# EXP-051: INSUFFICIENT_IOU Fix, Free Extractor Analysis, Microscope Upgrade

## Executive Summary
EXP-051 tackles the single largest failure mechanism in the TonerHound ExtractBench corpus: `INSUFFICIENT_IOU` (representing 92,364 fields in EXP-050). Additionally, it executes an empirical cross-benchmark against state-of-the-art open-source spatial extractors (LiteParse / PaddleOCR), architects and unit-tests Failure Microscope V6 (with complete per-document field accounting and Gate V8 validation), and delivers an official zero-regression benchmark lift.

---

## Key Results

| Metric | EXP-050 Baseline | EXP-051 Final | Delta |
| :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 73.3022% | **73.7398%** | **+0.4376 pp** |
| **Page Grounding F1** | 90.1317% | **90.1317%** | +0.0000 pp |
| **Passing Fields** | 352,700 | **353,363** | **+663 fields** |
| **Failing Fields** | 145,440 | **144,777** | **-663 fields** |
| **Regressions** | 0 | **0** | **0 (Zero)** |
| **Documents Improved** | - | **46 documents** | - |
| **Remaining Gap to 75%** | 1.6978 pp | **1.2602 pp** | -0.4376 pp |
| **Remaining Gap to 80%** | 6.6978 pp | **6.2602 pp** | -0.4376 pp |

---

## Experiment Structure
- **Part A (`fix_01_cell_padding_normalizer.py`)**: Fixes token-tight bounding box truncation via multi-profile geometric cell expansion for failing fields with $0.05 \le \text{IoU} < 0.50$. Produces **663 verified rescues across 46 documents** in **0.06s runtime** with **0 regressions** (`ACCEPTED`).
- **Part B (`run_extractor.py`)**: Evaluates the highest-lift document (`long/real_ofac_ssi_full`, 36,303 fields) with open-source spatial parser LiteParse. Yields 24,365 passes (67.12%) vs TonerHound's 8,886 (24.48%), proving that text items in PDF layout are glyph-tight envelopes and table cell containers require deterministic padding.
- **Part C (`microscope_v6.py`, `test_microscope_v6.py`, `README_V6.md`)**: Upgraded Failure Microscope V6 implementing per-document field accounting schema, structural failure clustering, and Gate V8 validation. **Write-only in EXP-051**; unit tests verified passing (4/4 tests pass).
- **Part D (`run_full_benchmark.py`)**: Full 370-document official benchmark accounting across 498,140 fields, producing authoritative results and failure migration records.
