# EXP-005 Part 4 Pass 4: Failure Correlation & Attribution Report

**Document**: `long/real_ftx_full_corrupted` (114 pages, 7,554 creditors, 75,543 test rules)  
**Experiment**: EXP-005 Part 4 Pass 4 — Failure Correlation & Attribution Engine  
**Agent**: Agent C (Failure Correlation Lead)  
**Date**: 2026-09-16  
**Baseline State**:  
- Word Grounding F1: **49.87%** (Precision: 49.87%, Recall: 49.87%)  
- Page Grounding F1: **100.00%**  
- Passing Gradeable Citations: **13,257**  
- Failing Citations: **13,326** (Pass 2 baseline before Pass 3 column calibration had 13,762 failing citations)  
- Unit Tests: **95/95 passing in 3.27s**  

---

## 1. Executive Summary

As Agent C for EXP-005 Part 4 Pass 4, the primary objective is to establish an exact, deterministic **Failure Correlation Engine** on `long/real_ftx_full_corrupted`. 

Prior iterations made massive leaps:
- **Baseline (EXP-005 Part 3)**: 0.45% Word F1
- **Pass 1 (Slot Budgeting)**: 22.94% Word F1
- **Pass 2 (Dynamic Width Scaling & Tilt Compensation)**: 48.23% Word F1 (12,821 passing, 13,762 failing)
- **Pass 3 (Empirical Column Coordinate Calibration)**: 49.87% Word F1 (13,257 passing, 13,326 failing)

With concurrent work underway by **Agent A** (targeting **G1: Row Slot Drift**) and **Agent B** (targeting **G3: Multi-Line Line Choice**), Agent C delivers:
1. An exhaustive mapping of all failing citations to the standardized error taxonomy (`G1`, `G2`, `G3`, `G4`, `T1`).
2. Exact category counts and percentage breakdowns across the document and per-field.
3. A frozen baseline failure registry (`experiments/EXP-005-ftx-failure-registry.json`) encompassing all 26,583 gradeable citations.
4. An automated **Correlation Framework** (`src/tonerhound/benchmark/correlation.py` and `scripts/verify_proposal_correlation.py`) to verify that fixes proposed by Agent A and Agent B genuinely resolve their intended target categories with high purity (>= 60%) and zero regressions.

---

## 2. Document Rule Universe Accounting

| Stratum | Description | Count | % of All 75,543 Rules |
| :--- | :--- | :---: | :---: |
| **Total Rules in Document Schema** | All extraction leaf fields defined in schema | **75,543** | **100.00%** |
| **Unpopulated Optional Subfields** | Null / empty address lines (`address_2` through `address_4`) without ground truth bboxes | **48,960** | **64.81%** |
| **Gradeable Citations with Evidence BBox** | Official evaluation denominator for Word Grounding Recall & Precision | **26,583** | **35.19%** |
| **Passing Gradeable Citations (IoU $\ge$ 0.50)** | Baseline passing citations on current `HEAD` (Pass 3) | **13,257** | **49.87%** of gradeable |
| **Failing Citations (IoU < 0.50)** | Baseline failing citations requiring attribution & remediation | **13,326** | **50.13%** of gradeable |

*(Note: Prior to Pass 3 column calibration, Pass 2 had 12,821 passing and exactly 13,762 failing citations. Pass 3 resolved 436 near-miss horizontal citations, lowering the failing citation pool to 13,326).*

---

## 3. Error Taxonomy Definition & Parameter Boundaries

Every failing citation is mapped to one of 5 mutually exclusive, deterministic failure categories:

| Taxonomy Code | Name | Geometric Parameter Boundaries | Physical Mechanism / Root Cause |
| :---: | :--- | :--- | :--- |
| **G1** | **Row Slot Drift** | Anchor $\ge 1$ slot off, $\|dy\| > 0.006$ (or page mismatch) | Cumulative slot budgeting drift across multi-line wrapped rows; line extrapolation misalignment |
| **G2** | **Cell Dimension Mismatch** | Correct row ($\|dy\| \le 0.006$), horizontal overlap intact, $0.30 \le \text{IoU} < 0.50$ | Rigid or approximate column bounding box width/height ratio clipping or overextending text |
| **G3** | **Multi-Line Line Choice** | $\|dy\| \approx 0.010 - 0.013$ ($0.0095 \le \|dy\| \le 0.0135$) within 2-slot cells | Target subfield printed on line 2 vs line 1 of a multi-line address cell; single vs double-slot height confusion |
| **G4** | **Column Boundary Offset** | Horizontal displacement $\|dx\| > 0.01$ (or horizontal IoU $i_x < 0.40$) on correct row | Scanner page tilt or subfield horizontal column gutter misalignment |
| **T1** | **Severe OCR Glyph Corruption** | Correct row/column position but $\text{IoU} < 0.30$, or missing candidate prediction | Tabular grid lines fragment characters into broken glyphs (`[RAMCONIRE`, `SICON`), causing lexical matching collapse |

---

## 4. Baseline Failure Breakdown & Distribution

### Exact Counts and Percentages on Baseline Failing Citations (13,326 Citations):

| Category | Description | Count | % of Failing Citations | % of Gradeable Citations (26,583) | Recoverable by Geometry? |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **G2** | **Cell Dimension Mismatch** | **6,056** | **45.44%** | **22.78%** | **YES** |
| **G1** | **Row Slot Drift** | **4,682** | **35.13%** | **17.61%** | **YES** |
| **T1** | **Severe OCR Glyph Corruption** | **2,023** | **15.18%** | **7.61%** | **NO** (Requires OCR Repair) |
| **G3** | **Multi-Line Line Choice** | **564** | **4.23%** | **2.12%** | **YES** |
| **G4** | **Column Boundary Offset** | **1** | **0.01%** | **0.00%** | **YES** |
| **TOTAL** | **All Failing Citations** | **13,326** | **100.00%** | **50.13%** | — |

### Key Observations:
1. **Geometry Dominates the Error Pool**: Categories `G1 + G2 + G3 + G4` account for **11,303 citations** (**84.82% of all remaining errors**).
2. **Pure-Geometry Ceiling**: Resolving all geometric errors would yield **92.39% Word Grounding F1** (24,560 / 26,583) without touching the OCR engine.
3. **Irreducible OCR Residual**: Only **2,023 citations (7.61% of gradeable citations)** are fundamentally blocked by broken OCR characters (`T1`).
4. **Pass 3 Column Calibration Impact**: Pass 3 calibrated empirical column positions, virtually eliminating pure column horizontal offset `G4` (reducing it from 1,090 to 1 citation), shifting those citations into either `CORRECT` (+436 passes) or `G2` dimension tuning.

---

## 5. Per-Field Error Decomposition

Breakdown of the 13,326 failing citations across individual schema fields:

| Field Name | Total Gradeable | Passes (Pass 3) | Pass % | G1 (Slot Drift) | G2 (Dim Mismatch) | G3 (Line Choice) | G4 (Col Offset) | T1 (OCR Corrupt) | Total Fails |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `name` | 7,554 | 4,145 | 54.87% | 1,289 | 1,575 | 99 | 0 | 446 | **3,409** |
| `address_1` | 6,982 | 4,047 | 57.96% | 1,116 | 1,391 | 98 | 0 | 330 | **2,935** |
| `city` | 3,315 | 1,510 | 45.55% | 626 | 794 | 81 | 1 | 303 | **1,805** |
| `state` | 2,576 | 798 | 30.98% | 503 | 829 | 63 | 0 | 383 | **1,778** |
| `postal_code` | 3,022 | 1,246 | 41.23% | 598 | 783 | 75 | 0 | 320 | **1,776** |
| `address_2` | 1,684 | 810 | 48.10% | 268 | 381 | 94 | 0 | 131 | **874** |
| `country` | 871 | 415 | 47.65% | 178 | 183 | 22 | 0 | 73 | **456** |
| `address_3` | 481 | 245 | 50.94% | 87 | 91 | 27 | 0 | 31 | **236** |
| `address_4` | 95 | 39 | 41.05% | 17 | 29 | 5 | 0 | 5 | **56** |
| `report_title` | 1 | 0 | 0.00% | 0 | 0 | 0 | 0 | 1 | **1** |
| `case_number` | 1 | 1 | 100.00% | 0 | 0 | 0 | 0 | 0 | **0** |
| `debtor` | 1 | 1 | 100.00% | 0 | 0 | 0 | 0 | 0 | **0** |
| **TOTAL** | **26,583** | **13,257** | **49.87%** | **4,682** | **6,056** | **564** | **1** | **2,023** | **13,326** |

---

## 6. Correlation Framework & Verification Protocol

To guarantee that algorithmic proposals from **Agent A** (G1 focus) and **Agent B** (G3 focus) actually repair the targeted root cause rather than producing accidental collateral gains or hidden regressions, Agent C established the following verification framework:

### Metrics Measured:
1. **Newly Passing Citations ($\Delta^+$)**: Citations that failed in baseline but achieve $\text{IoU} \ge 0.50$ in the candidate.
2. **Regressions ($\Delta^-$)**: Citations that passed in baseline ($\text{IoU} \ge 0.50$) but now fail in the candidate.
3. **Net Gain**: $\Delta_{\text{net}} = \Delta^+ - \Delta^-$.
4. **Attribution Purity**:
   $$\text{Purity}(\text{Agent}, \text{Target}) = \frac{\text{Newly Fixed Citations belonging to Target}}{\text{Total Newly Fixed Citations}} \times 100\%$$
   - **Verification Threshold**: $\text{Purity} \ge 60.00\%$ (minimum requirement to attribute the gain to the stated hypothesis).
5. **Target Category Recall**:
   $$\text{Recall}(\text{Target}) = \frac{\text{Newly Fixed Citations belonging to Target}}{\text{Total Baseline Target Pool}} \times 100\%$$
6. **Zero-Regression Invariant**: $\Delta^- == 0$. Any regressions on previously passing citations trigger an immediate verification rejection.

---

## 7. How to Use the Verification Harness

### CLI Interface:
```bash
# Verify the current adapter / working tree against Agent A's G1 target:
.venv/bin/python scripts/verify_proposal_correlation.py --target G1

# Verify the current adapter / working tree against Agent B's G3 target:
.venv/bin/python scripts/verify_proposal_correlation.py --target G3

# Verify an exported candidate citation JSON payload:
.venv/bin/python scripts/verify_proposal_correlation.py --target G1 --citations path/to/candidate_citations.json
```

### Python API:
```python
from tonerhound.benchmark.correlation import FailureRegistry, verify_agent_proposal

# Load baseline registry (instant, < 0.05s)
registry = FailureRegistry.load()

# Correlate candidate citations
report = registry.correlate(candidate_citations, target_category="G1", min_purity=0.60)
print(report.summary())

# Check verification status
if report.is_verified:
    print(f"Verified on-target! Purity: {report.purity*100:.2f}%, Net gain: +{report.net_gain}")
else:
    print(f"Verification failed: {report.verification_notes}")
```

### Automated Unit Test Suite:
```bash
.venv/bin/pytest tests/test_failure_correlation.py -v
```
All 6 correlation unit tests pass in 2.92s, verifying registry loading, self-correlation zero deltas, Agent A G1 simulation, Agent B G3 simulation, off-target purity rejection, and regression detection.
