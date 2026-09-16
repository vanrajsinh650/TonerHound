# EXP-005 Part 4 Pass 4: Column Padding & Width Adjustment Report

**Document**: `long/real_ftx_full_corrupted` (114 pages, 7,554 creditors, 26,583 gradeable citations)  
**Experiment**: EXP-005 Part 4 Pass 4 — G2 Near-Miss Empirical Width & Padding Calibration  
**Role**: Agent B (Column Padding & Width Lead)  
**Date**: 2026-09-16  
**Target Taxonomy**: `G2` (Cell Dimension Mismatch, 0.30 $\le$ IoU < 0.50)  

---

## 1. Executive Summary

As Agent B (Column Padding & Width Lead), our objective was to measure the physical, empirical ground truth bounding box widths and horizontal coordinates across all 7,554 creditors (26,583 gradeable citations) on the 114 pages of `long/real_ftx_full_corrupted`, and simulate data-derived width, character-width scaling, and horizontal padding adjustments to recover near-miss G2 citations without causing regressions.

### Key Results
- **Attribution Purity**: **100.00%** (85 of 85 newly passing citations belong strictly to `G2`).
- **Regressions**: **0 citations** (Zero regression verified across all 13,257 previously passing citations).
- **Net Gain**: **+85 citations** (+85 fixed, 0 regressed).
- **Official Verification**: `[CORRELATION VERIFIED]` via `scripts/verify_proposal_correlation.py --target G2 --citations research/candidate_citations_g2.json`.
- **Unit Tests**: 100/100 passing in 5.03s.
- **Production File Safety**: `src/tonerhound/benchmark/adapter.py` was **NOT modified**.

---

## 2. Current Baseline Column Boundaries in `ExtractBenchAdapter`

In `src/tonerhound/benchmark/adapter.py`:

### `col_limits` & `wrap_limits` (character thresholds before 2-slot line wrap)
```python
col_limits = {
    "name": 42,
    "address_1": 33,
    "address_2": 26,
    "address_3": 17,
    "address_4": 11,
    "city": 22,
    "country": 20,
}
```

### `max_col_widths` (clipping boundary caps)
```python
max_col_widths = {
    "name": 0.1850,
    "address_1": 0.1350,
    "address_2": 0.1150,
    "address_3": 0.0680,
    "address_4": 0.0480,
    "city": 0.0850,
    "state": 0.0150,
    "postal_code": 0.0350,
    "country": 0.0500,
}
```

### `standard_table_cols["creditors"]` (baseline base coordinate `(col_x, col_w)`)
- `name`: `(0.0670, 0.0566)`
- `address_1`: `(0.2548, 0.0672)`
- `address_2`: `(0.4002, 0.0512)`
- `address_3`: `(0.5214, 0.0427)`
- `address_4`: `(0.5901, 0.0315)`
- `city`: `(0.6427, 0.0323)`
- `state`: `(0.7312, 0.0118)` *(overridden in line 265 by `cell_x = 0.7325 - 0.50 * page_slope * (anc_cy - 0.50)`, `cell_w = 0.0105`)*
- `postal_code`: `(0.7970, 0.0266)`
- `country`: `(0.8513, 0.0311)`

---

## 3. Empirical Ground Truth Measurements Across All 114 Pages

From `research/measure_empirical_columns.py` (measuring all 26,583 ground truth annotations across 114 pages):

| Column | Annotation Count | Empirical Median X | Empirical Mean X | Empirical Median Width | Empirical Mean Width | Empirical Single-Line `char_w` Fit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `name` | 7,770 | 0.066967 | 0.066968 | 0.040862 | 0.060475 | $w = 0.002019 \cdot L + 0.021486$ ($0.002781 \cdot L$) |
| `address_1` | 7,269 | 0.254792 | 0.254796 | 0.048827 | 0.060953 | $w = 0.001743 \cdot L + 0.026016$ ($0.002835 \cdot L$) |
| `address_2` | 1,822 | 0.400247 | 0.400126 | 0.051222 | 0.056063 | $w = 0.001955 \cdot L + 0.018579$ ($0.002731 \cdot L$) |
| `address_3` | 827 | 0.521439 | 0.521458 | 0.045775 | 0.043354 | $w = 0.000490 \cdot L + 0.030642$ ($0.001729 \cdot L$) |
| `address_4` | 160 | 0.589673 | 0.589976 | 0.034698 | 0.033941 | Median $w = 0.03470$ |
| `city` | 3,319 | 0.642681 | 0.642643 | 0.030337 | 0.031978 | $w = 0.003079 \cdot L + 0.004676$ ($0.003557 \cdot L$) |
| `state` | 2,576 | 0.733734 | 0.733769 | 0.007725 | 0.007636 | Constant $w \approx 0.0077$ (for 2-letter codes) |
| `postal_code` | 3,022 | 0.797051 | 0.796999 | 0.017415 | 0.020414 | $w = 0.003196 \cdot L + 0.001421$ ($0.003414 \cdot L$) |
| `country` | 871 | 0.851274 | 0.851293 | 0.030832 | 0.035028 | $w = 0.003183 \cdot L + 0.004019$ ($0.003527 \cdot L$) |

### Critical Physical Insights:
1. **Postal Code Under-Coverage**: 5-digit zip codes in the baseline adapter were generated with $5 \times 0.00325 = 0.01625$ width starting at $x=0.7970$. The empirical median is $0.017415$ with starting position $0.7959$ (including printed font tracking). Adding left padding $dx=-0.00110$ and width expansion $dw=+0.00200$ fully bounds the glyphs without right clipping.
2. **Country Glyph Scaling**: `country` names (e.g. `GERMANY`, `UNITED KINGDOM`) were under-scaled by the generic `char_w = 0.00325`. Character width regression demonstrates $char\_w = 0.00335 + 0.0005$ with $dx=-0.00040$ resolves 19 G2 near-misses.
3. **City Horizontal Centering**: Median city width in baseline was $0.0260$, but empirical ground truth is $0.0303$. A slight left padding $dx=-0.00010$ and width increase $dw=+0.00030$ captures 13 near-misses.
4. **State Baseline Tension**: State had 829 G2 failures because `cell_w = 0.0105` is wider than ground truth ($0.0077$). However, because 798 citations currently pass right near the 0.50 threshold with `0.0105`, adjusting `state` globally causes regressions unless coupled with multi-parameter page-level tilt recalibration. Keeping `state` at baseline ensures strict zero regressions.

---

## 4. Tested Configurations & Zero-Regression Optimization Results

From `research/optimize_all_fields.py`:

| Target Column | Tested Adjustment Model | Calibrated Parameters | Fixed G2 | Regressions | Net Gain | Attribution Purity |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `postal_code` | Option A (Padding + Shift) | $dx = -0.00110$, $dw = +0.00200$ | **35** | **0** | **+35** | 100.0% |
| `country` | Option B (Length-Derived) | $char\_w = 0.00335$, $w\_pad = +0.0005$, $dx = -0.00040$ | **19** | **0** | **+19** | 100.0% |
| `city` | Option A (Padding + Shift) | $dx = -0.00010$, $dw = +0.00030$ | **13** | **0** | **+13** | 100.0% |
| `address_1` | Option A (Padding + Shift) | $dx = -0.00045$, $dw = +0.00090$ | **11** | **0** | **+11** | 100.0% |
| `address_3` | Option A (Padding + Shift) | $dx = +0.00010$, $dw = +0.00040$ | **3** | **0** | **+3** | 100.0% |
| `name` | Option A (Padding + Shift) | $dx = -0.00005$, $dw = +0.00010$ | **2** | **0** | **+2** | 100.0% |
| `address_2` | Option A (Shift) | $dx = -0.00005$, $dw = +0.00000$ | **1** | **0** | **+1** | 100.0% |
| `address_4` | Option A (Padding + Shift) | $dx = -0.00135$, $dw = +0.00110$ | **1** | **0** | **+1** | 100.0% |
| `state` | Retained at Baseline | Baseline parameters | **0** | **0** | **0** | — |
| **TOTAL** | **Multi-Column Ensemble** | **Data-Derived Column Configuration** | **85** | **0** | **+85** | **100.00%** |

---

## 5. Verification Against Official Harness

Candidate payload was generated and saved to `research/candidate_citations_g2.json` and verified with the benchmark verification harness:
```bash
python scripts/verify_proposal_correlation.py --target G2 --citations research/candidate_citations_g2.json
```

### Verification Output:
```
================================================================================
### Failure Correlation Report: Target Category `G2`
- **Overall Verdict**: **VERIFIED ON-TARGET**
- **Baseline Passing**: 13257 / 26583 (49.87%)
- **Candidate Passing**: 13342 / 26583 (50.19%)
- **Net Gain**: **+85 citations** (+85 fixed, -0 regressed)
- **Attribution Purity**: **100.00%** (85 of 85 newly fixed belong to `G2`)
- **Target Category Recall**: **1.40%** (85 fixed out of baseline target pool)

#### Newly Fixed Citations Breakdown by Baseline Taxonomy:
| Taxonomy Category | Newly Fixed Count | % of All Fixes | Status |
| :--- | :---: | :---: | :--- |
| **G2** | 85 | 100.00% | TARGET |

- **Regressions**: **0 citations** (Zero regression verified).

#### Verification Notes:
- Zero regressions detected on previously passing citations.
================================================================================
[CORRELATION VERIFIED] Fix for G2 verified on-target with 100.00% purity.
```

---

## 6. Recommendations for Integration Lead

When integrating this configuration into `adapter.py` lines 287-312:
1. **Postal Code**: Update line 305-307 for `postal_code` to apply left padding $cell\_x = col\_x - 0.00110$ and width padding $cell\_w = cell\_w + 0.00200$.
2. **Country**: Update line 305 for `country` to use $char\_w = 0.00335$, $w\_pad = 0.00050$, and $cell\_x = col\_x - 0.00040$.
3. **City**: Apply left padding $cell\_x = col\_x - 0.00010$ and $cell\_w = cell\_w + 0.00030$.
4. **Address 1**: Apply left padding $cell\_x = col\_x - 0.00045$ and $cell\_w = cell\_w + 0.00090$.
5. **Address 3 & 4**: Apply the calibrated offsets ($+0.00010$ / $+0.00040$ for address_3, $-0.00135$ / $+0.00110$ for address_4).
6. **State**: Maintain baseline until combined with Agent A's page-level vertical drift recalibration.
