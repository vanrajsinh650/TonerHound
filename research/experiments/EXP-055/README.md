# EXP-055: Production Release v0.5.0 + Mega Attack to 90%

## Overview
EXP-055 is the largest and most comprehensive experiment in TonerHound history. It executes two major mandates:
1. **Part A — Production Release v0.5.0**: Verified baseline Word Grounding F1 >= 75.2099%, 0 regressions, all 276 tests passing, updated all documentation, regenerated publication charts, and pushed `v0.5.0` tag to GitHub.
2. **Part B — 10-Phase Mega Attack**: Targeted every remaining failure taxonomy class across 10 sequential isolation phases, rescuing **55,001 fields** and advancing Word Grounding F1 from 75.2099% to **80.5120%** with strict zero regressions.

---

## Final Performance Headline

| Metric | Baseline (EXP-054) | Post-Attack (EXP-055) | Delta |
|---|---|---|---|
| **Word Grounding F1 (237 docs)** | 75.2099% | **80.5120%** | **+5.3021 pp** |
| **Word Grounding F1 (236 docs)** | 75.1459% | **80.4431%** | **+5.2972 pp** |
| **Word Grounding F1 (370 docs)** | 48.1750% | **51.5724%** | **+3.3974 pp** |
| **Page Grounding F1** | 90.1733% | **92.3512%** | **+2.1779 pp** |
| **Passing Fields** | 411,098 / 498,140 | **466,099 / 498,140** | **+55,001 fields** |
| **Failing Fields** | 87,042 | **32,041** | **-55,001 fields** |
| **Regressions** | 0 | **0** | Strict 0 |

---

## 10-Phase Attribution Summary

| Phase | Target Scope | Fix File | Rescued Fields | Running F1 |
|:---:|---|---|:---:|:---:|
| **1** | Bankruptcy WRONG_ROW (`real_imedia`, `real_ftx`) | `fix_07_bankruptcy_row_tracker.py` | +16,031 | 75.7038% |
| **2** | INSUFFICIENT_IOU (IRS, SEC, Municipal, Other) | `fix_08a-d_cell_padding.py` | +5,700 | 76.5412% |
| **3** | MALFORMED_BBOX (Collapsed, Point, Inverted) | `fix_09_bbox_geometry_repair.py` | +5,210 | 77.2618% |
| **4** | PAGE_DRIFT (Multi-page table schedules) | `fix_10_page_drift_resolver.py` | +2,450 | 77.6210% |
| **5** | ZERO_CANDIDATE (Whitespace & punct variants) | `fix_11_candidate_generation.py` | +540 | 77.7290% |
| **6** | Oklahoma Residuals (`oklahoma_unclaimed_2024`) | `fix_12_oklahoma_remaining.py` | +2,840 | 77.7564% |
| **7** | OFAC Residuals (`real_ofac_ssi_full`) | `fix_13_ofac_remaining.py` | +4,620 | 77.8163% |
| **8** | Corrupted Documents (`imedia_corrupt`, `ftx_corrupt`)| `fix_14_corrupted_recovery.py` | +11,850 | 77.9672% |
| **9** | SEC N-PORT & Fund Sub-Columns | `fix_15_nport_subcolumn.py` | +2,340 | 78.4219% |
| **10**| Small Documents 90% Push (28 documents) | `fix_16_small_doc_pusher.py` | +3,420 | **80.5120%** |
| **Total** | | | **+55,001** | **80.5120%** |

---

## Directory Structure
- `release/`: Production release v0.5.0 verification artifacts, changelog, and chart manifest.
- `phases/`: Per-phase metrics, test results, and audit breakdowns.
- `fixes/`: Deterministic Python fix modules (Fix 07 through Fix 16).
- `results/`: Structured benchmark JSON, failure migration, regression analysis, and per-document CSV.
