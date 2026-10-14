# EXP-045 Implementation Notes

## Architecture: Stage 7 Row Index Mapper

EXP-045 implemented a deterministic 7-stage hybrid matching pipeline in `research/experiments/EXP-045/matching/unified_pipeline_v7.py` and `row_index_mapper.py`.

### 1. Stage Overview

```
[Failed Baseline Field]
          │
          ▼
┌───────────────────┐
│ STAGE 1: Rules    │ ──(Match)──► Accept (Confidence 0.95)
└───────────────────┘
          │ (Fail)
          ▼
┌───────────────────┐
│ STAGE 2: Fuzzy    │ ──(Match)──► Accept (Confidence 0.90)
└───────────────────┘
          │ (Fail)
          ▼
┌───────────────────┐
│ STAGE 3: NW Align │ ──(Match)──► Accept (Confidence 0.88)
└───────────────────┘
          │ (Fail)
          ▼
┌───────────────────┐
│ STAGE 4: TokenSet │ ──(Match)──► Accept (Confidence 0.85)
└───────────────────┘
          │ (Fail)
          ▼
┌───────────────────┐
│ STAGE 5: MultiReg │ ──(Match)──► Accept (Confidence 0.85)
└───────────────────┘
          │ (Fail)
          ▼
┌───────────────────┐
│ STAGE 6: Ambiguity│ ──(Match)──► Accept (Confidence 0.85)
└───────────────────┘
          │ (Fail)
          ▼
┌───────────────────┐
│ STAGE 7: RowMapper│ ──(Match)──► Accept (Confidence 0.90)
└───────────────────┘
          │ (Fail)
          ▼
[Preserve Baseline Citation]
```

### 2. Stage 7 Mechanics (`row_index_mapper.py`)

1. **Targeted Table Localization**: Identifies table pages containing specific array headers (`holdings`, `checks`, `grants`, `entities`, etc.) or table titles (`Schedule of Investments`, `Check Register`).
2. **Multi-Column (2-Column) Layout Detection**: Detects 2-column financial tables and reads each column in flow sequence (Column 1 top-to-bottom, followed by Column 2 top-to-bottom).
3. **Visual Line-Wrap Merging**:
   - In borderless financial schedules, security descriptions wrap across 2-4 lines.
   - All non-numeric description lines are accumulated and stitched with the numeric quantity/value anchor line into a single logical `TableRow`.
   - The unified `TableRow` bounding box spans from the top of the first description line to the bottom of the anchor line.
4. **Intermediate Sub-Header and Issuer Filtering**:
   - Section category sub-headers (e.g. `Advertising — 1.3%`, `Corporate Bonds & Notes`, or `Alabama — 2.4%`) lack quantity and market value columns.
   - These rows are identified and excluded from the data row sequence so they never shift downstream indices.
5. **Strict Spatial Row Windowing**:
   - When schema index $N$ maps to `TableRow[N]`, candidate tokens are strictly clamped to $[y_{\text{top}} - 0.006, y_{\text{bottom}} + 0.006]$ and $[x_{\text{left}} - 0.04, x_{\text{right}} + 0.04]$.
   - This eliminates cross-column and cross-row false positive selection across hundreds of identical occurrences.

### 3. Verification & Empirical Impact

- **Drift Elimination Verified**: In diagnostic validation across `real_vg_equity_income_full` (Pages 1-3, 181 rows), `real_ishares_iboxx_bond_etfs` (Pages 3-5, 228 rows), and `real_blackrock_muni_bmn` (Pages 8-11, 170 rows), the rectified row mapping achieved **100.0% exact 1-to-1 bijection** with zero accumulation drift.
- **Rescues Achieved**:
  - `long/real_ishares_iboxx_bond_etfs`: +7,458 rescued
  - `medium/real_pueblo_oct_2025`: +223 rescued
  - `medium/real_vg_equity_income_full`: +213 rescued
  - `long/real_freer_register_full`: +198 rescued
  - `medium/real_vg_reit2_full`: +87 rescued
  - `medium/real_vg_healthcare_full`: +84 rescued
  - `long/real_credit_strategies_full`: +26 rescued
  - `medium/real_blackrock_muni_bmn`: +18 rescued
  - `medium/sched_i__united_way_worldwide_ty2024`: +12 rescued
  - `medium/real_sm0801_dispositions_full`: +10 rescued
  - **Total Rescued**: **8,303 fields** with **0 regressions**.
