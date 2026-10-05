# EXP-038 Before vs After Comparison

## Official Benchmark Metrics (370 Documents)

- **Harness:** Official ExtractBench `ExtractEvaluator` (`ExtractAssociationF1Metric` / Unified Evidence Metric)
- **Baseline V1:** `canonical_370_v1` (Git commit `178f81c`)
- **Baseline V2:** `canonical_baseline_v2` (Post-EXP-037, Git commit `178f81c`)
- **EXP-038 After:** `exp038_full_run_v1` (Git commit `178f81c`)

| Metric | Baseline V1 | Baseline V2 | EXP-038 After | Delta vs V2 | Delta vs V1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Word Grounding F1** | 56.0477% | 56.6707% | **58.1118%** | **+1.4411 pp** | **+2.0641 pp** |
| **Page Grounding F1** | 81.6639% | 81.9786% | **82.2750%** | **+0.2964 pp** | **+0.6111 pp** |
| **Word Grounding Precision** | 61.7275% | 62.4511% | **63.9733%** | **+1.5222 pp** | **+2.2458 pp** |
| **Word Grounding Recall** | 52.5686% | 53.1233% | **54.5086%** | **+1.3853 pp** | **+1.9400 pp** |
| **Passing Fields ($IoU \ge 0.50$)** | 307,373 | 307,514 | **308,756** | **+1,242 fields** | **+1,383 fields** |
| **Failing Fields** | 190,767 | 190,626 | **189,384** | **-1,242 fields** | **-1,383 fields** |
| **Total Gradeable Fields** | 498,140 | 498,140 | **498,140** | 0 | 0 |
| **Evaluated Docs (Word F1)** | 236 | 236 | **236** | 0 | 0 |
| **Evaluated Docs (Page F1)** | 293 | 293 | **293** | 0 | 0 |

---

## Document-Level Breakdown

| Category | Document Count | Percentage of Benchmark |
| :--- | :---: | :---: |
| **Improved Documents (Word F1 increased)** | **108** | **29.19%** |
| **Regressed Documents (Word F1 decreased)** | **0** | **0.00%** |
| **Unchanged Documents** | **262** | **70.81%** |
| **Total Documents** | **370** | **100.00%** |

---

## Top 15 Improved Documents in EXP-038

| Document ID | Baseline V2 Word F1 | EXP-038 Word F1 | F1 Delta | Fields Rescued | Primary Rescuing Fix |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `medium/real_vg_divappr_full` | 66.8643% | **94.7570%** | **+27.8927 pp** | +291 | `fix6_normalization` |
| `medium/real_penn_hills_pa_2023` | 13.9130% | **40.1739%** | **+26.2609 pp** | +151 | `standard_resolver` |
| `medium/sched_i__akron_community_foundation_ty2024` | 74.0741% | **92.5926%** | **+18.5185 pp** | +150 | `fix6_normalization` |
| `medium/real_vg_healthcare_full` | 70.3664% | **77.6186%** | **+7.2522 pp** | +83 | `fix6_normalization` |
| `medium/sec_13f_0031_loomis_sayles` | 73.0769% | **79.5604%** | **+6.4835 pp** | +50 | `standard_resolver` |
| `medium/real_vg_gcc_full` | 83.1707% | **89.3902%** | **+6.2195 pp** | +51 | `fix6_normalization` |
| `medium/real_dcd_sumf_full_corrupted` | 24.3455% | **31.9372%** | **+7.5917 pp** | +29 | `standard_resolver` |
| `medium/real_vg_megacap_full` | 79.7980% | **85.4545%** | **+5.6565 pp** | +28 | `fix6_normalization` |
| `medium/sched_i__united_way_worldwide_ty2024` | 78.4722% | **83.7963%** | **+5.3241 pp** | +23 | `standard_resolver` |
| `medium/real_blackrock_muni_bmn` | 82.5203% | **87.0000%** | **+4.4797 pp** | +22 | `standard_resolver` |
| `medium/real_enotes_deviations` | 46.8085% | **52.1277%** | **+5.3192 pp** | +15 | `standard_resolver` |
| `medium/bar-lev-2021` | 64.6739% | **72.8261%** | **+8.1522 pp** | +15 | `fix6_normalization` |
| `medium/bar-lev-2023` | 65.4054% | **72.9730%** | **+7.5676 pp** | +14 | `fix6_normalization` |
| `medium/becerra-2022` | 73.1959% | **79.8969%** | **+6.7010 pp** | +13 | `standard_resolver` |
| `medium/nport__advisors_inner_circle_fund` | 85.3448% | **88.2759%** | **+2.9311 pp** | +34 | `standard_resolver` |

---

## Fix Attribution Summary

```
Total Rescued Fields: 1,242
  - standard_resolver:  723 fields (58.21%)
  - fix6_normalization: 494 fields (39.77%)
  - fix1_ocr_noise:      16 fields (1.29%)
  - fix2_checkbox:        5 fields (0.40%)
  - fix4_dates:           4 fields (0.32%)
```
