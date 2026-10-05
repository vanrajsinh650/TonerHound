# EXP-036D Before/After Benchmark Comparison

## 1. High-Level Summary
- **Baseline Benchmark**: EXP-028E (370 documents)
- **Experiment Intervention**: EXP-036D Deterministic Visual Provider with Policy A/D (strictly gated on boolean/checkbox schema fields; injected only when baseline has no candidate).
- **Core Technology**: 100% Classical Deterministic Computer Vision (Wireframe morphology, Hough diagonal line detection, core occupancy thresholding; NO LLM, NO VLM, NO neural network).

---

## 2. Official Unified Evidence Metrics

| Metric | Production Baseline (EXP-028E) | EXP-036D Full Benchmark | Net Delta |
|:---|:---:|:---:|:---:|
| **Word Grounding F1** | **56.0477%** | **56.8237%** | **+0.7760 pp** |
| **Page Grounding F1** | **81.6639%** | **81.6955%** | **+0.0316 pp** |
| **Word Grounding Precision** | 61.7275% | 62.1810% | +0.4534 pp |
| **Word Grounding Recall** | 52.5686% | 53.5522% | +0.9835 pp |

---

## 3. Subsystem Field & Document Impact

- **Total Boolean/Checkbox Targets**: 286
- **Fields Rescued**: **185** (64.69%)
- **Fields Regressed**: **0** (0.00%)
- **Net Field Change**: **+185**
- **Documents Improved**: **48** (target docs with net positive Word F1 gain up to +2.50 pp)
- **Documents Regressed**: **22** (slight precision penalty from sub-threshold candidates in dense tabular forms)
- **Documents Neutral**: **1**
- **Unmodified Benchmark Documents**: **299** (0.00 pp delta, 100% bit-for-bit preserved)
- **False-Positive Explosion**: **0** (Policy A preserves existing candidates; Policy D isolates non-boolean fields).

---

## 4. Failure Microscope Classification Migration

| Causal Failure Class | Baseline Count | EXP-036D Count | Net Reduction |
|:---|:---:|:---:|:---:|
| `NON_TEXT_BOOLEAN_GROUNDING` | 286 | 101 | **185 fields resolved** |
| `ALREADY_RESOLVED` | 0 | 185 | **+185 fields** |

---

## 5. Top Rescued Field Examples

| Document ID | Field Path | Gold State | Predicted State | Baseline IoU | EXP-036D IoU |
|:---|:---|:---:|:---:|:---:|:---:|
| `medium/arif-2023` | `form_8879.taxpayer_pin_self_entered_box` | UNCHECKED | UNCHECKED | 0.00 | **0.78** |
| `medium/arif-2023` | `schedule_8812.next_line16b_4500_no_box` | CHECKED | CHECKED | 0.00 | **0.81** |
| `medium/becerra-2021` | `form_8949.part2_box_e` | UNCHECKED | UNCHECKED | 0.00 | **0.60** |
| `medium/becerra-2021` | `form_8949.part2_box_f` | UNCHECKED | UNCHECKED | 0.00 | **0.75** |
| `medium/becerra-2022` | `form_8949.part1_box_b` | UNCHECKED | UNCHECKED | 0.00 | **0.57** |
| `medium/becerra-2022` | `form_8949.part1_box_c` | UNCHECKED | UNCHECKED | 0.00 | **0.67** |
| `medium/becerra-2022` | `form_8949.part2_box_e` | UNCHECKED | UNCHECKED | 0.00 | **0.64** |
| `medium/becerra-2022` | `form_8949.part2_box_f` | UNCHECKED | UNCHECKED | 0.00 | **0.65** |
| `medium/becerra-2023` | `form_8949.part1_box_b` | UNCHECKED | UNCHECKED | 0.00 | **0.60** |
| `medium/becerra-2023` | `form_8949.part1_box_c` | UNCHECKED | UNCHECKED | 0.00 | **0.56** |

---

## 6. Latency and Operational Overhead
- **Detection Time**: 50.4s across 71 target documents (~0.71s/doc).
- **Latency Overhead per Page**: ~530ms only on pages with boolean/checkbox queries.
- **Production Safety**: Zero dependencies on neural networks or external services; pure PyMuPDF + OpenCV.
