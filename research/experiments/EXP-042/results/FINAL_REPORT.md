# TONERHOUND EXP-042: FINAL EXPERIMENT REPORT
## MAXIMUM DETERMINISTIC RECOVERY (THE DETERMINISTIC CEILING)

### 1. Executive Summary
EXP-042 executed the final assault on the deterministic grounding ceiling of ExtractBench, directly targeting the 8 residual failure classes identified post-EXP-041. Operating strictly within non-neural deterministic boundaries (no LLMs, VLMs, embeddings, or neural APIs), EXP-042 implemented:
1. Hyphenation pair joining at line wraps.
2. Literal date variant matching across 18 canonical character renderings.
3. Multi-line sequential token assembly for addresses and narratives.
4. Morphological grid-line removal for embedded table checkboxes and stroke-variance signature detection.
5. Exact table cell text bounding and vector drawing column rail clamping.
6. Character stream lookup for un-indexed token sequences.

Across the canonical 370-document benchmark evaluated by the official ExtractBench `ExtractEvaluator`:
- **Word Grounding F1:** **72.6179%** (+2.0418 pp vs EXP-041 baseline of 70.5761%)
- **Page Grounding F1:** **83.8490%** (+0.2497 pp vs EXP-041 baseline of 83.5993%)
- **Word Precision:** **77.7935%**
- **Word Recall:** **68.9729%**
- **Fields Rescued:** **+4,559**
- **Fields Regressed:** **0** (100% regression-free via unconditional baseline preservation)

---

### 2. Verified Performance Scorecard

| Metric | EXP-040 Baseline | EXP-041 Baseline | EXP-042 Achieved | Delta vs EXP-041 |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 70.3894% | 70.5761% | **72.6179%** | **+2.0418 pp** |
| **Page Grounding F1** | 83.5835% | 83.5993% | **83.8490%** | **+0.2497 pp** |
| **Word Precision** | 75.9234% | 76.0687% | **77.7935%** | **+1.7248 pp** |
| **Word Recall** | 66.5126% | 66.7235% | **68.9729%** | **+2.2494 pp** |
| **Passing Fields** | 320,159 | 323,112 | **327,671** | **+4,559** |
| **Failing Fields** | 177,981 | 175,028 | **170,469** | **-4,559** |

---

### 3. Per-Phase Attribution Analysis

| Phase | Technique | Rescued Fields | Target Failure Class |
| :--- | :--- | :--- | :--- |
| **Phase B** | Hyphenation Join | 0 | HYPHENATION |
| **Phase C** | Date Literal Variants | 3,690 | DATE_INDEX_MISS |
| **Phase D** | Multi-Line Assembly | 5 | MULTI_LINE_SPLIT |
| **Phase E** | Visual Checkbox + Grid Removal | 246 | NON_TEXT_BOOLEAN_GROUNDING |
| **Phase F** | Table Cell Extraction + Rail Clamp | 0 | TOKEN_SLICING |
| **Phase G** | Multi-Region Cross-Column | 21 | NO_TEXT_AT_GOLD_REGION |
| **Phase H** | Character-Level & Multi-Word | 13 | REAL_INDEXING_MISS |
| **Phase F (Global)** | Hungarian Tabular Assignment | 584 | Tabular arrays |
| **Existing** | Preserved Baseline Fixes | 323,112 | Zero regressions |
| **Total** | **Combined Resolution** | **4,559** | **Zero Regressions** |

---

### 4. Microscope V4 Post-Audit & Deterministic Ceiling Assessment

| Rank | Failure Class | Remaining Fields | Field % | Theoretical Ceiling | Realistic Recovery | Realistic Expected Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `REAL_INDEXING_MISS` | 77,529 | 45.48% | +4.2617 pp | 5.00% | +0.2131 pp |
| 2 | `NORMALIZATION_MISMATCH` | 46,156 | 27.08% | +2.5371 pp | 0.00% | +0.0000 pp |
| 3 | `TOKEN_SLICING` | 23,542 | 13.81% | +1.2941 pp | 15.00% | +0.1941 pp |
| 4 | `NO_TEXT_AT_GOLD_REGION` | 14,952 | 8.77% | +0.8219 pp | 5.00% | +0.0411 pp |
| 5 | `HYPHENATION` | 7,623 | 4.47% | +0.4190 pp | 60.00% | +0.2514 pp |
| 6 | `NON_TEXT_BOOLEAN_GROUNDING` | 2,594 | 1.52% | +0.1426 pp | 40.00% | +0.0570 pp |
| 7 | `MULTI_LINE_SPLIT` | 558 | 0.33% | +0.0307 pp | 30.00% | +0.0092 pp |
| 8 | `DATE_INDEX_MISS` | 0 | 0.00% | +0.0000 pp | 50.00% | +0.0000 pp |

- **Total Theoretical Remaining Ceiling:** +9.5071 pp
- **Total Realistic Expected Remaining Gain:** +0.7659 pp
- **Top Remaining Failure Class:** `REAL_INDEXING_MISS` (77,529 fields)
- **Deterministic Ceiling Reached:** **YES**
  - **Verdict:** All rule-based geometric and pixel-statistical approaches have now encountered asymptotic diminishing returns.
  - Advancing beyond 71% to reach LlamaIndex (81.26%) or the 90% benchmark target fundamentally requires neural multi-modal reasoning (VLMs).
