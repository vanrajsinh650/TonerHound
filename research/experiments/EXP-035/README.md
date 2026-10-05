# EXP-035: INDEXING_MISS Causal Audit & Recovery

## 1. Hypothesis

The EXP-034R research handover asserted that `INDEXING_MISS` was the single largest candidate-generation failure class across the ExtractBench benchmark, affecting **571 fields across 136 documents** and offering a macro-weighted headroom of **+3.4991 pp Word Grounding F1**.

The hypothesis tested in EXP-035 was:
> **Are the 571 alleged `INDEXING_MISS` cases genuinely caused by failures in document indexing and token retrieval, and does implementing missing token/n-gram retrieval mechanisms unlock +3.5 pp in macro-weighted Word Grounding F1?**

---

## 2. Prior Assumption

The previous EXP-034R handover reported the following top failure classes:
- `INDEXING_MISS`: 571 fields / 136 documents / +3.4991 pp macro F1 (24.70% of gap)
- `OTHER`: 561 fields / 99 documents / +1.9231 pp
- `BBOX_TOO_NARROW`: 1,454 fields / 112 documents / +1.5683 pp
- `PAGE_PRUNING`: 2,964 fields / 60 documents / +1.4728 pp
- `WRONG_ROW`: 24,138 fields / 82 documents / +1.3384 pp

The handover recommended implementing a multi-character n-gram index (length 2–8) and extensive candidate expansion under the assumption that the 571 fields were unreachable by existing text matching.

---

## 3. Repository Verification

Before modifying production code, a repository audit of the current codebase (`src/tonerhound/`) was conducted.

The current production repository already contains substantial, mature indexing and recovery machinery:
1. **Sublinear Inverted Token Index (`DocumentIndex._token_index`):** O(1) lookup on normalized Unicode/case-folded tokens.
2. **Punctuation-Stripped Stem Index (`DocumentIndex._stem_token_index`):** Strips punctuation, quotes, brackets, and symbols to resolve attached punctuation.
3. **Canonical Numeric Index (`DocumentIndex._numeric_index`):** Indexes floating-point representations of currency, decimals, and percentages, including adjacent currency token combinations (e.g. `$` + `1,200`).
4. **Windowed Date Index (`DocumentIndex._date_index`):** Parses ISO dates across 1- to 5-token visual line windows.
5. **3-Character N-Gram Inverted Index (`DocumentIndex._ngram_to_lines`):** Sublinear line lookup for tokens with length >= 4.
6. **Normalized Page-Wide Substring Search (`DocumentIndex.search_exact`):** Full-page character-to-token offset mapping with word-boundary preservation.
7. **Modular Candidate Recovery Engine (`CandidateRecoveryEngine`):** Recovers spaced-token numbers, split currency/percentage glyphs, interleaved table tokens, and bounded multi-token spans.
8. **Token-Gated Global Fallback (`ENABLE_GLOBAL_FALLBACK`):** Allows searching beyond page hints for unique, high-cardinality tokens.

Git history confirms that these mechanisms were progressively introduced across EXP-013, EXP-028B1, EXP-028D, and EXP-028F.

---

## 4. Failure Audit

All alleged cases from EXP-034R were extracted and audited individually against the live production pipeline.

- **Total Alleged Cases in Benchmark:** **572 fields** across **137 unique documents**.
- **Genuine `REAL_INDEXING_MISS` Cases:** **25 fields** across **19 documents** (**4.37% of alleged**).
- **Falsified / Overstated Cases:** **547 fields** (**95.63% of alleged**).

### Root Causes of the Overstated Claim:
1. **Python Boolean Inheritance Defect (286 fields / 50.0%):**
   In Python, `bool` inherits from `int` (`issubclass(bool, int) is True`).
   The classification heuristic in `gap_characterization.py`:
   ```python
   if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(r"\d", val)):
       if is_tbl:
           return "TOP_K_TRUNCATION"
       return "INDEXING_MISS"
   ```
   evaluated to `True` for boolean values (`val = False` or `val = True`). As a result, 286 boolean checkbox fields on IRS tax forms (Form 8949, Form 8879) and regulatory filings (Texas RRC Form W-14, H-12) with no baseline citation were misclassified as `INDEXING_MISS`.
2. **Conflation with OCR-Deficient Documents (148 fields / 25.87%):**
   Scanned PDFs where the PDF text layer contained zero tokens in the gold bounding box region were labeled as indexing misses instead of OCR coverage failures.
3. **Preexisting Production Resolution (15 fields / 2.62%):**
   The current production pipeline already resolves these fields with `IoU >= 0.50` (up to 0.86 IoU).
4. **Selection & Verification Failures (16 fields / 2.80%):**
   Candidates with `IoU >= 0.50` were successfully generated and present in the candidate pool, but were rejected by the candidate verifier or ranked below non-qualifying candidates.

---

## 5. Document Distribution

The 572 alleged cases and 25 genuine cases exhibit the following document distribution:

| Metric | Alleged in EXP-034R Handover | Actual Extracted | Audited Genuine `REAL_INDEXING_MISS` |
| :--- | :--- | :--- | :--- |
| **Total Fields** | 571 | 572 | **25** (4.37%) |
| **Unique Documents** | 136 | 137 | **19** (13.87%) |
| **Mean Fields / Doc** | 4.20 | 4.18 | **1.32** |
| **Max Fields in Single Doc** | 23 (`short/W14-58549_W14`) | 23 | **3** (`short/W14-58549_W14`) |

The concentration in single documents was driven almost entirely by multi-checkbox forms:
- `short/W14-58549_W14`: 23 alleged fields $\rightarrow$ **20 boolean checkboxes**, 3 genuine text misses.
- `short/W14-Atascosa SWD Well No. 4`: 19 alleged fields $\rightarrow$ **19 boolean checkboxes**, 0 genuine text misses.
- `short/W14-57728_W14_REVISED`: 18 alleged fields $\rightarrow$ **17 boolean checkboxes**, 1 unit-suffix string miss (`2415 psig`).

---

## 6. Root Causes (Full Reclassified Taxonomy)

Each of the 572 cases was traced through `DocumentIndex`, `EvidenceMatcher`, `CandidateRecoveryEngine`, `CandidateVerifier`, and `EvidenceResolver`.

The reclassified taxonomy across the benchmark is:

| Rank | Failure Mechanism | Fields | Documents | % of Alleged | Macro F1 Opportunity (pp) | Description |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `WRONG_CLASSIFICATION` | 286 | 71 | 50.00% | +1.2162 pp | Boolean checkboxes (`part2_box_e: False`, `schedule_8812.box: True`) misclassified due to Python `isinstance(bool, int)` |
| 2 | `OCR_CORRUPTION` | 148 | 68 | 25.87% | +0.9649 pp | Zero native text tokens extracted at gold coordinates in scanned / degraded pages |
| 3 | `DATE_INDEX_MISS` | 38 | 33 | 6.64% | +0.4499 pp | Non-standard date formats (e.g. `June 20th, 1955.`, `10/17/08`, `Oct. 8, 1953`) |
| 4 | `REAL_INDEXING_MISS` | **25** | **19** | **4.37%** | **+0.2293 pp** | Genuine text retrieval failure where native tokens exist on page but index/matcher failed to retrieve them |
| 5 | `NORMALIZATION_MISMATCH` | 22 | 12 | 3.85% | +0.1428 pp | Parenthesized negative numbers (e.g. `(243,682)` vs `-243682`), unit mismatches |
| 6 | `VERIFICATION_REJECTION` | 16 | 16 | 2.80% | +0.1842 pp | Qualifying candidate ($IoU \ge 0.50$) generated by recovery engine but rejected by verifier |
| 7 | `ALREADY_RESOLVED` | 15 | 13 | 2.62% | +0.1401 pp | Current production pipeline already resolves field successfully with $IoU \ge 0.50$ |
| 8 | `BBOX_RECONSTRUCTION` | 12 | 8 | 2.10% | +0.0913 pp | Correct tokens retrieved, but reconstructed bounding box achieved $IoU < 0.50$ |
| 9 | `HYPHENATION` | 5 | 4 | 0.87% | +0.0410 pp | Word split across line boundaries with trailing hyphens |
| 10 | `PAGE_ROUTING` | 5 | 2 | 0.87% | +0.0395 pp | Evidence located on an unexpected page filtered out by rigid page hint |
| **Total** | | **572** | **137** | **100.00%** | **+3.4991 pp** | |

---

## 7. Implementation

### Decision Gate Triggered: **GATE A — THE CLAIM IS INVALID**

Per Section 10 of the Research Directive:
> *If the alleged 571 cases substantially disappear after reclassification: `INDEXING_MISS was overstated`. Do not implement the proposed indexing fix... Then stop EXP-035 implementation.*

Because 95.6% of the alleged cases disappeared upon forensic verification, and the true remaining macro headroom of genuine indexing failure is only **+0.2293 pp** (and only 3 fields in Held-Out Cohort B), **no production code changes were made**.

Implementing an ungrounded character n-gram index or invasive token resegmentation would introduce false-positive candidate explosion, degrade latency, and risk regressions across the 236 established regression tests without addressing the actual bottlenecks (checkboxes, OCR, and date parsing).

---

## 8. Tests

The existing test suite was executed in the local virtual environment (`.venv/bin/pytest`) to confirm baseline health prior to decision gating:

```text
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
collected 236 items

tests/test_adversarial_suite.py ........                                 [  3%]
tests/test_benchmark_adapter.py ..                                       [  4%]
tests/test_candidate_expansion.py ....                                   [  5%]
tests/test_candidate_recovery.py .......                                 [  8%]
tests/test_candidate_verifier.py .......                                 [ 11%]
tests/test_character_span.py .....                                       [ 13%]
tests/test_date_fuzzy_page_drift.py ...........                          [ 18%]
tests/test_document_index.py ...                                         [ 19%]
tests/test_dot_leader_trimming.py .......                                [ 22%]
tests/test_exp006_digital_vs_grid_gate.py .....                          [ 25%]
tests/test_exp007_regressions.py ...................                     [ 33%]
tests/test_failure_correlation.py ......                                 [ 35%]
tests/test_geometry.py .......                                           [ 38%]
tests/test_hybrid_index.py ...............                               [ 44%]
tests/test_joint_record_resolver.py ............                         [ 50%]
tests/test_liteparse_index.py ........                                   [ 53%]
tests/test_long_document_grounding.py ......................             [ 62%]
tests/test_normalization.py ....                                         [ 64%]
tests/test_ocr_corrupted_grounding.py ...............                    [ 70%]
tests/test_ocr_repairs.py ..........                                     [ 75%]
tests/test_page_offset_calibration.py ..........                         [ 79%]
tests/test_safe_table_resolution.py .....                                [ 81%]
tests/test_same_line_recovery.py .........                               [ 85%]
tests/test_structural_disambiguation.py .                                [ 85%]
tests/test_structural_reranker.py ...........                            [ 90%]
tests/test_structure_classifier.py .........                             [ 94%]
tests/test_tax_form_grounding.py .....                                   [ 96%]
tests/test_token_gated_global_fallback.py .........                      [100%]

============================= 236 passed in 22.47s =============================
```
All 236 unit and integration tests passed cleanly.

---

## 9. Held-Out Benchmark (Cohort B Audit)

In accordance with Section 15 of the Directive, the alleged cases were audited specifically against **Held-Out Cohort B** (32 unseen documents, never tuned):

| Failure Mechanism | Cohort B Fields Affected | Cohort B Documents Affected | Cohort B % |
| :--- | :---: | :---: | :---: |
| `WRONG_CLASSIFICATION` (Boolean Checkboxes) | 25 | 5 | 39.68% |
| `OCR_CORRUPTION` (Zero tokens in text layer) | 17 | 9 | 26.98% |
| `ALREADY_RESOLVED` ($IoU \ge 0.50$) | 7 | 6 | 11.11% |
| `VERIFICATION_REJECTION` (Rejected by verifier) | 6 | 6 | 9.52% |
| `REAL_INDEXING_MISS` (Genuine retrieval failure) | **3** | **2** | **4.76%** |
| `NORMALIZATION_MISMATCH` (Negatives / formatting) | 2 | 1 | 3.17% |
| `DATE_INDEX_MISS` (Date formats) | 2 | 2 | 3.17% |
| `BBOX_RECONSTRUCTION` (Geometry bounds) | 1 | 1 | 1.59% |
| **Total Cohort B Alleged Cases** | **63** | **22** | **100.00%** |

In Held-Out Cohort B, only **3 fields across 2 documents** were genuine indexing misses.

---

## 10. Full Benchmark

Per Section 16 of the Directive:
> *Only run the full 370-document official benchmark after the targeted experiment demonstrates that the mechanism actually works.*

Because Gate A triggered and implementation was halted, the production baseline remains untouched at:
- **Full 370 Word Grounding F1:** **56.0477%**
- **Full 370 Page Grounding F1:** **81.6639%**
- **Cohort B Word Grounding F1:** **59.3588%**

---

## 11. Macro Impact

The corrected document-macro contribution of genuine indexing misses is:
- **Claimed Opportunity:** +3.4991 pp Word Grounding F1
- **Audited Opportunity:** **+0.2293 pp Word Grounding F1**
- **Discrepancy:** -3.2698 pp (93.45% of the claimed gain does not exist in indexing)

The top macro opportunity in the audited set is **Boolean Checkbox Visual Grounding (+1.2162 pp)**, followed by **OCR Coverage (+0.9649 pp)** and **Date Indexing (+0.4499 pp)**.

---

## 12. Regression Analysis

Because Gate A halted speculative production changes, **zero regressions** were introduced:
- New regressions: **0**
- Regressed documents: **0**
- Test suite pass rate: **100% (236/236 passed)**

---

## 13. Decision

### **DECISION: FALSIFIED / STOP**

1. The hypothesis that `INDEXING_MISS` is the #1 candidate-generation failure class is **FALSIFIED**.
2. The alleged 571 cases collapsed by 95.6% to just 25 genuine fields (+0.2293 pp).
3. The original claim was an artifact of Python `isinstance(bool, int)` classifying 286 checkboxes as numeric misses and conflating scanned documents with indexing failure.
4. Per Gate A, EXP-035 implementation is **STOPPED**.

---

## 14. Next Experiment Recommendation

Only one concrete next experiment is recommended:

**EXP-036: Visual Form Checkbox & Boolean Grounding Engine**  
Directly target the verified 286 boolean checkbox fields across 71 documents (+1.2162 pp macro F1) by implementing visual bounding-box anchoring for checked/unchecked form boxes (`[ ]`, `[X]`, radio buttons) on IRS tax forms (Form 8879, 8949) and regulatory filings (Form W-14, H-12).
