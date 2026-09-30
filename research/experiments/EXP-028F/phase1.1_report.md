# TONERHOUND — EXP-028F PHASE 1.1 REPORT
## Safe Date/Fuzzy Page Drift Implementation

**Date:** 2026-09-30  
**Status:** COMPLETE — ALL DECISION GATES PASSED  
**Commit Target:** `exp028f-phase1.1: allow page drift for date/fuzzy matching`  
**Prior Verified Baselines:**
- EXP-028D Targeted Smoke Suite Average: 62.84% Word Grounding F1
- EXP-028E Full 370-doc Production: 56.05% Word Grounding F1 (212 unit tests passing)

---

## 1. Executive Summary

During the EXP-028F Phase 0 audit (`research/experiments/EXP-028F/phase0_production_audit.md`), **Defect 6 (Strict Page-Hint Filtering in Date/Fuzzy Matching)** was confirmed:
- `find_normalized_date_candidates()` strictly skipped any token occurrences outside `page_hint`.
- `find_fuzzy_candidates()` strictly dropped all n-gram and word postings on pages other than `page_hint`.
- While `find_exact_candidates()` and `find_normalized_numeric_candidates()` already implemented `_allow_drift` (`page_hint +/- 1`), date and fuzzy matching discarded valid physical evidence on adjacent pages.

In **EXP-028F Phase 1.1**, we safely implemented bounded page drift (`page_hint +/- 1`) behind the feature flag `ENABLE_STRICT_PAGE_HINT`, strictly preventing unrestricted document-wide search (which is reserved for Phase 1.2).

### Key Accomplishments
1. **Narrow, Non-Breaking Implementation**:
   - Added module-level feature flag `ENABLE_STRICT_PAGE_HINT: bool = False` in `src/tonerhound/matching/matcher.py`.
   - Updated `EvidenceMatcher` to support instance-level and call-level `enable_strict_page_hint`.
   - Updated `find_normalized_date_candidates` and `find_fuzzy_candidates` with bounded drift (`page_hint +/- 1`).
   - Prioritized exact `page_hint` candidates over drifted candidates using priority sorting `(0 if c.page == page_hint else 1, -c.raw_similarity)` and a 0.95 similarity penalty on drifted candidates.
2. **Comprehensive Unit Verification**:
   - Added `tests/test_date_fuzzy_page_drift.py` covering all 5 prompt-specified test cases plus flag override tests (11 new tests).
   - All 223 unit tests passing (212 baseline + 11 new tests in 16.68s).
3. **Targeted 7-Document Benchmark**:
   - Total runtime: **24.70s** (well under the 60s safety ceiling).
   - Smoke suite average: **62.84%** (non-regressive to baseline).
   - All individual documents met or exceeded decision gate thresholds.
   - Overall Gate Status: **PASSED**.

---

## 2. Implementation Details

Target File: `src/tonerhound/matching/matcher.py`

### Feature Flag Specification
```python
# Feature flag for EXP-028F Phase 1.1: Strict page-hint filtering vs +/- 1 page drift
ENABLE_STRICT_PAGE_HINT: bool = False
```
- When `ENABLE_STRICT_PAGE_HINT = True`: Preserves legacy strict production filtering (strictly skips `p_num != page_hint`).
- When `ENABLE_STRICT_PAGE_HINT = False`:
  1. Searches `page_hint` first.
  2. If no candidate found on `page_hint`, allows bounded drift to `page_hint - 1` and `page_hint + 1`.
  3. Applies 0.95 similarity discount to drifted candidates.
  4. Ranks exact `page_hint` candidates ahead of drifted candidates (`candidates.sort(key=lambda c: (0 if c.page == page_hint else 1, -c.raw_similarity))`).
  5. Strictly prohibits global document-wide fallback (`page_hint +/- 2` and beyond are excluded).

### Resolution Method
```python
class EvidenceMatcher:
    def __init__(
        self,
        index: DocumentIndex,
        enable_strict_page_hint: bool | None = None,
    ) -> None:
        self.index = index
        self.enable_strict_page_hint = enable_strict_page_hint
        self._numeric_cache: dict[int, list[tuple[DocumentToken, float, int]]] = {}

    def _resolve_strict_page_hint(self, override: bool | None = None) -> bool:
        """Resolve effective strict page hint setting, respecting method overrides."""
        if override is not None:
            return override
        if self.enable_strict_page_hint is not None:
            return self.enable_strict_page_hint
        return ENABLE_STRICT_PAGE_HINT
```

---

## 3. Unit Test Verification

File: `tests/test_date_fuzzy_page_drift.py`

| Test Function | Description | Result |
|---|---|---|
| `test_target_evidence_on_page_hint_date` | Date evidence on `page_hint` returned with flag True and False | **PASSED** |
| `test_target_evidence_on_page_hint_fuzzy` | Fuzzy evidence on `page_hint` returned with flag True and False | **PASSED** |
| `test_target_evidence_on_page_hint_plus_one_date` | Date on `page_hint + 1` returned when False, blocked when True | **PASSED** |
| `test_target_evidence_on_page_hint_plus_one_fuzzy` | Fuzzy on `page_hint + 1` returned when False, blocked when True | **PASSED** |
| `test_target_evidence_on_page_hint_minus_one_date` | Date on `page_hint - 1` returned when False | **PASSED** |
| `test_target_evidence_on_page_hint_minus_one_fuzzy` | Fuzzy on `page_hint - 1` returned when False | **PASSED** |
| `test_target_evidence_on_page_hint_plus_two_date` | Date on `page_hint + 2` NOT returned (confirms no global fallback) | **PASSED** |
| `test_target_evidence_on_page_hint_plus_two_fuzzy` | Fuzzy on `page_hint + 2` NOT returned (confirms no global fallback) | **PASSED** |
| `test_page_hint_candidate_priority_date` | Date on both `page_hint` and `page_hint + 1`: index 0 is on `page_hint` | **PASSED** |
| `test_page_hint_candidate_priority_fuzzy` | Fuzzy on both `page_hint` and `page_hint + 1`: index 0 is on `page_hint` | **PASSED** |
| `test_module_flag_and_instance_default` | Module flag toggle and instance override behavior | **PASSED** |

**Full Suite Status:** 223 passed in 16.68s.

---

## 4. Benchmark Verification (7 Targeted Documents)

Evaluated with official ExtractEvaluator (`research/experiments/EXP-028F/run_targeted_benchmark.py --workers 2`):

| Document | Word F1 | EXP-028D | Delta | Word Prec | Page F1 | False Grnd | Abstention |
|---|---|---|---|---|---|---|---|
| `long/real_sm0801_eco_full` | 92.48% | 92.48% | -0.00pp | 92.50% | 97.57% | 7.50% | 11.21% |
| `medium/sec_13f_0031_loomis_sayles` | 86.28% | 86.30% | -0.02pp | 87.38% | 93.82% | 12.62% | 21.10% |
| `short/W14-Atascosa SWD Well No. 4` | 54.01% | 54.01% | +0.00pp | 68.52% | 61.09% | 31.48% | 56.48% |
| `short/bianco-2024` | 31.44% | 31.44% | -0.00pp | 32.64% | 69.57% | 67.36% | 88.33% |
| `medium/real_pueblo_oct_2025` | 99.60% | 99.60% | +0.00pp | 99.63% | 99.84% | 0.37% | 0.33% |
| `short/real_wyo_Goshen_2024` | 99.49% | 99.49% | -0.00pp | 100.00% | 99.65% | 0.00% | 0.27% |
| `medium/veralto_earnings_deck_q4fy25` | 0.00% | 0.00% | +0.00pp | 0.00% | 0.00% | 0.00% | 19.05% |
| **Smoke Suite Average (6 Docs)** | **62.84%** | **62.84%** | **-0.00pp** | — | — | — | — |
| **All 7 Targeted Docs Average** | **66.19%** | — | — | — | — | — | — |

### Decision Gate Checks

| Criterion | Measured | Threshold | Gate Status |
|---|---|---|---|
| SM0801 Word F1 | 92.48% | >= 91.48% | **PASSED** |
| Loomis Sayles Word F1 | 86.28% | >= 85.30% | **PASSED** |
| Atascosa Word F1 | 54.01% | >= 53.00% | **PASSED** |
| Bianco Word F1 | 31.44% | >= 30.44% | **PASSED** |
| Pueblo Word F1 | 99.60% | >= 98.60% | **PASSED** |
| Goshen Word F1 | 99.49% | >= 98.49% | **PASSED** |
| Veralto Word F1 | 0.00% | >= 0.00% | **PASSED** |
| Smoke Suite Average | 62.84% | >= 62.84% | **PASSED** |
| Total Benchmark Runtime | 24.70s | <= 60.0s | **PASSED** |

**Overall Gate Status:** **PASSED — ALL CRITERIA SATISFIED**

---

## 5. Artifacts Produced

1. Code changes in `src/tonerhound/matching/matcher.py`:
   - Feature flag `ENABLE_STRICT_PAGE_HINT`
   - Bounded drift for dates and fuzzy queries (`page_hint +/- 1`)
   - Candidate priority ordering
2. Unit tests in `tests/test_date_fuzzy_page_drift.py` (11 tests)
3. Benchmark suite in `research/experiments/EXP-028F/`:
   - `run_targeted_benchmark.py`
   - `per_document.csv`
   - `results.json`
   - `phase1.1_report.md`

---

## 6. Next Steps & Stop Condition

In strict accordance with the prompt guidelines:
- We do **NOT** begin Phase 1.2 (Long-Document & Global Fallback).
- We do **NOT** run the full 370-document ExtractBench benchmark.
- All code and artifacts are committed and pushed to `origin/main`.
