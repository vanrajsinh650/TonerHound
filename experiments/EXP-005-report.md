# EXP-005 Engineering Report: Pushing Toward 90% Grounding Accuracy

**Project**: TonerHound  
**Iteration**: EXP-005  
**Date**: 2026-09-14  
**Baseline Metric**: EXP-004 Official Full Benchmark F1: **45.49%** (Word Precision: 59.45%, Word Recall: 38.61%, Page F1: 59.11%, Candidate Recall@20: 78.97%)  
**Benchmark Target**: Stratified Frozen Local Benchmark (32 documents, 881 pages, 131,811 citations)

---

## 1. Frozen Local Benchmark Formulation

To enable rapid, deterministic, and rigorous algorithmic iteration without running the 370-document suite prematurely (which takes several hours and obscures per-stage errors), we created a frozen 32-document representative manifest (`benchmarks/exp005_local_manifest.json`):

- **Train / Development Set**: 20 documents (Short: 9, Medium: 7, Long: 4)
- **Local Validation Set**: 12 documents (Short: 5, Medium: 5, Long: 2)
- **Coverage**:
  - Dense tabular filings (SEC 13F, Form 1065 K-1, IRS Schedule I)
  - Unstructured narrative & legal deeds (Clinton property deed, bankruptcy service lists)
  - Scanned and corrupted OCR documents (`real_clinton_property_corrupted`, `real_ftx_full_corrupted`, `real_bbb_service_list_corrupted`)
  - Massive long-document filings (FTX 114 pages, iMedia 114 pages, Credit Strategies 59 pages, Renaissance 13F 81 pages)
  - Multi-line addresses, dates, currency amounts, boolean checkboxes, and repeated entity names

---

## 2. Baseline Benchmark Run (`EXP-005-baseline`)

The baseline system was executed across all 32 documents in the frozen local suite:

| Metric | Overall (32 Docs) | Train / Dev (20 Docs) | Local Val (12 Docs) | Short | Medium | Long |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | **38.58%** | **35.92%** | **43.00%** | **51.50%** | **31.89%** | **21.78%** |
| **Word Precision** | 52.36% | 49.80% | 56.63% | 65.61% | 46.85% | 29.80% |
| **Word Recall** | 32.70% | 29.81% | 37.51% | 45.42% | 25.75% | 18.06% |
| **Page Grounding F1** | 61.01% | 58.62% | 65.00% | 76.81% | 55.40% | 40.85% |
| **False Grounding Rate** | 47.64% | 50.20% | 43.37% | 34.39% | 53.15% | 70.20% |
| **Total Runtime** | 926.8s (15.4m) | — | — | — | — | — |

---

## 3. Diagnostic Error Analysis: Why Medium & Long Docs Collapsed

Deep investigation into the baseline failures uncovered two catastrophic algorithmic bottlenecks:

### Pathology 1: Artificial Page Pruning on Documents > 10 Pages
In `src/tonerhound/matching/matcher.py`:
- Lines 165, 205, 260 contain: `if page_hint is None and len(self.index.pages) > 10: return []`
- Because ExtractBench almost never provides a page hint, numeric matching (`find_normalized_numeric_candidates`), date matching (`find_normalized_date_candidates`), and fuzzy matching (`find_fuzzy_candidates`) were **completely disabled** on every document exceeding 10 pages!
- Documents like `cabrera-2023` (28 pages, 97 numeric fields), `becerra-2024` (27 pages, 221 numeric fields), and `real_credit_strategies_full` (59 pages, 2,227 numeric fields) suffered near-total candidate failure (F1 between 0.3% and 10.5%).
- **Cause**: This 10-page abort was originally introduced because linear scanning over all pages was taking 15+ minutes.
- **Solution**: Replace linear scans with an Inverted Numeric and Token Index.

### Pathology 2: Lack of Inverted Index for Exact & Numeric Lookups
- For every query, `find_exact_candidates` performed a linear nested loop: `for page in pages: for line in page.lines: ...`
- On large documents, this generated tens of millions of string checks per document.

---

## 4. Algorithmic Roadmap & Execution Sprints

1. **Sprint 1 (Candidate Generation & Inverted Index)**:
   - Implement Inverted Token, Numeric Canonical, and N-gram Index in `DocumentIndex`.
   - Remove the artificial 10-page truncation in `matcher.py`.
   - Measure Candidate Recall@20 and Word Grounding F1 across Medium and Long documents.
2. **Sprint 2 (Cascading Alignment Engine)**:
   - Implement candidate-local Smith-Waterman and token alignment with character bounding box slicing.
   - Group multi-line evidence bounding boxes.
3. **Sprint 3 (Structural Table Reasoning & Record Anchoring)**:
   - Upgrade table row anchoring to handle non-anchor-keyword fields, multi-table layouts, and monotonic sequence assignments.
4. **Sprint 4 (Verification & Calibrated Abstention)**:
   - Fine-tune score margins and geometric plausibility gates on `train_dev` to drive False Grounding Rate toward zero.
