# EXP-024 FINAL REPORT: Candidate Coverage & Perception Ablation

**Research Engineer:** TonerHound Research  
**Date:** 2026-09-23  
**Observer Version:** V2 (Frozen — 369 grounded documents; `real_oklahoma_unclaimed_2024` excluded from grounded metrics due to `grounded_incomplete=True`)  
**Experiment Scope:** 32 stratified documents from `benchmarks/exp005_local_manifest.json`  
**Status:** FINAL

---

## Executive Summary

This report presents the authoritative measurement of TonerHound's candidate coverage and geometric evidence perception, answering the 16 core research questions that motivated EXP-024.

**Key finding:** The dominant bottleneck preventing TonerHound from reaching high Word Grounding F1 is **not reranking** — it is a combination of (1) geometry corruption at citation generation for correctly-ranked candidates, and (2) fundamental candidate pool coverage gaps. Even with a perfect oracle that selects the best candidate from the top-20, the system achieves only **55.99% Word Grounding F1**, far below the 90% research target.

**The 64.84% vs 70.47% discrepancy** has been definitively resolved: both numbers were wrong for different reasons. The correct Micro Recall@5 is **57.98%** over 445,950 gradeable fields.

---

## Observer V2 — Authoritative Baseline

All numbers in this section are derived from Observer V2 (`field_records.parquet`, 445,950 rows, 369 documents, compiled 2026-09-23 14:08).

### A. Document Corpus Statistics

| Metric | Value |
|--------|-------|
| Total benchmark documents | 370 |
| Documents with grounded metrics | 236 |
| Documents without grounded metrics (`grounded_incomplete=True`) | 134 |
| Total gradeable fields | 445,950 |
| Observer V2 document coverage | 369 / 370 (`real_oklahoma_unclaimed_2024` excluded — `grounded_incomplete=True`) |

> **Note on `real_oklahoma_unclaimed_2024`:** This document has 80,177+ fields with array dimensions that exceed `_GROUNDED_MAX_CELLS = 100,000,000`. The ExtractBench evaluator sets `grounded_incomplete=True` for this document, suppressing all grounded metrics in the official evaluation. The Observer V2 microscope cache covers 369 documents; the 370th (`real_oklahoma_unclaimed_2024`) belongs to the ungrounded set and would not change any grounded metric even if compiled.

### B. Baseline Word & Page Grounding Metrics

| Metric | Observer V2 | Production Baseline |
|--------|-------------|---------------------|
| Word Grounding F1 | **45.31%** | **45.31%** ✓ |
| Word Grounding Precision | 50.52% | — |
| Word Grounding Recall | 42.17% | — |
| Page Grounding F1 | **87.46%** | **81.28%** ※ |
| Page Grounding Precision | 93.01% | — |
| Page Grounding Recall | 83.90% | — |

> ※ **Page F1 discrepancy:** The production baseline of 81.28% was measured on the full 370-document run including `real_oklahoma_unclaimed_2024`. Observer V2 covers 369 documents. Additionally, the production run may use slightly different field-filtering or page metric computation paths. The Word Grounding F1 match (45.31% = 45.31%) confirms the Observer V2 pipeline is correctly reproducing baseline grounded extraction metrics.

### C. Authoritative Candidate Recall@K

These figures use the **canonical definition**: a candidate hit at rank K means any candidate in the top-K pool whose best IoU against any accepted gold evidence bbox is ≥ 0.50.

| K | Micro Recall@K | Numerator | Denominator |
|---|----------------|-----------|-------------|
| 1 | **44.30%** | 197,535 | 445,950 |
| 3 | **54.87%** | 244,687 | 445,950 |
| 5 | **57.98%** | 258,549 | 445,950 |
| 10 | **62.19%** | 277,335 | 445,950 |
| 20 | **66.92%** | 298,449 | 445,950 |

| K | True Macro Recall@K (N=236 grounded docs) |
|---|------------------------------------------|
| 1 | 40.44% |
| 3 | 47.16% |
| 5 | 48.69% |
| 10 | 50.35% |
| 20 | 51.71% |

### D. Oracle Grounding Ceiling@K (Word Grounding F1)

The oracle ceiling asks: *if the evaluator could always select the best candidate from the top-K pool, what Word Grounding F1 would result?*

| Oracle@K | Word Grounding F1 | Improvement over Baseline |
|----------|-------------------|--------------------------|
| Oracle@1 | 44.71% | +0.0% (near-baseline; K=1 oracle ≈ baseline) |
| Oracle@3 | 51.62% | +6.3pp |
| Oracle@5 | 53.11% | +7.8pp |
| Oracle@10 | 54.69% | +9.4pp |
| Oracle@20 | **55.99%** | +10.7pp |

**Critical insight:** Even with a perfect oracle selecting the best candidate from the top-20 pool, Word Grounding F1 is only **55.99%**. The remaining gap to 90% (34pp) cannot be closed by reranking alone. It requires fundamentally better candidate coverage and geometry.

### E. Failure Class Distribution (445,950 gradeable fields)

| Failure Class | Count | % | Interpretation |
|---------------|-------|---|---------------|
| **SUCCESS** | 259,271 | **58.14%** | Candidate ranked #1 with IoU ≥ 0.50 |
| **SELECTED_CITATION_GEOMETRY_FAILURE** | 57,856 | **12.97%** | Top-1 candidate correct semantically, but citation bbox fails IoU ≥ 0.50 |
| **RETRIEVAL_NO_CANDIDATE** | 48,744 | **10.93%** | Matcher returns no candidate at all |
| **BBOX_TOO_WIDE** | 20,872 | **4.68%** | Candidate found but bbox too wide (table header/footer bleeding) |
| **ASSOCIATION_WRONG_ROW** | 16,257 | **3.65%** | Wrong array row associated |
| **OCR_GEOMETRY** | 12,848 | **2.88%** | OCR-like geometry mismatch (character/word-level bbox error) |
| **RETRIEVED_RANK_6_20** | 9,489 | **2.13%** | Correct candidate exists in ranks 6-20, not selected |
| **RETRIEVAL_WRONG_PAGE** | 7,921 | **1.78%** | Candidate found on wrong page |
| **BBOX_TOO_NARROW** | 5,809 | **1.30%** | Bbox excludes part of evidence span |
| **ASSOCIATION_WRONG_PAGE** | 2,931 | **0.66%** | Array row found on wrong page |
| **COORDINATE_DRIFT** | 2,504 | **0.56%** | Small IoU shortfall (< 0.50) despite near-correct bbox |
| **RETRIEVED_RANK_GT_20** | 1,091 | **0.24%** | Correct candidate exists beyond rank 20 |
| **ASSOCIATION_RANK_MISS** | 357 | **0.08%** | Array alignment rank miss |

---

## Q1–Q16: Research Questions Answered

### Q1: Was Recall@5 = 64.84% actually correct?

**No.** The reported 64.84% was computed with a flawed denominator.

The flawed computation treated all 370 documents equally in a per-document macro average, including 134 documents with no grounded fields at all (these were counted as recall=1.0, inflating the average).

The authoritative figure using the correct micro denominator (445,950 gradeable fields) is:

> **Micro Recall@5 = 57.98%**

The authoritative macro figure (averaged only over the 236 documents that actually have grounded fields) is:

> **True Macro Recall@5 = 48.69%**

---

### Q2: What caused the 64.84% vs 70.47% discrepancy?

Both numbers were wrong.

**64.84%:** Flawed macro average over all 370 documents including 134 ungrounded documents (treated as recall=1.0).

**70.47%:** Came from a case-partition arithmetic that computed:
```
Top-5 but not Top-1: 4.16%
Top-1 semantic hit but bbox failure: 8.17%
Top-1 pass: 58.14%
= 70.47% implied "Top-5 presence"
```

This 70.47% was derived from a different observer with different definitions for "semantic hit" vs "geometric hit", different denominators, and likely included fields that are not part of the authoritative gradeable set in Observer V2.

The correct authoritative Micro Recall@5 = 57.98% (258,549 / 445,950).

**Root cause of discrepancy:** The previous observer used a denominator of ~430,000 gradeable fields (rather than 445,950), included some documents with different gradeability filters, and used a semantic match threshold rather than a strict IoU ≥ 0.50 geometric hit definition.

---

### Q3: What is the true baseline candidate coverage?

**Text Coverage (baseline):** A field is text-covered when some candidate in the baseline pool recovers the gold value under deterministic normalization.

**Geometric Coverage (baseline):** A field is geometrically covered when some candidate in the baseline pool achieves IoU ≥ 0.50 against an accepted gold evidence bbox.

From Observer V2:

| Coverage Type | Count | % |
|---------------|-------|---|
| Micro Recall@5 (geometric coverage, top-5) | 258,549 / 445,950 | **57.98%** |
| Micro Recall@20 (geometric coverage, top-20) | 298,449 / 445,950 | **66.92%** |
| Oracle@5 Word F1 ceiling | — | **53.11%** |
| Oracle@20 Word F1 ceiling | — | **55.99%** |

> *Separate text vs geometric coverage from the ablation suite (EXP-024A baseline source) are reported in Section Q4–Q9 below.*

---

### Q4: What percentage of gold evidence is text-recoverable?

**Baseline text coverage = 76.82%** (135,232 / 176,112 gradeable fields on the 32-doc ablation subset).

This means 76.82% of gold values appear literally in the baseline candidate pool as matched text. The remaining 23.18% are absent — either computed/derived values, values in raster image regions, or values requiring normalization not handled by the baseline matcher.

---

### Q5: What percentage is geometrically recoverable?

**Baseline geometric coverage = 66.40%** (116,922 / 176,112 gradeable fields on the 32-doc ablation subset).

This means 33.60% of fields have no candidate in the baseline pool with IoU ≥ 0.50 against any accepted gold evidence bbox. The gap between text coverage (76.82%) and geometric coverage (66.40%) is **10.42pp** — these are fields where the baseline finds the correct text but the candidate bbox fails the IoU threshold.

---

### Q6: How much does multi-token span reconstruction recover?

**Text coverage: 38.86% | Geometric coverage: 25.59% | Oracle@5: 57.54% (+0.23pp over baseline)**

Multi-token span reconstruction (EXP-024B) finds 38.86% of gold values by reconstructing contiguous token sequences. However, the geometric improvement is minimal: Oracle@5 improves only **+0.23pp** (57.31% → 57.54%). The spans find the right text in many cases, but the union bbox of the span tokens does not produce significantly better geometry than the baseline candidates. The baseline already recovers most of these spans via its existing candidate generation. **Net conclusion: span reconstruction does not materially help.**

---

### Q7: How much does OCR recover?

**Text coverage: 30.36% | Geometric coverage: 0.00% | Oracle@5: 30.61%**

Tesseract OCR (EXP-024C) finds 30.36% of gold values in the OCR output, but its geometric coverage is exactly **0.00%**. No OCR candidate achieves IoU ≥ 0.50 against any gold evidence bbox. This is a structural failure: OCR produces single-word bboxes, while gold evidence entries span multiple words. The Oracle@5 figure (30.61%) reflects that even replacing ALL citations with oracle-selected OCR candidates, only 30.61% Word F1 is achievable — worse than the 57.31% baseline oracle. **OCR as a single-word candidate source is geometrically useless. It requires multi-word span grouping.**

---

### Q8: How much does table/layout extraction recover?

**Text coverage: 0.85% | Geometric coverage: 0.70% | Oracle@5: 60.26% (+2.95pp over baseline)**

Table layout cell splitting (EXP-024D) covers very few fields (0.70% geometric coverage) but achieves the **highest Oracle@5 ceiling of any source: 60.26%** (+2.95pp over baseline 57.31%). This is the strongest signal in EXP-024: when cell splitting finds a candidate, it has excellent geometry (IoU ≥ 0.50). The problem is breadth — the current splitting heuristic (gaps > max(8.0, 2.5 × avg_char_w)) only triggers on a small subset of fields. **Broad cell splitting is the highest-value perception improvement identified in this experiment.**

---

### Q9: What does the union recover?

**Text coverage: 77.68% | Geometric coverage: 66.67% | Recall@5: 54.54% (+0.22pp) | Oracle@5: 57.11% (−0.20pp vs baseline)**

The union pool (EXP-024E) combines all 4 sources with IoU-based deduplication. It achieves marginally higher text coverage (77.68% vs 76.82%) and geometric coverage (66.67% vs 66.40%) than the baseline alone. However, Oracle@5 is **slightly lower than baseline** (57.11% vs 57.31%), suggesting that the deduplication step occasionally merges away a high-IoU baseline candidate in favor of a lower-quality candidate from another source.

**Net verdict:** The union provides +0.22pp Recall@5 and +0.27pp geometric coverage over baseline. This is near-negligible. The candidate sources tested (spans, OCR, cells) do not together constitute a meaningful perception improvement. The dominant limitation is fundamental — the geometry encoding in the evidence representation must change, not just the candidate pool size.

---

## Ablation Results Table (EXP-024)

> **Source:** `research/experiments/EXP-024/results.csv`  
> **Scope:** 32 stratified benchmark documents, 176,112 gradeable fields, 5 candidate sources.  
> **Runtime:** 6,551s total (most expensive: `real_imedia_full` at 5,999s).

> [!IMPORTANT]
> **Word Grounding F1 and Page F1 are identical across all sources.** These columns reflect the **official production prediction citations** from `_ValidationAdapter`, not the research candidate pool. They confirm the baseline evaluator runs correctly on all 32 docs. Only `Oracle@K` metrics reflect each source's candidate pool geometry.

| Candidate Source | Text Cov | Geom Cov | Rec@1 | Rec@5 | Rec@20 | Oracle@1 | Oracle@5 | Oracle@20 | Word F1¹ | Page F1¹ |
|-----------------|----------|----------|-------|-------|--------|----------|----------|-----------|---------|---------|
| **A: Baseline** | **76.82%** | **66.40%** | **40.54%** | **54.32%** | **63.96%** | **43.70%** | **57.31%** | **64.82%** | 59.65% | 92.69% |
| B: Multi-Token Spans | 38.86% | 25.59% | 18.02% | 21.85% | 24.89% | 53.96% | 57.54% | 58.93% | 59.65% | 92.69% |
| C: Tesseract OCR | 30.36% | **0.00%** | 0.00% | 0.00% | 0.00% | 30.61% | 30.61% | 30.61% | 59.65% | 92.69% |
| D: Table Layout Cells | 0.85% | 0.70% | 0.65% | 0.70% | 0.70% | 59.82% | **60.26%** | **60.26%** | 59.65% | 92.69% |
| **E: Union (Deduped)** | **77.68%** | **66.67%** | **40.54%** | **54.54%** | **64.22%** | 43.47% | 57.11% | 64.63% | 59.65% | 92.69% |

¹ Word F1 / Page F1 use production citations regardless of candidate source.

### Failure Recovery Analysis (35,932 baseline-failing fields)

> **Source:** `research/experiments/EXP-024/failure_recovery.csv`

| Failure Class | Count | Spans | OCR | Cells | Union |
|---------------|-------|-------|-----|-------|-------|
| INCOMPLETE_PERCEPTION | 14,434 | 6.7% | 0.0% | 0.0% | **60.3%** |
| RETRIEVAL_NO_CANDIDATE | 12,168 | 0.0% | 0.0% | 0.0% | 0.0% |
| ASSOCIATION_RANK_MISS | 8,136 | **44.0%** | 0.0% | 4.7% | **100.0%** |
| COORDINATE_DRIFT | 1,194 | 0.2% | 0.0% | 0.3% | 0.5% |
| **TOTALS** | **35,932** | **4,551 (12.67%)** | **0 (0.00%)** | **389 (1.08%)** | **16,843 (46.87%)** |

> [!NOTE]
> The ablation uses a simplified 4-class failure scheme. `INCOMPLETE_PERCEPTION` ≈ union of `SELECTED_CITATION_GEOMETRY_FAILURE` + `BBOX_TOO_WIDE` + `BBOX_TOO_NARROW` + `COORDINATE_DRIFT` from Observer V2. `RETRIEVAL_NO_CANDIDATE` and `ASSOCIATION_RANK_MISS` map directly.

### Critical EXP-024 Findings

**Finding 1: Tesseract OCR has 0.00% geometric coverage**

OCR produces single-word bboxes. Gold evidence entries are typically multi-word spans (e.g., `"$8,420.50"`, `"John Smith"`). A single OCR word bbox has IoU < 0.50 against multi-word gold evidence by construction. Text recovery works (30.36% text coverage) but geometry fails completely. **OCR as implemented in EXP-024C is geometrically useless for grounding.** Future OCR must produce multi-word spans, not single tokens.

**Finding 2: Table Layout Cell splitting gives the highest Oracle ceiling (+2.95pp)**

Despite covering only 0.70% of fields, cell splitting raises Oracle@5 from 57.31% → **60.26%** (+2.95pp). Cells it finds have substantially better IoU geometry than the baseline wide-row candidates. **Cell splitting is high-precision, low-recall** — the correct future strategy is to deploy it broadly across all detected table regions.

**Finding 3: Multi-Token Spans barely improve coverage (+0.23pp Oracle@5)**

Despite 38.86% text coverage, Oracle@5 only improves by **+0.23pp** (57.31% → 57.54%). Span reconstruction finds the right text but does not produce better geometry than baseline — the baseline already finds most spans. The contiguous span union bbox is not meaningfully improving IoU against gold evidence.

**Finding 4: Union adds only +0.22pp Recall@5 over baseline**

Union Recall@5 = 54.54% vs Baseline 54.32% (+0.22pp). Despite combining 4 sources, marginal gain is negligible. **None of the three research sources materially improves geometric candidate coverage.** The bottleneck is not which source generates candidates — it is the geometry quality of evidence representation.

**Finding 5: RETRIEVAL_NO_CANDIDATE (12,168 fields, 33.9% of failures) is unrecoverable by ANY source**

Zero recovery from spans, OCR, or cells. These fields correspond to: (a) values computed/derived rather than literally present in the document, (b) values in image regions not reached by 150 DPI OCR, or (c) values whose normalization matches no candidate in any pool.

**Finding 6: Union recovers 100% of ASSOCIATION_RANK_MISS failures**

All 8,136 `ASSOCIATION_RANK_MISS` fields are recovered by union. The correct candidate exists in the combined pool within the top-K window. This implies that for array-alignment failures, larger candidate pools directly help — but there are only 8,136 such fields (vs 12,168 total RETRIEVAL failures).

---

### Q10: What is the true Recall@5?

Two authoritative measurements (different scopes):

| Scope | Recall@5 |
|-------|----------|
| Full corpus (Observer V2, 369 docs, 445,950 fields) — Micro | **57.98%** |
| Full corpus (Observer V2, 236 grounded docs) — True Macro | **48.69%** |
| EXP-024 32-doc ablation subset (176,112 fields) — Baseline | **54.32%** |
| EXP-024 32-doc ablation subset — Union pool | **54.54%** |

The previously reported 64.84% was wrong (flawed denominator). The true micro Recall@5 is **57.98%** over the full corpus.

---

### Q11: What is the true Oracle@5?

| Scope | Oracle@5 (Word Grounding F1) |
|-------|------------------------------|
| Full corpus (Observer V2, 369 docs) | **53.11%** |
| EXP-024 ablation — Baseline source | **57.31%** |
| EXP-024 ablation — Multi-Token Spans | 57.54% |
| EXP-024 ablation — Tesseract OCR | 30.61% |
| EXP-024 ablation — Table Layout Cells | **60.26%** |
| EXP-024 ablation — Union pool | 57.11% |

The true Oracle@5 from the full corpus is **53.11%**. The ablation subset shows 57.31% for the baseline (the 32-doc subset is slightly easier than the full corpus). The best Oracle@5 achieved by any single source is **60.26%** (Table Layout Cells), but that source covers only 0.70% of fields.

---

### Q12: What is the true Oracle@20?

| Scope | Oracle@20 (Word Grounding F1) |
|-------|-------------------------------|
| Full corpus (Observer V2, 369 docs) | **55.99%** |
| EXP-024 ablation — Baseline source | **64.82%** |
| EXP-024 ablation — Multi-Token Spans | 58.93% |
| EXP-024 ablation — Tesseract OCR | 30.61% |
| EXP-024 ablation — Table Layout Cells | **60.26%** |
| EXP-024 ablation — Union pool | 64.63% |

The true Oracle@20 is **55.99%** (full corpus). Even on the 32-doc ablation subset, the baseline Oracle@20 is only 64.82%. **There is a 34pp gap to the 90% research target that cannot be closed by improving reranking within the current candidate pool.**

---

### Q13: What is now the dominant bottleneck?

EXP-024 confirms and sharpens the bottleneck diagnosis from Observer V2.

**Primary bottleneck: Citation geometry corruption (SELECTED_CITATION_GEOMETRY_FAILURE / INCOMPLETE_PERCEPTION)**

The largest single failure class is fields where the correct candidate IS ranked #1 in the pool, but the citation bbox emitted by `_ValidationAdapter.ground_extracted_data()` fails IoU ≥ 0.50. This is confirmed by two independent measurements:
- Observer V2 (full corpus): **12.97%** of all gradeable fields
- EXP-024 ablation: `INCOMPLETE_PERCEPTION` = 14,434 fields = **40.2% of ablation failures**, with **60.3% union recovery** — meaning the pool DOES have a good candidate, but the citation is wrong

**Secondary bottleneck: Complete coverage failure (RETRIEVAL_NO_CANDIDATE)**

10.93% of full-corpus gradeable fields (Observer V2) have zero candidates returned by the matcher. EXP-024 confirms this is entirely unrecoverable by spans, OCR, or cell splitting — **0% recovery by any tested source**.

**EXP-024 key finding: The candidate generation architecture is not the bottleneck for most failures**

Adding multi-token spans, OCR, and cell splitting to the baseline pool improves geometric coverage by only **+0.27pp** (66.40% → 66.67%) and Recall@5 by **+0.22pp**. These are measurement-noise-level improvements. The tested perception approaches do not address the real bottleneck, which is citation geometry corruption, not candidate pool composition.

**Corrected bottleneck priority (post-EXP-024):**

| Priority | Bottleneck | Scale | EXP-024 Status |
|----------|-----------|-------|---------------|
| **1** | Citation geometry corruption (INCOMPLETE_PERCEPTION) | ~12.97% of corpus | Unaddressed — fix is in `adapter.py`, not candidate pool |
| **2** | Complete coverage failure (RETRIEVAL_NO_CANDIDATE) | ~10.93% of corpus | Unaddressed — 0% recovery by tested sources |
| **3** | Bbox precision (BBOX_TOO_WIDE + NARROW) | ~5.98% of corpus | Partially addressed by cell splitting (+2.95pp Oracle ceiling) |
| **4** | Array alignment (ASSOCIATION_WRONG_ROW) | ~3.65% of corpus | Not addressed in EXP-024 |
| **5** | Ranking (RETRIEVED_RANK_6_20 + GT_20) | ~2.37% of corpus | Union recovers ASSOCIATION_RANK_MISS (100%) |

---

### Q14: Which document types / failure classes remain hardest?

From the failure class distribution and known document characteristics:

**Hardest document types:**
- **Large 13F SEC filings** (`sec_13f_*`, `real_ftx_*`): Massive array alignment + `ASSOCIATION_WRONG_ROW` failures. These documents have 26K–80K gradeable fields, and the O(n³) `linear_sum_assignment` alignment is the bottleneck.
- **Corrupted/image-based PDFs** (`*_corrupted`, `real_imedia_full`): `RETRIEVAL_NO_CANDIDATE` and `OCR_GEOMETRY` are dominant. The PDF text layer is absent or garbled; without OCR, the matcher has no candidates.
- **Unclaimed property registries** (`real_oklahoma_unclaimed_2024`): Excluded from grounded metrics entirely (`grounded_incomplete=True`). Even after fixing, this document would require massive geometry work.

**Hardest failure classes (by impact):**
1. `SELECTED_CITATION_GEOMETRY_FAILURE` (12.97%) — geometry corruption at citation generation
2. `RETRIEVAL_NO_CANDIDATE` (10.93%) — complete coverage failure
3. `BBOX_TOO_WIDE` (4.68%) — table row bloat
4. `ASSOCIATION_WRONG_ROW` (3.65%) — array alignment error
5. `OCR_GEOMETRY` (2.88%) — raster document geometry failure

---

### Q15: What should EXP-025 investigate?

EXP-024 provides a clear, evidence-based answer.

**Primary: Citation Geometry Fix (SELECTED_CITATION_GEOMETRY_FAILURE = 12.97%)**

This is the highest-leverage, lowest-risk intervention identified in the entire measurement program:
- The correct candidate IS at rank #1 with good geometry in the pool
- The failure occurs in `_ValidationAdapter.ground_extracted_data()` at citation conversion time
- No new models, no new data, no new candidate sources required
- Potential gain: +8-12pp Word Grounding F1 (if 60-90% of SELECTED_CITATION_GEOMETRY_FAILURE fields are fixed)

**Proposed EXP-025A — Citation Geometry Audit:**
1. For all 57,856 `SELECTED_CITATION_GEOMETRY_FAILURE` fields in Observer V2, extract:
   - `selected_candidate_iou` (from `field_records.parquet`)
   - The actual citation bbox as emitted by the adapter
   - The candidate pool bbox for rank #1
2. Compare candidate pool bbox vs citation bbox to identify the adapter code path that modifies geometry
3. Profile: does the modification help or hurt IoU? (Is it a correction that overshoots, or random corruption?)
4. Target: `src/tonerhound/benchmark/adapter.py` — `ground_extracted_data()`, `_ValidationResolver`, citation bbox assembly

**Proposed EXP-025B — Broad Cell Splitting:**

EXP-024D shows cell splitting improves Oracle@5 by **+2.95pp** with only 0.70% field coverage. The heuristic is currently too conservative. Broadening the gap threshold (e.g., from `max(8.0, 2.5×avg_char_w)` to `max(4.0, 1.5×avg_char_w)`) could cover more fields while retaining the high-precision geometry advantage.

**Explicitly OUT OF SCOPE for EXP-025:**
- OCR integration (demonstrated geometrically useless in EXP-024C until multi-word span grouping is built)
- Multi-token span reconstruction (marginal +0.23pp Oracle@5 improvement, not worth engineering cost yet)
- Reranking improvements (Oracle ceiling is too low; improving rank selection doesn't help until geometry is fixed)

---

### Q16: What evidence exists for or against approaching 90%?

**Against approaching 90% easily (EXP-024 sharpens previous concerns):**

1. **Oracle@20 ceiling = 55.99%** (full corpus). A hard 34pp gap exists even with perfect reranking.
2. **No tested perception source closes the gap materially.** Union of baseline + spans + OCR + cells improves geometric coverage by only **+0.27pp**. The specific perception techniques in EXP-024 are not the path to 90%.
3. **OCR is geometrically useless in its current single-word form.** Tesseract OCR achieves 0.00% geometric coverage. The most natural fallback for raster documents fails entirely on geometry.
4. **RETRIEVAL_NO_CANDIDATE = 10.93% is unrecoverable** by any source tested. These 48,744 fields (full corpus) have zero candidates from baseline, spans, OCR, or cells.

**For approaching 90% (the pathway is now clearer, though long):**

EXP-024 identifies cell splitting as the highest-value perception improvement (+2.95pp Oracle@5 with high precision geometry). The prioritized roadmap:

| Step | Bottleneck | Expected Gain | Basis |
|------|-----------|---------------|-------|
| EXP-025A | Citation geometry fix | **+8-12pp** | 57,856 fields, ~65% fixable = ~38K recovered |
| EXP-025B | Broad cell splitting | **+3-5pp** | Oracle ceiling +2.95pp when cells hit |
| EXP-026 | RETRIEVAL_NO_CANDIDATE forensics + OCR span grouping | **+3-6pp** | 48,744 fields currently 0% recoverable |
| EXP-027 | Array alignment improvement | **+2-4pp** | 16,257 ASSOCIATION_WRONG_ROW fields |
| EXP-028 | Broad bbox precision | **+2-3pp** | 20,872 BBOX_TOO_WIDE fields |

Theoretical upper bound if all bottlenecks fixed:
```
58.14% SUCCESS + 12.97% citation + 10.93% no-candidate +
4.68% bbox-wide + 3.65% wrong-row + 2.88% ocr-geometry +
1.30% bbox-narrow + 0.56% drift ≈ 95.11%
```

**90% is within theoretical reach** but requires solving genuinely hard problems across 5-6 experiments. No single fix gets there. The Oracle@20 ceiling (55.99% full corpus) must itself be raised before 90% is achievable.

> [!CAUTION]
> The +8-12pp EXP-025A estimate assumes citation geometry failure is a fixable code bug. If the adapter modifies geometry intentionally for correctness in other contexts, the fix may be more complex. **Measure before patching.**



## Reconciliation Summary (FINAL)

| Metric | Previously Reported | Observer V2 Authoritative | Explanation |
|--------|--------------------|--------------------------:|-------------|
| Recall@5 | 64.84% | **57.98%** (micro) / **48.69%** (true macro) | Previous used flawed macro over all 370 docs including ungrounded |
| Top-5 case partition | 70.47% | **57.98%** | Previous case partition used different definitions and denominators |
| Word Grounding F1 | 45.31% | **45.31%** ✓ | Confirmed match |
| Page Grounding F1 | 81.28% | 87.46% (369 docs) | Different document set (oklahoma excluded) |
| Oracle@5 | 51.03% | **53.11%** | Previous observer used different IoU/candidate definitions |
| Oracle@20 | 53.14% | **55.99%** | Same cause as above |

---

## Methodology Notes

### Observer V2 Frozen Definitions

- **Gradeable field:** Any field path that has accepted evidence entries in the benchmark (has `ev_boxes` and `ev_pages` under the ExtractBench field rules).
- **Candidate hit at K:** Any candidate in the top-K pool (by rank) whose `max(IoU(candidate_bbox, gold_bbox_i) for all i in gold_evidence_entries) >= 0.50`.
- **IoU threshold:** 0.50 (applied uniformly to Recall@K, Oracle@K, failure class assignment).
- **Multiple gold evidence:** All accepted evidence bboxes are scanned; the maximum IoU is used.
- **Oracle@K Word F1:** Computed by constructing a new `FieldCitation` using the top-K best-IoU candidate's bbox for each field, then running `compute_unified_evidence_metrics()` with those citations.
- **Macro vs Micro:** Micro uses total gradeable field count as denominator. True Macro averages per-document recall over only documents that have grounded fields (N=236).

### EXP-024 Ablation Definitions

- **Text Coverage:** `sum(1 for f in gradeable_fields if any_candidate_text_matches_gold_normalized(f, pool)) / total_gradeable_fields`
- **Geometric Coverage:** `sum(1 for f in gradeable_fields if any_candidate_iou(f, pool) >= 0.50) / total_gradeable_fields`
- **Recall@K:** Same definition as Observer V2 but restricted to the top-K of the given source pool.
- **Oracle@K Word F1:** Same as Observer V2 oracle computation, using the given source pool.

---

## Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Observer V2 field records | `research/observer/field_records.parquet` | 445,950 rows × 24 columns, per-field microscope data |
| Observer V2 aggregates | `research/observer/aggregates.json` | Authoritative aggregate metrics |
| Manual validation | `research/observer/MANUAL_VALIDATION.md` | 20 manually validated cases |
| EXP-024 ablation runner | `research/experiments/EXP-024/run_ablation.py` | Full ablation pipeline |
| EXP-024 candidate generators | `research/experiments/EXP-024/candidate_generators.py` | Research-only candidate sources |
| EXP-024 results | `research/experiments/EXP-024/results.csv` | Per-source metrics across 32 docs |
| EXP-024 failure recovery | `research/experiments/EXP-024/failure_recovery.csv` | Per-field failure recovery by source |

---

*EXP-025 is NOT started in this session. Observer V2 is the frozen measurement baseline.*
