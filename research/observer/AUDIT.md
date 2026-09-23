# Comprehensive Architectural & Evaluator Audit: TonerHound & ExtractBench

**Author**: TonerHound Research Engineering  
**Date**: September 23, 2026  
**Stack Baseline**: Frozen Production Stack (`EXP-011` + `EXP-012` + `EXP-013` + `EXP-015` + `EXP-017R` + `EXP-018` at commit `26e8692`)  
**Scope**: Full end-to-end evidence pipeline, candidate lifecycle, geometry engines, normalization, benchmark adapter, and ExtractBench evaluation mechanics.

---

## 1. System Overview & Relevant Files

The TonerHound system locates physical bounding-box evidence for arbitrary structured data extracted from documents. The benchmark adapter runs TonerHound against the official ExtractBench test suite.

### Primary Production Files
- **`src/tonerhound/document/index.py`**: Base document indexing, inverted token indexes, numeric index, date index, 3-gram line index, PDFium page loading, and line clustering.
- **`src/tonerhound/document/hybrid_index.py`**: `HybridDocumentIndex` combining `LiteParse` digital layout parsing with PDFium + Tesseract OCR fallback for scanned/degraded documents.
- **`src/tonerhound/models/types.py`**: Core data contracts: `DocumentToken`, `VisualLine`, `DocumentPage`, `ExtractionInput`, `ResolutionResult`, `ProvenanceStatus`.
- **`src/tonerhound/geometry/coordinates.py`**: Normalized $[0, 1]$ coordinate system `BBox(x, y, width, height, page)`, intersection, union, and IoU implementations.
- **`src/tonerhound/geometry/character_span.py`**: Character-level sub-token bounding box refinement (`reconstruct_safe_character_span`, EXP-015).
- **`src/tonerhound/geometry/same_line_recovery.py`**: Contiguous same-line token expansion bounded by column corridors (`extend_same_line_tokens`, EXP-017 / EXP-017R).
- **`src/tonerhound/geometry/dot_leader_trimming.py`**: Dot-leader and trailing whitespace trimming (`trim_dot_leaders`, EXP-018).
- **`src/tonerhound/geometry/structure_classifier.py`**: Heuristic table and layout structure classification (`classify_table_structure`).
- **`src/tonerhound/normalization/normalizers.py`**: Canonical string cleaning, Unicode NFKC, numeric value parsing, date parsing, checkbox detection.
- **`src/tonerhound/matching/matcher.py`**: `EvidenceMatcher`: Exact phrase matching, normalized numeric matching, normalized date matching, boolean/checkbox matching, and rapidfuzz fuzzy alignment.
- **`src/tonerhound/matching/candidate_recovery.py`**: `CandidateRecoveryEngine` (EXP-013): Fallback candidate recovery for hyphenated words, fragmented tokens, and numeric formatting variations.
- **`src/tonerhound/resolution/resolver.py`**: `EvidenceResolver`: Multi-tier candidate collection, context scoring, y-hint proximity filtering, structural reranking dispatch.
- **`src/tonerhound/resolution/reranker.py`**: `StructuralReranker` (EXP-011): Spatial scoring incorporating column rails, row corridors, sibling anchors, sequence monotonicity, and page priors.
- **`src/tonerhound/resolution/flat_form_reranker.py`**: `FlatFormLabelReranker` (EXP-012): Label-guided reranking for structured tax and government flat forms (Form 1040, W-2, etc.).
- **`src/tonerhound/benchmark/adapter.py`**: `ExtractBenchAdapter`: Converts ExtractBench extraction payloads into official `FieldCitation` lists, handles table corridor analysis, row-pitch estimation, monotonic page filtering, and geometry enhancement passes.
- **`benchmarks/run_final_370_benchmark.py`**: Orchestrates `_ValidationAdapter` and `_ValidationResolver` for the frozen production stack across all 370 ExtractBench documents.

### Evaluator & Reference Files
- **`research/reference/ExtractBench/src/extract_bench/evaluation/metrics/extract/unified_evidence_metric.py`**: Official ExtractBench `compute_unified_evidence_metrics`: Hungarian row alignment, value matching, page grounding, and bounding-box IoU gating.
- **`research/reference/ExtractBench/src/extract_bench/evaluation/runner.py`**: Official `EvaluationRunner`: Result finding, task pooling, and unweighted macro metric aggregation (`_aggregate_metrics`).

---

## 2. Execution Flow & Architecture

```
                 PDF Document
                      │
                      ▼
            [DocumentIndex.from_pdf]
         (Hybrid LiteParse + OCR fallback)
                      │
                      ▼
   [Pages -> Lines -> Tokens -> Inverted Indexes]
                      │
                      ▼
            [ExtractBenchAdapter]
   Receives extracted data (scalars + array records)
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
Scalar Fields                   Array Tables
       │                             │
       │                   [Pass 1: Anchor Discovery]
       │                   (Row pitches, page mapping,
       │                    monotonic LNDS filtering)
       │                             │
       │                   [Pass 2: Row Extraction]
       │                   (y_hint, corridors, siblings)
       └──────────────┬──────────────┘
                      ▼
            [EvidenceResolver.resolve]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[collect_candidates]        [Candidate Ranking]
• Tier 1a: Exact text       • Context word proximity
• Tier 1-bool: Checkbox     • Y-hint distance penalty
• Tier 1b: Float numeric    • StructuralReranker (EXP-011)
• Tier 1c: String exact     • FlatFormLabelReranker (EXP-012)
• Tier 1d: Date index                │
• Tier 2: String numeric             ▼
• Tier 3: Fuzzy alignment    [Selected Top Candidate]
• Recovery (EXP-013)                 │
                                     ▼
                      [Geometry Enhancement Passes]
                      • Character-span trim (EXP-015)
                      • Same-line expansion (EXP-017R)
                      • Dot-leader trimming (EXP-018)
                                     │
                                     ▼
                           [Official FieldCitation]
                                     │
                                     ▼
                      [ExtractBench ExtractEvaluator]
```

---

## 3. Candidate Lifecycle & Representation

### Representation
Every candidate instance is represented as a `MatchCandidate` (`src/tonerhound/matching/matcher.py`):
```python
@dataclass(frozen=True, slots=True)
class MatchCandidate:
    page: int
    bbox: BBox
    tokens: tuple[DocumentToken, ...]
    matched_text: str
    match_type: str  # "exact", "normalized_number", "normalized_date", "fuzzy", "recovery"
    raw_similarity: float
    line_index: int = 0
```
- `bbox`: Normalized $[0, 1]$ coordinates $(x, y, w, h)$ representing the union bounding box of all participating tokens.
- `page`: 1-indexed page number.
- `tokens`: Immutable tuple of atomic `DocumentToken` objects.

### Candidate Generation Pipeline
1. **Tier 1a (Exact Phrase)**: Exact phrase search using token inverted index (`_token_index`) and line token clusters.
2. **Tier 1-bool (Checkbox)**: Detects boolean visual indicators (checked boxes, cross marks, Yes/No radio pairs) via `detect_checkbox_state`.
3. **Tier 1b (Normalized Numeric)**: Numeric float lookup in `_numeric_index` within floating-point tolerance ($10^{-6}$), handling currency symbols, commas, and negative signs.
4. **Tier 1c (Stringified Exact)**: Direct string match against raw stringified values.
5. **Tier 1d (Normalized Date)**: ISO date match (`YYYY-MM-DD`) via `_date_index`.
6. **Tier 2 (String Numeric Fallback)**: Attempts numeric parsing on strings that contain formatted numbers (e.g. `"$1,250.00"`).
7. **Tier 3 (Fuzzy String Alignment)**: Substring fuzzy search via `rapidfuzz` (threshold $\ge 0.82$).
8. **Page Hint Relaxation**: If a provided `page_hint` yields zero candidates, the search is relaxed across all pages.
9. **EXP-013 Recovery Engine**: `CandidateRecoveryEngine` recovers split tokens, punctuation anomalies, and hyphenated line breaks.

### Candidate Deduplication
Within `EvidenceResolver.collect_candidates`, candidates generated across tiers or recovery engines on the same page are deduplicated if their mutual IoU exceeds $0.70$:
```python
if not any(rc.page == c.page and rc.bbox.iou(c.bbox) >= 0.70 for c in candidates):
    candidates.append(rc)
```

### Candidate Ranking & Association
When multiple candidates exist:
1. **Base Context Scoring (`_score_candidates_with_context`)**:
   - Matches words in `field_context` to nearby tokens on the page within a vertical corridor ($\Delta y \le 0.05$).
   - Adds string similarity weight and exact match bonuses.
   - Penalizes candidates deviating from `y_hint` by $(1.0 - \min(1.0, 20.0 \times \Delta y))$.
2. **EXP-011 Structural Reranking (`StructuralReranker.rank_candidates`)**:
   - Column corridor alignment ($w_{\text{col}} = 8.0$): reward candidates falling within the column rail $[x_{\min}, x_{\max}]$.
   - Row corridor alignment ($w_{\text{row}} = 10.0$): reward candidates within horizontal band $[y_{\min}, y_{\max}]$.
   - Sibling anchor affinity ($w_{\text{sib}} = 8.0$): reward alignment with sibling cells in the same record.
   - Sequence monotonicity ($w_{\text{seq}} = 5.0$): penalize vertical sequence reversals.
   - Page confidence ($w_{\text{page}} = 15.0$): heavy prior when target page is determined.
3. **EXP-012 Flat-Form Label Reranking (`FlatFormLabelReranker.rerank`)**:
   - Activated for recognized flat forms (IRS 1040, W-2, etc.).
   - Matches official form line labels and direction vectors to select the unambiguous cell.

---

## 4. Bounding Box Generation & Precision Refinement

The initial candidate bounding box is the geometric union of the participant tokens (`union_bbox_list`).
Before generating the final citation, `ExtractBenchAdapter` passes the selected bounding box through three geometry refinement stages:
1. **EXP-015 Safe Character Span (`reconstruct_safe_character_span`)**:
   - Computes sub-token character widths using font metrics and character ratios.
   - Trims extraneous leading/trailing characters when the target value is a sub-span of a token.
2. **EXP-017 / EXP-017R Same-Line Recovery (`extend_same_line_tokens`)**:
   - Scans adjacent tokens on the same visual line.
   - Expands candidate bounding boxes to absorb fragmented multi-token values (e.g. `"$"` + `"1,200"` + `".50"`).
   - Constrained by `max_right_boundary` to prevent encroaching into adjacent table columns.
3. **EXP-018 Dot-Leader Trimming (`trim_dot_leaders`)**:
   - Detects repeated periods, underscores, or dashes used as tabular leaders (`"Total ......... $500"`).
   - Crops the bounding box to isolate the actual semantic value.

---

## 5. Gold Evidence Loading & Benchmark Field Filtering

ExtractBench test cases are loaded via `load_test_case(pdf_path)`:
- `expected_output`: Ground truth JSON structure.
- `field_rules`: List of `ExtractFieldTestRule` specifying:
  - `target_field`: Field path (e.g. `creditors[0].name`).
  - `evidence`: List of accepted evidence dictionaries, each specifying `page`, `bbox` $[x, y, w, h]$, `quote`, and `value`.
  - `normalizers`: List of allowed deterministic normalizers.

### Indexing Rules in `build_rule_indexes`
ExtractBench builds four primary ground-truth indexes:
1. `alt_values[path]`: List of all accepted string/number values declared across evidence entries for `path`.
2. `ev_boxes[path]`: List of accepted bounding boxes `(page, (x, y, w, h))` for `path`.
3. `ev_pages[path]`: Set of accepted page numbers for `path`.
4. `normalizers[path]`: Set of active normalizers for `path`.

### Document & Field Gradeability Filtering
In ExtractBench's `compute_unified_evidence_metrics`:
- **Grounded Expected Denominator (`g_expected`)**: A field is gradeable for bounding-box grounding if and only if `len(ev_boxes[path]) > 0`.
- **Page Expected Denominator (`p_expected`)**: A field is gradeable for page grounding if and only if `len(ev_pages[path]) > 0`.
- **Ungradeable Documents**: Documents with `g_expected == 0` (134 of the 370 documents carry only value/page annotations without bounding boxes) emit **NO** `extract_unified_grounded_*` metrics.
- **Dataset Average**: Official dataset `word_grounding_f1` is computed as an unweighted macro-average strictly across the 236 documents that emitted `extract_unified_grounded_*`.

---

## 6. ExtractBench Evaluator Flow & Metric Computation

Official evaluation is executed via `ExtractEvaluator` invoking `compute_unified_evidence_metrics`.

### Array & Record Alignment (Hungarian Matching)
1. Extracted and ground-truth arrays are converted into row dictionaries (`as_rows`).
2. Mismatch cost matrix is constructed:
   $$\text{Cost}(j, i) = \sum_{s \in \text{subfields}} \mathbb{I}(\text{Cell}(j, s) \neq \text{Cell}(i, s))$$
3. Optimal bijective row assignment is solved using scipy's `linear_sum_assignment`:
   $$\min_{\pi} \sum_{i} \text{Cost}(\pi(i), i)$$
4. Predicted cells on matched rows are aligned to ground truth cells:
   $$\text{gt\_path} = \text{array}[i].s \longleftrightarrow \text{pred\_path} = \text{array}[\pi(i)].s$$

### Bounding Box IoU Computation
IoU between predicted box $a = (a_x, a_y, a_w, a_h)$ and gold box $b = (b_x, b_y, b_w, b_h)$ is computed via `iou_xywh`:
$$\text{inter}_x = \max(0, \min(a_x + a_w, b_x + b_w) - \max(a_x, b_x))$$
$$\text{inter}_y = \max(0, \min(a_y + a_h, b_y + b_h) - \max(a_y, b_y))$$
$$\text{inter} = \text{inter}_x \times \text{inter}_y$$
$$\text{union} = a_w a_h + b_w b_h - \text{inter}$$
$$\text{IoU} = \frac{\text{inter}}{\text{union}}$$
A predicted citation passes bounding-box grounding if:
$$\text{pred.page} = \text{gold.page} \quad \text{and} \quad \text{IoU}(\text{pred.bbox}, \text{gold.bbox}) \ge 0.50$$

### Metric Formulae

#### 1. Value Metrics
- $\text{TP}_v = c.v\_correct$ (value matches canonical or alternate evidence)
- $\text{Precision}_v = \frac{\text{TP}_v}{c.predicted}$
- $\text{Recall}_v = \frac{\text{TP}_v}{c.expected}$
- $\text{Value F1} = \frac{2 \cdot P_v \cdot R_v}{P_v + R_v}$

#### 2. Page Grounding Metrics
- $\text{TP}_p = c.p\_correct$ (value matches AND predicted page $\in$ gold pages)
- $\text{Precision}_p = \frac{\text{TP}_p}{c.p\_claims}$
- $\text{Recall}_p = \frac{\text{TP}_p}{c.p\_expected}$
- $\text{Page F1} = \frac{2 \cdot P_p \cdot R_p}{P_p + R_p}$

#### 3. Word Grounding Metrics
- $\text{TP}_w = c.g\_correct$ (value matches AND predicted page $==$ gold page AND $\text{IoU} \ge 0.50$)
- $\text{Precision}_w = \frac{\text{TP}_w}{c.g\_claims}$ (claims on cells whose GT carries a bbox)
- $\text{Recall}_w = \frac{\text{TP}_w}{c.g\_expected}$ (GT cells carrying a bbox)
- $\text{Word F1} = \frac{2 \cdot P_w \cdot R_w}{P_w + R_w}$

#### 4. Macro Benchmark Aggregation
For $N = 236$ grounded documents:
$$\text{Benchmark Word F1} = \frac{1}{N} \sum_{d=1}^{N} \text{Word F1}_d = 45.31\%$$
$$\text{Benchmark Page F1} = \frac{1}{N} \sum_{d=1}^{N} \text{Page F1}_d = 81.28\%$$

---

## 7. Summary of Audited Bottlenecks & Critical Requirements

1. **Measurement Integrity**: In previous instrumentation (`microscope_runner.py`), `recall_at_k` was averaged over all 370 documents with 134 ungrounded documents hardcoded to 1.0, and candidates were re-ranked with a naive context scorer without structural reranking. This caused artificial divergence between reported recall (64.84%) and case partition sums (70.47%).
2. **True Candidate Hit Definition**: Candidate Hit @ K must be evaluated strictly using the actual candidate pool generated by the pipeline, where:
   $$\max_{g \in \text{accepted\_gold}} \text{IoU}(\text{candidate.bbox}, g.\text{bbox}) \ge 0.50 \quad \text{with candidate.page} == g.\text{page}$$
3. **No Production Mutation**: Production code in `src/tonerhound` remains strictly frozen during the measurement and reconciliation phase.
