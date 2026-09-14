# EXP-005: Technique Decision Matrix & ROI Ranking

**Project**: TonerHound  
**Document ID**: `EXP-005-technique-matrix`  
**Purpose**: Prioritized ranking and evaluation of 45 grounding, retrieval, alignment, and verification techniques based on technical ROI, latency impact, and feasibility for EXP-005.

---

## ROI Methodology

Each technique is evaluated across:
- **Expected Accuracy Gain ($\Delta$ F1)**: Quantitative estimate based on EXP-004 failure mode distribution.
- **Runtime Impact**: Computational latency added (or saved) per document query.
- **Implementation Complexity**: Engineering effort and regression risk (Low / Medium / High).
- **ROI Tier**:
  - **Tier 1 (Immediate High-ROI)**: Core architectural upgrades that eliminate known bottlenecks and yield major accuracy/speed gains with low complexity.
  - **Tier 2 (High-Value Refinement)**: Structural, geometric, and alignment additions for specific failure modes.
  - **Tier 3 (Selective / Tail)**: Specialized algorithms for rare failure modes or edge cases.
  - **Tier 4 (Discarded / Overkill)**: Unsuitable, excessively heavy, non-deterministic, or negative-ROI approaches.

---

## Ranked Technique Decision Matrix

| Rank | Technique | Problem Solved | Expected Accuracy Benefit | Runtime Impact | Implementation Complexity | Dependencies | Deterministic? | Works on Digital PDF? | Works on OCR? | Works on Tables? | Works on Long Docs? | Recommended? | Priority / Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Inverted Token & Value Index** | Linear scan $O(Q \cdot N)$ on every query | +6.0% Recall | **-80% Runtime** (Huge Speedup) | Low | None (Built-in) | Yes | Yes | Yes | Yes | Yes | **YES (Tier 1)** | Fixes 3-hour long document timeouts and unlocks sublinear candidate retrieval. |
| **2** | **Cached Normalization Property** | Recomputing `normalize_unicode_and_case` millions of times | +0.0% (Fixes bug) | **-75% Runtime** (17.6x faster) | Low | None | Yes | Yes | Yes | Yes | Yes | **YES (Tier 1)** | 1-line fix removing massive CPU hotspot in context scoring. |
| **3** | **Relaxed Page Pruning on Long Docs** | Current matcher aborts numeric/date/fuzzy if pages > 10 & no hint | +12.0% Recall | Modest with inverted index | Low | None | Yes | Yes | Yes | Yes | Yes | **YES (Tier 1)** | Directly caused the 34.82% Long Document F1 failure in EXP-004. |
| **4** | **Page-Partitioned Monotonic Table Anchors** | Rows jumping, duplicate numbers in table grids | +10.0% F1 | Low | Medium | SciPy | Yes | Yes | Yes | Yes | Yes | **YES (Tier 1)** | Solves 80% of tabular repeated-value ambiguities. |
| **5** | **Character N-Gram Inverted Index** | Corrupted text, OCR typos, compound tokens | +5.0% Recall | Negligible | Low | None | Yes | Yes | Yes | Yes | Yes | **YES (Tier 1)** | Lifts Candidate Recall@20 above 95% on noisy documents. |
| **6** | **Candidate-Local Smith-Waterman Alignment** | Sub-word tokens, punctuation attachment, multi-line wrapping | +4.0% F1 | Very Low (< 1ms) | Medium | RapidFuzz / NumPy | Yes | Yes | Yes | Yes | Yes | **YES (Tier 2)** | Delivers exact glyph boundary boxes matching Anchorite precision. |
| **7** | **BM25 Page Prior Routing** | Long documents with identical values across pages | +8.0% Page F1 | Low (~5ms) | Low | rank_bm25 / math | Yes | Yes | Yes | Yes | Yes | **YES (Tier 2)** | Direct fix for "Wrong Page / Sparse Form" failure mode. |
| **8** | **Multi-Line Span Grouping (Multi-Region)** | Inflated single-box union over multi-line addresses | +3.5% F1 | Negligible | Low | None | Yes | Yes | Yes | Yes | Yes | **YES (Tier 2)** | Eliminates Low-IoU bounding box failures on wrapped text. |
| **9** | **Calibrated Score-Margin Abstention** | Confident false groundings on ambiguous candidates | +15.0% Precision | None ($O(1)$) | Low | None | Yes | Yes | Yes | Yes | Yes | **YES (Tier 2)** | Guarantees Zero Silent False Groundings mandate. |
| **10** | **Hungarian Bipartite Row Matching** | Combinatorial race conditions in dense record extraction | +4.5% F1 | Low for $N \le 100$ | Medium | `scipy.optimize` | Yes | Yes | Yes | Yes | Yes | **YES (Tier 2)** | Mirrors ExtractBench evaluator's exact Hungarian row alignment. |
| **11** | **Hierarchical Document Index (Page $\to$ Block $\to$ Line)** | Unstructured flat line arrays losing 2D layout | +2.5% F1 | Saves runtime | Medium | None | Yes | Yes | Yes | Yes | Yes | **YES (Tier 2)** | Clean structural abstraction for block and table clustering. |
| **12** | **OCR-Specific Weighted Edit Distance** | Character confusion (`0` $\leftrightarrow$ `O`, `1` $\leftrightarrow$ `l`) | +2.5% Recall | Low | Low | RapidFuzz | Yes | Yes | Yes | Yes | Yes | **YES (Tier 2)** | Solves numeric and date mismatches on scanned documents. |
| **13** | **Reading-Order Gutter Detection** | Multi-column layout line interleaving | +2.0% F1 | Low (< 2ms) | Medium | None | Yes | Yes | Yes | Yes | Yes | **YES (Tier 2)** | Fixes SEC 13F and presentation deck column scrambling. |
| **14** | **Sub-Word Character BBox Interpolation** | Degenerate character boxes in malformed PDF fonts | +1.0% F1 | Negligible | Low | None | Yes | Yes | Yes | Yes | Yes | **YES (Tier 3)** | Fallback safety net for PDF font descriptor edge cases. |
| **15** | **Lazy Targeted OCR Execution** | Slow OCR on pages that already have digital text | +0.0% (Speedup) | **-95% OCR Time** | Low | pytesseract | Yes | No | Yes | Yes | Yes | **YES (Tier 3)** | Skips OCR invocation on digital PDF pages. |
| **16** | **Controlled Multiprocess Evaluation** | CPU underutilization during benchmark suite runs | +0.0% (Speedup) | **-80% Total Time** | Medium | `multiprocessing` | Yes | Yes | Yes | Yes | Yes | **YES (Tier 3)** | Enables rapid 10-minute turnaround for local benchmarking. |
| **17** | **Semantic Cross-Encoder Reranking** | Distinguishing subtle semantic label synonyms | +1.5% F1 | **+50-200ms per query** | High | PyTorch, HuggingFace | No | Yes | Yes | Yes | No | **NO (Tier 4)** | Too heavy, slow, non-deterministic; lexical context is sufficient. |
| **18** | **ANN Vector Embeddings (HNSW/FAISS)** | Approximate vector similarity | +0.5% Recall | High memory & build | High | FAISS | No | Yes | Yes | No | No | **NO (Tier 4)** | Approximate recall loss is incompatible with exact physical coordinates. |
| **19** | **Visual VLM Grounding (ColPali)** | Image-level bounding box prediction | -5.0% IoU | **+2-5s per page** | High | VLM, GPU | No | Yes | Yes | Yes | No | **NO (Tier 4)** | Hallucinates bounding boxes; IoU typically fails $< 0.50$. |
| **20** | **Needleman-Wunsch Global Alignment** | End-to-end string matching | -2.0% F1 | Low | Low | None | Yes | Yes | Yes | Yes | Yes | **NO (Tier 4)** | Penalizes sub-string matches; Smith-Waterman is strictly superior. |

---

## Action Plan & Staged Execution

1. **Sprint 1: Performance Hotspots & Retrieval Overhaul (Ranks 1, 2, 3, 5)**
   - Apply the cached `norm_text` fix in `resolver.py`.
   - Remove the artificial 10-page truncation in `matcher.py` by replacing linear scans with an Inverted Token and N-Gram Index.
   - Target: Candidate Recall@20 lifts from **78.97% to > 95%**; runtime drops by > 70%.

2. **Sprint 2: Structural Table Reasoning & Monotonic Record Anchoring (Ranks 4, 10, 13)**
   - Implement row-level multi-field binding and monotonic row progression for array records.
   - Target: Word Grounding F1 on tabular documents lifts from **53.68% to > 75%**.

3. **Sprint 3: Cascading Character Alignment & Multi-Region Box Refinement (Ranks 6, 8, 12, 14)**
   - Integrate candidate-local Smith-Waterman alignment with character coordinate mapping.
   - Implement multi-region line grouping for wrapped text.
   - Target: Coordinate IoU $\ge 0.50$ rate lifts above 95%; Word Precision approaches 90%.

4. **Sprint 4: Calibrated Abstention & Verification Tuning (Rank 9)**
   - Fit score-margin and spatial consistency thresholds on the dev split.
   - Target: Zero Silent False Groundings; Word Precision $> 92%$.
