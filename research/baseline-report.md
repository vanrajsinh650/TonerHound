# Baseline Report: ExtractBench Grounding & Public State of the Art

## 1. Executive Summary & Current Leaderboard Verification

As mandated by Section 2 and Section 7 of the TonerHound specification, the live repository `run-llama/ExtractBench` was cloned and inspected directly at commit `94ceac1`.

The current public ExtractBench leaderboard (`leaderboard.csv`) confirms that **LlamaExtract Agentic Plus** remains the undisputed #1 public system for word-level document grounding:

| Rank | Provider | Category | Overall Value F1 | Word Grounding F1 | Page Grounding F1 | Short Word F1 | Medium Word F1 | Long Word F1 | Cost / Page |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LlamaExtract Agentic Plus** | LlamaExtract | **95.59%** | **46.43%** | **84.92%** | 43.74% | 54.01% | 54.67% | $0.0811 |
| 2 | **LlamaExtract Agentic** | LlamaExtract | 89.55% | **44.14%** | 66.12% | 42.30% | 50.47% | 45.68% | $0.0312 |
| 3 | **Reducto Deep Extract** | Specialized APIs | 90.44% | **43.30%** | 71.71% | 42.84% | 45.57% | 41.13% | $0.3444 |
| 4 | **LlamaExtract Cost-Effective** | LlamaExtract | 86.78% | **40.43%** | 64.15% | 40.20% | 42.30% | 36.67% | $0.0100 |
| 5 | **Extend (Max Context)** | Specialized APIs | 88.62% | **25.20%** | 49.04% | 33.93% | 0.21% | 0.02% | $0.1000 |
| 6 | **Extend Extract** | Specialized APIs | 85.72% | **15.96%** | 53.58% | 21.13% | 1.03% | 0.01% | $0.0680 |
| 7 | **Datalab (Accurate + Balanced)**| Specialized APIs | 85.70% | **2.02%** | 48.50% | 2.67% | 0.24% | 0.00% | $0.0350 |
| — | **Codex (GPT-5.5)** | Coding Agents | 93.57% | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% | $0.2783 |
| — | **OpenAI GPT-6 Astra** | Commercial VLM | 91.91% | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% | $0.1109 |
| — | **Google Gemini 3.8 Flash** | Commercial VLM | 80.71% | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% | $0.0043 |
| — | **Qwen3.6 35B** | OSS VLM | 88.11% | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% | — |
| — | **Claude Code (Opus 4.8)** | Coding Agents | 87.09% | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% | $0.1617 |

### Crucial Findings from Leaderboard Inspection:
1. **The 46.43% Word Grounding F1 is still the authoritative, current public baseline.**
2. **Every standard frontier VLM (Codex, GPT-6 Astra, Gemini 3.8 Flash, Qwen 35B, Claude Code, etc.) scores exactly 0.00% Word Grounding and 0.00% Page Grounding.** They extract values effectively (Overall Value F1 80%–93%) but emit zero physical bounding box citations.
3. Even among specialized document extraction systems that *do* produce bounding boxes, there is a massive drop from value accuracy to word-level grounding:
   - LlamaExtract Agentic Plus: **95.59% Value F1** vs **46.43% Grounding F1** (-49.16% gap)
   - Reducto Deep Extract: **90.44% Value F1** vs **43.30% Grounding F1** (-47.14% gap)
   - Extend Extract: **85.72% Value F1** vs **15.96% Grounding F1** (-69.76% gap)
   - Datalab: **85.70% Value F1** vs **2.02% Grounding F1** (-83.68% gap)

This quantitatively validates the core thesis of TonerHound: **Current document systems extract values reasonably well, but fail over 50% of the time to pinpoint the physical word-level evidence.**

---

## 2. Evaluation Engine Deep Dive (`unified_evidence_metric.py`)

ExtractBench evaluates grounding using `compute_unified_evidence_metrics` located in `src/extract_bench/evaluation/metrics/extract/unified_evidence_metric.py`.

### A. Coordinate System & Geometry Specifications
- **Format**: Normalized COCO `[x, y, width, height]` floats in the range `[0.0, 1.0]`.
- **Page Indexing**: **1-indexed** integers (`page=1` is the first page).
- **IoU Calculation**:
  ```python
  def iou_xywh(a: BBox, b: BBox) -> float:
      ax, ay, aw, ah = a
      bx, by, bw, bh = b
      ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
      iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
      inter = ix * iy
      union = aw * ah + bw * bh - inter
      if union <= 0:
          return 0.0
      return min(1.0, inter / union)
  ```
- **Threshold**: Strict IoU threshold of **0.50** (`bbox_iou_threshold=0.5`).

### B. The Three Nested Metrics
1. **`extract_unified_value_f1` (Overall Value F1)**:
   - Precision: $V_{correct} / \text{predicted\_cells}$
   - Recall: $V_{correct} / \text{expected\_cells}$
   - Solved via Hungarian matching (`linear_sum_assignment`) across array rows and recursive object paths.
2. **`extract_unified_page_f1` (Page Grounding F1)**:
   - Condition: Value matches **AND** predicted citation page is in the GT evidence page set.
   - Precision: $P_{correct} / P_{claims}$
   - Recall: $P_{correct} / P_{expected}$
3. **`extract_unified_grounded_f1` (Word Grounding F1)**:
   - Condition: Value matches **AND** predicted citation has a box on the same page with `iou_xywh >= 0.50`.
   - Precision: $G_{correct} / G_{claims}$
   - Recall: $G_{correct} / G_{expected}$
   - $G_{claims}$: Predicted citation-bbox claims on cells aligned to a bbox-bearing GT cell. Claims on cells without GT boxes are excluded (prevents punishing models on unannotated fields).
   - If a document has zero bbox-bearing ground truth cells, no `extract_unified_grounded_f1` metric is emitted (does not drag the dataset average down with 0.0).

---

## 3. The Extraction Output Contract

The official contract expected by ExtractBench evaluators:
```python
from pydantic import BaseModel, Field

class FieldCitation(BaseModel):
    field_path: str               # Dotted path, e.g. "invoice_number" or "items[0].price"
    page: int                     # 1-indexed page integer
    bbox: list[float] | None      # Normalized [x, y, w, h] in [0.0, 1.0]
    polygon: list[list[float]] | None = None
    reference_text: str | None = None
    confidence: float | None = None

class ExtractOutput(BaseModel):
    task_type: str = "extract"
    example_id: str
    pipeline_name: str
    extracted_data: dict[str, Any] | list[dict[str, Any]]
    field_citations: list[FieldCitation]
```

---

## 4. Why Native Pipelines Fail at Word-Level Grounding

Analyzing the providers in ExtractBench reveals why existing systems cap at 46.43%:

1. **OCR / Layout Engine Box Drift**:
   LlamaExtract relies on LlamaParse's OCR bounding boxes (`granular_bboxes`). OCR bounding boxes frequently segment words unevenly, chop hyphenated tokens, or merge adjacent columns, causing IoU with ground truth to drop below 0.50 even when the correct word was sighted.
2. **Short Scalar Ambiguity**:
   Numbers like `50.00`, dates like `2024-01-01`, and short codes like `USD` appear multiple times across invoices, statements, and tables. Naive search attaches the first occurrence, resulting in wrong-page or wrong-row citations.
3. **Multi-line Wrapping**:
   Long strings (addresses, terms, descriptions) wrap across multiple lines. Many pipelines emit either a single bounding box enclosing all lines (which contains large amounts of white space, destroying IoU) or only emit the bounding box of the first token.
4. **Value Normalization Mismatch**:
   When an LLM extracts `1450.0` from text reading `$1,450.00`, naive verbatim string search fails completely.
5. **No Independent Disambiguation / Evidence Verification**:
   Native systems output bounding boxes as an afterthought of parsing or LLM generation. They do not run a dedicated resolution pipeline that considers label-value proximity, table grid alignment, and reading order.

---

## 5. TonerHound Target Milestone Progression

```text
[Baseline: LlamaExtract Agentic Plus] -> 46.43% Word Grounding F1
[Milestone 1: Baseline Victory]       -> 47.00%+ Word Grounding F1
[Milestone 2: Strong Improvement]     -> 60.00%+ Word Grounding F1
[Milestone 3: Major Result]           -> 75.00%+ Word Grounding F1
[Milestone 4: State-of-the-Art]       -> 85.00%+ Word Grounding F1
```
