# ExtractBench Unified Evidence Evaluator Mechanics

**Experiment**: EXP-032 (True Record-Level Oracle under the Official ExtractBench Evaluator)  
**Evaluator Implementation File**: `research/reference/ExtractBench/src/extract_bench/evaluation/metrics/extract/unified_evidence_metric.py`  
**Runner File**: `research/reference/ExtractBench/src/extract_bench/evaluation/runner.py`  
**Evaluator Integration File**: `research/reference/ExtractBench/src/extract_bench/evaluation/evaluators/extract.py`

---

## 1. Executive Summary

This document details the exact mechanics of the official ExtractBench evaluator used for scoring Word Grounding F1 and Page Grounding F1. All statements are derived directly from the source code of `unified_evidence_metric.py`, `runner.py`, and `extract.py`.

The primary insight for oracle construction is:
1. **Value-Only Row Alignment**: Hungarian table row alignment (`scipy.optimize.linear_sum_assignment`) operates **strictly over value match/mismatch costs**. Citation bounding boxes and pages **do not enter the cost matrix**.
2. **Fixed Row Assignment in Frozen Pipeline**: Because TonerHound operates on extraction outputs where `extracted_data == expected_output`, the value mismatch cost between `act_rows[i]` and `exp_rows[i]` is zero along the diagonal, fixing the row alignment to the exact identity mapping `exp_row_i <-> act_row_i`.
3. **Per-Path Evaluation**: Once rows are aligned, evidence evaluation is performed independently per cell path:
   `gt_path = f"{table}[{i}].{col}"` vs `pred_path = f"{table}[{i}].{col}"`.
4. **Hit Criteria**: A predicted cell receives grounded credit if and only if:
   $$\text{value\_matches} \land \exists (gp, gb) \in \text{ev\_boxes}[gt\_path], (pp, pb) \in \text{pred\_boxes}[pred\_path] : gp == pp \land \text{IoU}(gb, pb) \ge 0.50$$

---

## 2. Detailed Specifications (Sections A – H)

### A. Prediction Representation

* **Source**: `InferenceResult.output.extracted_data` and `InferenceResult.output.field_citations` (`extract.py` lines 270–275).
* **Data Structure**:
  * `extracted_data`: Nested dictionary of extracted fields, sub-objects, and lists of row objects.
  * `field_citations`: List of citation objects/dicts with fields:
    * `field_path`: String identifier in JSON-pointer/bracket notation (e.g. `"invoice_number"`, `"line_items[0].unit_price"`).
    * `page`: 1-indexed integer.
    * `bbox`: 4-element sequence `[x, y, w, h]` in normalized COCO coordinates $[0, 1]$.
* **Indexing**:
  * Evaluator indexes citations via `index_citations(field_citations)` (`unified_evidence_metric.py` lines 782–803).
  * Produces:
    * `pred_boxes: BoxIndex` $\to$ `dict[str, list[tuple[int, tuple[float, float, float, float]]]]`
    * `pred_pages: dict[str, set[int]]`
* **Hierarchy**: The evaluator recursively traverses schemas using `_score_node` (lines 717–736), decomposing arrays into rows and subfields (`_score_array`, lines 465–700) and scalar fields via `_score_scalar_cell` (lines 702–716).

---

### B. Gold Representation

* **Source**: `test_case.expected_output` and `test_case.get_extract_field_rules()`.
* **Field Rules**: `list[ExtractFieldTestRule]`, where each rule specifies:
  * `field_path`: Path string matching the expected output schema (e.g. `"line_items[0].description"`).
  * `evidence`: List of `ExtractEvidence` objects containing `page` (int), `bbox` (`[x, y, w, h]`), and `value` (alternate ground-truth value strings).
  * `normalizers`: Optional normalizer list (e.g. `case_insensitive`, `lenient_date`, `optional_terminal_punctuation`).
* **Indexing**:
  * Evaluator indexes gold rules via `build_rule_indexes(field_rules)` (`unified_evidence_metric.py` lines 747–780).
  * Produces:
    * `alt: dict[str, list[Any]]` — alternative acceptable values for the path.
    * `ev_boxes: BoxIndex` — gold `(page, bbox)` tuples per path.
    * `ev_pages: dict[str, set[int]]` — gold pages per path.
    * `normalizers: dict[str, tuple[str, ...]]` — active normalizers per path.

---

### C. Alignment Mechanics

* **Algorithm**: Hungarian assignment via `scipy.optimize.linear_sum_assignment` (`unified_evidence_metric.py` lines 616, 628, 659).
* **Nodes**:
  * Left nodes: Predicted/actual table rows (`act_rows`, index $j \in \{0, \dots, N-1\}$).
  * Right nodes: Ground-truth/expected table rows (`exp_rows`, index $i \in \{0, \dots, M-1\}$).
* **Cost Matrix**:
  * Cost matrix is of shape $(N, M)$ where $N = \text{len}(act\_rows)$, $M = \text{len}(exp\_rows)$.
  * Built using `mismatch_cost_matrix` or explicit configured cell mismatch calculation (`unified_evidence_metric.py` lines 602–615):
    $$\text{cost}[j, i] = \sum_{s \in \text{cost\_names}} \mathbb{I}(\text{configured\_cell\_match}(gt\_path, cv, avals[si], s) == \text{False})$$
  * Matching level: **Strictly record-to-record (row-to-row)**. Once Hungarian pairs actual row $j$ with expected row $i$, every subfield $s$ in row $i$ is evaluated strictly against subfield $s$ in row $j$.
  * **Crucial Property**: **Citation bounding boxes and pages are NEVER included in the alignment cost matrix.** Alignment is 100% driven by text and numeric values.

---

### D. Value Matching

* **Function**: `_value_match` (`unified_evidence_metric.py` lines 361–366).
* Checks if `actual` matches canonical expected value or any alternate in `alt[gt_path]` using `configured_cell_match`.
* Supports fuzzy thresholds (`DEFAULT_FUZZY_FIELD_THRESHOLDS`) and normalizers:
  * `optional_terminal_punctuation` (strips trailing `.` or `,`)
  * `case_insensitive` (case-folded whitespace-normalized match)
  * `null_equals_false` (`False` matches `None`)
  * `phone_digits` (last 10 digits match)
  * `lenient_date` (century expansion, split-year handling)
  * `punctuation_spacing`
* Can a wrong-row candidate match because text is identical?
  * Yes, if two rows have identical text values, Hungarian row cost for swapping them is identical (cost = 0).
  * However, when actual rows are in the exact same order as expected rows (as in our frozen benchmark), the diagonal cost is 0 for all rows, producing the canonical identity assignment.

---

### E. Evidence Matching (Grounding)

* **Function**: `_box_match` (`unified_evidence_metric.py` lines 368–373):
  ```python
  def _box_match(self, gt_path: str, pred_path: str) -> bool:
      gt_boxes = self._ev_boxes.get(gt_path)
      pred = self._pred_boxes.get(pred_path)
      if not gt_boxes or not pred:
          return False
      return any(gp == pp and iou_xywh(gb, pb) >= self._iou for gp, gb in gt_boxes for pp, pb in pred)
  ```
* **Timing**: Evaluated **after** alignment during the traversal of aligned pairs $(i, mj)$.
* **Threshold**: $\text{IoU} \ge 0.50$ (`self._iou = bbox_iou_threshold = 0.5`).
* **Multi-region Support**: Both gold evidence and predicted citations can have multiple boxes. An existential quantifier ($\exists$) is used: if ANY pair of gold and predicted boxes on the same page achieves $\text{IoU} \ge 0.50$, the cell is credited.
* **Page Match**: `_page_match` (lines 385–394) requires `not gt_pages.isdisjoint(pred_pages)`.

---

### F. Precision / Recall Formulation

* **Counters** (`_Counts`, lines 242–268):
  * `v_correct`: Aligned cells where value matches.
  * `expected`: Total expected leaf cells (value recall denominator).
  * `predicted`: Total predicted leaf cells (value precision denominator).
  * `g_correct`: Aligned cells where value matches AND bbox matches ($\text{IoU} \ge 0.50$).
  * `g_expected`: Gold cells whose evidence carries a bounding box (grounded recall denominator).
  * `g_claims`: Predicted cells with citation bboxes that are **aligned to a gold cell carrying an evidence bbox** (grounded precision denominator).
* **Ungradeable Claims Rule**:
  * Predicted citations on cells whose gold counterpart has no bbox (or extra predicted rows) do **NOT** increment `g_claims` (`_note_grounded_claim`, lines 379–383):
    ```python
    def _note_grounded_claim(self, gt_path: str, pred_path: str, c: _Counts) -> None:
        if not self._pred_boxes.get(pred_path):
            return
        if self._ev_boxes.get(gt_path):
            c.g_claims += 1
    ```
  * This is critical: emitting an incorrect citation on a bbox-bearing cell increments `g_claims` without incrementing `g_correct`, directly reducing Precision!
  * Emitting NO citation on an ungroundable cell avoids incrementing `g_claims`.
* **Formulae** (`_prf`, lines 806–810):
  * $\text{Precision} = \frac{g\_correct}{g\_claims}$
  * $\text{Recall} = \frac{g\_correct}{g\_expected}$
  * $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

---

### G. Tie-Breaking & Determinism

* `scipy.optimize.linear_sum_assignment` is deterministic and uses C-level Jonker-Volgenant / augmenting path algorithm.
* When diagonal costs are zero ($C_{ii} = 0$), SciPy's implementation maintains the initial column assignment, giving the identity matching.

---

### H. Aggregation across Documents

* `runner.py` lines 1684–1686:
  * For each document with `g_expected > 0`: `compute_unified_evidence_metrics` emits `extract_unified_grounded_f1`.
  * Across all evaluated documents:
    $$\text{Word Grounding F1} = \text{avg\_extract\_unified\_grounded\_f1} = \frac{1}{|D_{\text{bbox}}|} \sum_{d \in D_{\text{bbox}}} \text{extract\_unified\_grounded\_f1}(d)$$
  * This is an unweighted **macro-average** across all documents that carry gold bounding box annotations.
