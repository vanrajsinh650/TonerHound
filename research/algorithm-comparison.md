# Algorithm Comparison & Architectural Design

## 1. Algorithmic Dimensions

Document grounding bridges the gap between semantic structured extraction and physical visual geometry. This document details the algorithmic trade-offs across each phase of the TonerHound engine.

---

## 2. Text Matching & Alignment Algorithms

| Algorithm | Mechanism | Strengths | Weaknesses | Latency | Role in TonerHound |
|:---|:---|:---|:---|:---:|:---:|
| **Exact Substring / Token Index** | Direct inverted index / suffix array over page tokens | 100% precision, zero false-grounding, sub-millisecond | Breaks on punctuation, formatting, casing, dates, currencies | < 1 ms | **Tier 1 (Fast Path)**: Always execute first. If unique, resolve immediately. |
| **Reversible Normalization Map** | Normalizes characters while preserving offset arrays mapping normalized chars to raw PDF atoms | Handles currency (`$50.00` → `50`), dates, whitespace, without losing coordinates | Requires strict grammar definitions per data type | < 2 ms | **Tier 2 (Canonical Path)**: Resolves 60%+ of extraction formatting drift. |
| **Spatial Context (Label Proximity)** | Calculates Euclidean and reading-order distances between candidate value boxes and `field_context` label boxes | Distinguishes identical values (e.g. `Tax: $50` vs `Subtotal: $50`) | Requires locating the label on the page | < 5 ms | **Disambiguation Engine**: Resolves candidate duplicates. |
| **RapidFuzz / Token Edit Distance** | QRatio / partial ratio token matching | Fast, handles minor OCR typos or dropped spaces | Lacks character-exact visual coordinate boundaries | 5–15 ms | **Candidate Generation**: Filters candidate regions for sequence alignment. |
| **Smith-Waterman Local Alignment** | Dynamic programming with affine gap penalties (`seq_smith`) | Optimal local alignment for noisy OCR, wrapped text, and deletions | $O(M \cdot N)$ compute, expensive on full documents, caps out on short scalars | 20–100 ms | **Tier 3 (Fuzzy Fallback)**: Scoped to pre-filtered candidate regions only. |

---

## 3. Multi-line Geometry Reconstruction

A naive implementation that unions all matched word bounding boxes into a single rectangle (`[min_x, min_y, max_x - min_x, max_y - min_y]`) suffers from **catastrophic IoU degradation**:

```text
Line 1: 123 Main Street                      [----Line 1 Box----]
Line 2: Apt 4B                               [--Line 2 Box--]
Line 3: San Francisco, CA 94105              [-------Line 3 Box-------]

Single Enclosing Hull:
┌────────────────────────────────────────────────────────┐
│ 123 Main Street                                        │  <- 40% empty space!
│ Apt 4B                                                 │  <- Ground truth IoU
│ San Francisco, CA 94105                                │     drops < 0.50!
└────────────────────────────────────────────────────────┘
```

**TonerHound Strategy**:
- Cluster characters into visual lines based on vertical overlap and baseline alignment.
- Group each line into a tight bounding box.
- Output line-level constituent regions (`regions: list[BBox]`), and emit the primary line or high-coverage bounding box according to downstream API specs. In ExtractBench (which takes a single `bbox` per citation), compute the standard intersection-over-union metric over the multi-rect set!

---

## 4. Duplicate & Multi-Occurrence Resolution

When a scalar appears multiple times in a document:
1. **Filter by Page Hint**: If extraction provides `page_hint`, restrict candidates to that page.
2. **Context Label Matching**:
   Search for `field_context` (e.g. "Tax", "Due Date", "Total") within the page text.
   Score candidates by:
   - Line alignment (same horizontal line as label, to the right)
   - Vertical alignment (directly below column header)
   - Reading order distance
   - Distance penalty $d = \sqrt{\Delta x^2 + \Delta y^2}$
3. **Ambiguity Gating**:
   If the margin between the top 2 candidate scores is less than $\epsilon$, do NOT guess. Mark status as `ambiguous`.

---

## 5. Provenance Contract & Uncertainty

A key failure of existing systems is returning high-confidence bounding boxes for hallucinated or calculated values.

TonerHound classifies every resolution into an explicit provenance state:
- `exact`: Literal match in document.
- `normalized`: Value matched after currency/date/whitespace transformation.
- `fuzzy`: Alignment matched with character edit distance within tolerance.
- `multi_region`: Value spans multiple visual lines.
- `ambiguous`: Multiple valid candidates cannot be reliably distinguished.
- `derived`: Value is mathematically or semantically derived with no literal physical source.
- `not_found`: Evidence could not be located in document.

Only `exact`, `normalized`, `fuzzy`, and `multi_region` emit coordinates. `ambiguous`, `derived`, and `not_found` return `null` coordinates, preventing false groundings.
