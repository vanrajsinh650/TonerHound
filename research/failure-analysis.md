# Failure Analysis Framework: Taxonomy, Diagnostics & Error Categorization

## 1. Grounding Failure Taxonomy

To methodically outperform existing benchmarks, every error observed during benchmarking is categorized according to this taxonomy:

| Failure Code | Category | Root Cause | Example | TonerHound Remedy |
|:---|:---|:---|:---|:---|
| `ERR_OCCURRENCE` | Duplicate Selection | Multiple identical values; wrong occurrence selected | Subtotal `$50.00` vs Tax `$50.00` | Label context scoring (`field_context` proximity) and line-level relationship |
| `ERR_NORM_CURRENCY` | Normalization | Value extracted as raw float; text has currency symbol/commas | Extracted: `1450.0`, Doc: `USD 1,450.00` | Reversible currency & numeric normalization mapping |
| `ERR_NORM_DATE` | Normalization | Date formatted in ISO 8601; text in natural language | Extracted: `2026-03-15`, Doc: `15 March 2026` | Multi-format date parser with char-offset preservation |
| `ERR_MULTILINE_IOU` | Geometry | Multi-line text represented as a giant rectangular hull | Wrapped 4-line postal address | Multi-box visual line clustering (`regions: list[BBox]`) |
| `ERR_COORD_FRAME` | Coordinate | Mismatch between PDF points, pixel coordinates, and normalized 0-1 COCO | Box inverted, scaled wrong, or 0-indexed page | Strict `CoordinateFrame` type checks and 1-indexed page conversion |
| `ERR_TABLE_ROW` | Structural | Repeated identical numeric cells across rows/columns | Quantity `1` in row 3 matched to row 1 | Table grid awareness (row & column header bounding constraints) |
| `ERR_OCR_DROP` | Noise | OCR misread or dropped characters | Doc: `501(c)(3)`, OCR: `501(c) 3` | Sequence alignment with gap penalties & phonetic tolerance |
| `ERR_DERIVED_FORCED`| Policy | Model forced a box onto a calculated value | Total `$30` calculated from items `$10` + `$20` | Output `status="derived", bbox=None` |
| `ERR_AMBIGUOUS_FORCED`| Policy | Model guessed between two indistinguishable candidates | Two identical rows with identical labels | Output `status="ambiguous", bbox=None` |

---

## 2. Quantitative Diagnostic Metrics

Beyond benchmark F1, TonerHound tracks:
- **False Grounding Rate (FGR)**: $\text{False Positive Groundings} / \text{Total Grounding Claims}$
  - Target: < 5% (Zero tolerance for silent false boxes).
- **Ambiguity Precision**: Percentage of `status="ambiguous"` decisions that were genuinely structurally indistinguishable.
- **Derived Precision**: Percentage of `status="derived"` decisions that were genuinely unprinted calculations.
- **Localization Margin**: The mean IoU on accepted groundings (Target: > 0.85).

---

## 3. Failure Logging Format

When running benchmark suites, failed predictions are exported to `research/failures/` as structured JSON:
```json
{
  "test_id": "short/invoice_42",
  "field_path": "tax_amount",
  "extracted_value": 50.0,
  "predicted_citation": {
    "page": 1,
    "bbox": [0.75, 0.45, 0.08, 0.02]
  },
  "ground_truth_evidence": [
    {
      "page": 1,
      "bbox": [0.75, 0.55, 0.08, 0.02],
      "value": "$50.00"
    }
  ],
  "measured_iou": 0.0,
  "failure_code": "ERR_OCCURRENCE",
  "diagnosis": "Selected Subtotal row box at y=0.45 instead of Tax row box at y=0.55."
}
```
