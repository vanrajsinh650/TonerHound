# Missing Failure Patterns & Edge Cases Discovered

## 1. Accounting Parentheses on Negative Numbers (`ACCOUNTING_PARENTHESES_NEGATIVE`)
- **Observed Fact**: In IRS forms and corporate balance sheets, negative values written as `(1,234.56)` are frequently represented in ground truth as `-1234.56`.
- **Classification Today**: Grouped under `NORMALIZATION_MISMATCH`.
- **Evidence**: Affects 36 fields across tax and financial documents.

## 2. Table Column Bleed & Overlapping Cell Spans (`TABLE_COLUMN_BLEED_OVERLAP`)
- **Observed Fact**: In wide tables (SEC 13F and N-PORT), long company names extend across cell boundaries, causing bounding boxes to capture adjacent numeric columns.
- **Classification Today**: Grouped under `TOKEN_SLICING` or `BBOX_TOO_WIDE`.
- **Evidence**: Affects 22771 table cell fields.

## 3. Multi-Line Narrative Address Wrapping (`MULTI_LINE_NARRATIVE_WRAP`)
- **Observed Fact**: Entity names and addresses spanning 2-4 lines are partially captured on line 1, causing IoU to hover between 0.30 and 0.45.
- **Classification Today**: Grouped under `BBOX_TOO_NARROW`.
- **Evidence**: Affects 0 fields.
