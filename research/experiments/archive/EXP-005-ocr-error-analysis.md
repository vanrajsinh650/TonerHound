# EXP-005 Part 3: Corrupted OCR Diagnostic Error Analysis

**Experiment ID**: `EXP-005-ocr-error-analysis`  
**Date**: 2026-09-15  
**Target Documents**:
1. `short/real_clinton_property_25_11073_corrupted` (9 pages, 246 test rules, baseline Word F1: 18.48%)
2. `medium/real_bbb_service_list_corrupted` (17 pages, 1502 test rules, baseline Word F1: 1.61%)
3. `long/real_ftx_full_corrupted` (114 pages, 75543 test rules, baseline Word F1: 0.26%)

---

## 1. Baseline Performance & Diagnostic Discovery

All three target corrupted documents exhibit near-total grounding collapse despite high page-level precision:

| Document | Pages | Citations | Word Precision | Word Recall | **Word F1** | Page Precision | Page Recall | **Page F1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `short/real_clinton_property_25_11073_corrupted` | 9 | 66 | 41.46% | 11.89% | **18.48%** | 98.21% | 34.81% | **51.40%** |
| `medium/real_bbb_service_list_corrupted` | 17 | 289 | 25.64% | 0.83% | **1.61%** | 100.00% | 19.84% | **33.10%** |
| `long/real_ftx_full_corrupted` | 114 | 9155 | 2.25% | 0.14% | **0.26%** | 98.83% | 26.50% | **41.80%** |

### Key Diagnostic Discovery:
Page Precision is **98.2% to 100.0%**. The system reliably identifies the correct physical page when it resolves a citation. However, **Word Grounding Recall is between 0.14% and 11.89%**. 
Candidate generation and exact text lookup fail completely because corrupted OCR introduces non-deterministic character mutations, merged tokens, noise punctuation, and page-segmentation dropouts.

---

## 2. Systematic Error Decomposition (Categories A – O)

Below is the verified failure taxonomy with real examples extracted directly from the three target corrupted documents:

### Category A: Character Substitution
Tesseract OCR confuses visually similar glyphs under compression noise, low contrast, and blur:
- **`0` ↔ `O`**: `2093` read as `2003` (BBB service list party 0 address); `PO BOX 5300` read as `POBOX5300` with `O`/`0` ambiguity.
- **`I` ↔ `1` ↔ `l` / `J`**: `1015 TANZANIA DRIVE` read as `1015 TANZANJADRIVE` (`I` substituted by `J`).
- **`1` ↔ `4`**: Creditor number `2.1` read as `24` on Clinton page 4 line y=0.195.
- **`0` ↔ `t` / `6`**: `$19.90` read as `$19.9t` on Clinton page 6 line y=0.723.
- **`@` ↔ `®`**: `YISSACHAR@ALTO-INV.COM` read as `YISSACHAR®ALTO` on BBB service list page 1.
- **`N` ↔ `R`, `F` ↔ `I`, `L` ↔ `R`**: `NAME ON FILE` read as `[RAMCONIRE` on FTX page 2 line y=0.0976.
- **`i` ↔ `_`, `t` ↔ `(`**: `Unliquidated` read as `Unl_lqmda(_ed` on Clinton page 4 line y=0.250.

### Category B: Dropped Characters
Internal characters or boundary characters are lost during thresholding:
- `TARGET EXTERMINATING INC.` ↔ `TARGETEXTERMINATING` (space dropped).
- `Basis for the Claim:` ↔ `Basis for the laim:` (leading `C` dropped on Clinton page 4 line y=0.279).
- `2.1` ↔ `21` / `3.12` ↔ `312` (decimal point dropped).

### Category C: Inserted Characters
Scanner noise, specks, and vertical rules are recognized as punctuation or brackets:
- `CONSOLIDATED EDISON` ↔ `*“CONSOLIDATED` (leading `*“` inserted).
- `ATTN: GENERAL COUNSEL` ↔ `[ATTN: GENERAL COUNSEL` (leading bracket `[` inserted).
- `CLAYMONT DE 19703` ↔ `[CLAYMONT DE 19703` (leading bracket `[` inserted).
- `NAME ON FILE` ↔ `[RAMCONIRE` (leading bracket `[` inserted).

### Category D: Merged Words (Space Elision)
Lack of whitespace between words causes Tesseract to emit single tokens:
- `PO BOX 5300` ↔ `POBOX5300`
- `TARGET EXTERMINATING` ↔ `TARGETEXTERMINATING`
- `BIG MIKE'S A/C` ↔ `BIGMIKE'SAC`
- `TANZANIA DRIVE` ↔ `TANZANJADRIVE`
- `name and mailing` ↔ `name-and.mailing`

### Category E: Split Words (Spurious Whitespace)
Tokens are broken apart by noise:
- `ALTO-INV.COM` ↔ `ALTO` and `INV.COM` (separated into distinct tokens).
- `25-11073` ↔ `25-` and `11073`.
- `07/08/25` ↔ `07/08/25` and `00:38: 02`.

### Category F: Punctuation Corruption
Leading, trailing, or internal punctuation alters the token string:
- `VERIZON` ↔ `.VERIZON`
- `SYNCO` ↔ `SYNCO.`
- `SYSTEMS` ↔ `SYSTEMS;`
- `AND HEARINGS` ↔ `AND‘HEARINGS`

### Category G: Decimal & Number Corruption
Periods in numbers are either lost, merged, or shifted:
- `3.11` ↔ `311.` (period shifted to end)
- `3.12` ↔ `312` (period lost)
- `2.1` ↔ `24` (period dropped, 1 mutated to 4)
- `$121.50` ↔ `$1215` (period and trailing 0 lost)

### Category H: Currency Corruption
Currency signs or formatted numbers get corrupted or grouped with neighboring text:
- `$19.90` ↔ `$19.9t`
- `$121.50` ↔ `$1215`
- `$6,867.62` ↔ `Total $6,867.62 amount`

### Category I: Date Corruption
Slashes and numbers in dates mutate:
- `06/12/23` ↔ `06/12123` (second slash dropped and merged with digit 1).
- `07/08/25` ↔ `07/08/25 00:38: 02`.

### Category J: Line-Break & Multi-Line Splitting
Multi-line fields (e.g. addresses, descriptions) span 2 to 4 vertical lines:
- `ATTN: GENERAL COUNSEL\n2093 PHILADELPHIA PIKE # 1971\nCLAYMONT DE 19703` (BBB service list party 0).
- Multi-line creditor mailing addresses on Clinton property pages 4–7.

### Category K: Column-Order & Horizontal Interleaving
In `DocumentIndex._cluster_tokens_into_lines`, tokens at approximately the same vertical height ($y \pm 0.005$) are grouped into a single line. In multi-column tables, tokens from column 1 (`ALBANY, NY 12205`), column 2 (`NYS DEPARTMENT OF TAX AND FINANCE`), column 3 (`POBOX5300`), and column 4 (`$6,867.62`) are all merged into a single line at $y = 0.195$. Exact multi-token sequence search fails because words from adjacent columns are interleaved.

### Category L: Completely Missing OCR (Page Segmentation Failure)
In Tesseract's default Page Segmentation Mode (PSM 3), tabular pages without clear paragraph headings are misclassified as graphics or empty margins:
- **`real_bbb_service_list_corrupted.pdf` Page 1**:
  - PSM 3 extracts only **72 words** (missing 85% of party listings!).
  - PSM 11 extracts **470 words** (all 250 party listings restored!).
- **`real_ftx_full_corrupted.pdf` Page 2**:
  - PSM 3 extracts only **133 words** (missing 83% of creditor listings!).
  - PSM 11 extracts **785 words** (full 70-row table restored!).

### Category M: Correct OCR Exists but Exact Retrieval Fails
Because tokenization splits on whitespace, exact token subsequence search `_find_token_subsequence` searches for `['TARGET', 'EXTERMINATING']`, but the index contains `['TARGETEXTERMINATING']`. Without character n-gram or substring indexing, exact retrieval returns 0 candidates.

### Category N: Candidate Exists but Ranking Selects Wrong Occurrence
In Clinton property, duplicate amounts like `$6,867.62` appear in column 1 (Priority amount) and column 2 (Total amount of claim). Without row-level spatial constraints, the priority claim resolves to the total claim column.

### Category O: Geometry Exists but BBox is Wrong
Aligning to the entire visual line produces a bounding box spanning all columns ($w \approx 0.88$), causing ExtractBench IoU to fail against ground truth boxes ($w \approx 0.15$).

---

## 3. Targeted Solution Plan for EXP-005 Part 3

1. **Page-Segmentation Dual OCR (Solving Category L)**:
   - When a page contains $< 150$ tokens in tabular/corrupted context, execute a targeted sparse-text pass (PSM 11) or fallback to ensure complete token coverage.
2. **OCR-Aware Normalization with Position Tracking (Solving Categories A, C, F, G, H)**:
   - Strip leading/trailing noise symbols (`*`, `“`, `’`, `[`, `]`, `_`).
   - Canonical numeric and alphanumeric normalization that maps corrupted digits/letters (`O` $\to$ `0`, `l`/`I` $\to$ `1`, `S` $\to$ `5`).
3. **Character N-gram Inverted Index (Solving Categories B, D, M)**:
   - Index character 3-grams and 4-grams for fast token retrieval. When exact lookup fails, query the n-gram index to retrieve candidates with merged words (`TARGETEXTERMINATING` matching `TARGET` and `EXTERMINATING`).
4. **Fuzzy Sequence Alignment & Bounding Box Slicing (Solving Categories D, J, O)**:
   - For retrieved candidates, compute character-level alignment (Smith-Waterman or bounded edit distance) and slice the exact character bounding boxes, avoiding giant line-level boxes.
5. **Structural & Contextual Verification (Solving Categories K, N)**:
   - Check local vertical proximity to field headers or sibling fields to select the correct column/row occurrence.
