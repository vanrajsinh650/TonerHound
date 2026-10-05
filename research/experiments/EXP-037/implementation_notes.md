# EXP-037: Technical Architecture & Implementation Notes

## 1. Phase A Repository Audit Findings

### 1. Where native PDF text is extracted
- **Engine**: Native digital characters and words are extracted using `pypdfium2` in `tonerhound.document.index._extract_tokens_from_page`.
- **Character to Word Clustering**: Character bounding boxes (`textpage.get_charbox(i)`) are buffered and flushed on whitespace/newlines into `DocumentToken` objects with normalized coordinates `[0, 1]`.
- **In Hybrid Backend**: `LiteParse` parses the digital layer when available (`_extract_tokens_and_blocks_from_liteparse_page`), returning structured word tokens and layout block hints (tables, paragraphs, headings).

### 2. How empty-text pages are represented
- Empty-text pages (scanned documents, raster image pages, un-OCR'd photocopies) produce `tokens = []` and `lines = []` on the `DocumentPage` object.
- The `DocumentIndex` stores an empty token list for that page number.

### 3. Where OCR can be inserted
- OCR is hooked directly inside `DocumentIndex.from_pdf` (lines 224–233) and `HybridDocumentIndex.from_pdf` (lines 591–617).
- In EXP-037, the standalone `DeterministicOCRRouter` (`research/experiments/EXP-037/ocr_router.py`) encapsulates page-selective routing before index construction.

### 4. How OCR text enters the existing index
- `_extract_tokens_from_ocr` renders the PDF page to a bitmap (`scale = 200 / 72 ≈ 2.78`), calls `pytesseract.image_to_data`, and parses word boxes into `DocumentToken` objects.
- Tokens are clustered into visual lines via `_cluster_tokens_into_lines`.
- The resulting `DocumentPage` is passed to `DocumentIndex.__init__`, which automatically builds all inverted indexes (`_token_index`, `_stem_token_index`, `_numeric_index`, `_date_index`, `_page_normalized_text`).

### 5. Whether OCR already existed but was disabled
- **Yes**. In `DocumentIndex.from_pdf`, `enable_ocr` defaults to `False` (`actual_enable_ocr = False if enable_ocr is None else enable_ocr`).
- In the canonical baseline audit (`canonical_370_v1`), the Failure Microscope explicitly ran with `enable_ocr=False` to execute the audit rapidly in 145 seconds.

### 6. How OCR results are converted into searchable tokens
- `_parse_ocr_data` applies `repair_ocr_text` (resolving common OCR glyph confusions) and constructs `DocumentToken` instances with normalized COCO bounding boxes.
- Stems, numeric values (`parse_numeric_value`), and dates (`parse_date_value`) are indexed identically to native digital tokens.

### 7. Whether OCR bounding box coordinates are preserved
- **Yes**. Bounding boxes are stored as `BBox(x, y, width, height, page)` on every `DocumentToken`.
- Coordinates are normalized to $[0, 1]$ relative to page dimensions (`width`, `height`).

### 8. Whether grounding can use those coordinates
- **Yes**. `EvidenceResolver` and `EvidenceMatcher` retrieve candidates directly from `DocumentIndex`.
- Candidates retain token bounding boxes and visual line geometry, which `ExtractBenchAdapter._apply_geometry_enhancements` uses to emit official COCO `[x, y, w, h]` predictions.

### 9. Whether OCR must be page-level or document-level
- **Strictly Page-Level**. Scanned exhibits and mixed-content filings often have clean digital cover pages or forms alongside scanned schedules.
- Running document-wide OCR introduces OCR character noise onto clean digital pages, degrading precision. Page-selective routing applies OCR ONLY where native digital text is absent.

### 10. How to avoid unnecessary OCR on digital pages
- A digital token threshold (`min_digital_tokens = 25`) gates OCR invocation:
  $$\text{IF } \text{len}(\text{native\_tokens}) \ge 25 \implies \text{PAGE\_MODE} = \text{'native'} \quad (\text{SKIP OCR})$$
  $$\text{IF } \text{len}(\text{native\_tokens}) < 25 \implies \text{PAGE\_MODE} = \text{'ocr'} \quad (\text{INVOKE OCR})$$

---

## 2. Deterministic Routing Architecture

```
                 Incoming PDF Page
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   pypdfium2                         LiteParse
_extract_tokens_from_page       Complexity Check
        │                                 │
        └────────────────┬────────────────┘
                         ▼
        Native Tokens >= 25 & Not Garbled?
              │                    │
             YES                   NO
              │                    │
              ▼                    ▼
     [SKIP OCR: Native]     [INVOKE OCR: Tesseract]
     Cluster into Lines     Bitmap Render (scale 2.78)
              │             repair_ocr_text
              │             PSM 11 Sparse Fallback
              │                    │
              └──────────┬─────────┘
                         ▼
             Construct DocumentPage
                         ▼
           Populate DocumentIndex (v3)
      - Exact Token Inverted Index
      - Stem Token Inverted Index
      - Numeric Canonical Index
      - Date Window Index
                         ▼
              EvidenceResolver Query
```

---

## 3. Operational Safety & Reproducibility Constraints

1. **Deterministic Stack**:
   - Zero LLMs, Zero VLMs, Zero neural networks, Zero embeddings, Zero external cloud APIs.
   - Fixed Tesseract binary invocation with deterministic parameters.
2. **Laptop Safety**:
   - Single-threaded math execution enforced via environment variables (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).
   - Explicit garbage collection (`gc.collect()`) after processing each document.
3. **Cache Isolation**:
   - OCR index cache files are isolated with SHA-256 source hash and router version key:
     `{file_hash}_{enable_ocr}_{ocr_scale}_{min_digital_tokens}_{INDEX_VERSION}.pkl`
