# Competitive Analysis: Document Grounding Systems & Prior Art

## 1. Executive Summary & Objective

TonerHound is designed as an independent evidence-resolution and provenance layer.
To maintain scientific and architectural integrity, we must not claim novelty for capabilities that already exist. This report systematically audits the 8 principal document grounding systems and libraries:

1. **Anchorite** (`populationgenomics/anchorite`)
2. **Groundmark** (`populationgenomics/groundmark`) [Archived / Predecessor]
3. **LlamaExtract / LlamaParse** (`run-llama`)
4. **Reducto** (Reducto Deep Extract / Extract API)
5. **Docling** (IBM Granite Document AI / Deep Search)
6. **Google Document AI** (Form Parser / Custom Extractor)
7. **AWS Textract** (Analyze Document / Queries API)
8. **Unstructured** (Unstructured.io Partitioning & Extraction)

---

## 2. In-Depth System Evaluations

### 1. Anchorite (`populationgenomics/anchorite`)
* **1. What does it already solve?**
  Bidirectional spatial alignment between generated Markdown/quotes and physical PDF/OCR atoms. Bridges LLM Markdown transcribing to per-character bounding boxes via `pypdfium2` and OCR anchors.
* **2. How does it represent evidence?**
  `Anchor` (`text`, `page` 0-indexed, tuple of `BBox(top, left, bottom, right)` in normalized 0–1000 integer space), `SpanAnchor`.
* **3. How does it generate coordinates?**
  Extracts per-atom/char bounding boxes using `pypdfium2`. Uses `seq_smith` (C++/Cython accelerated Smith-Waterman sequence alignment) to align quote characters to the document-wide flat character stream, then reconstructs line boxes (`line_bboxes`).
* **4. How does it resolve duplicates?**
  Smith-Waterman picks the highest alignment score across the entire document. If two identical quotes exist, it has no label proximity or semantic context mechanism to choose between them—it will match whichever occurrence scores highest or appears first in the flat string.
* **5. Does it normalize values?**
  Only basic text normalization: lowercasing and collapsing non-alphanumeric characters to spaces (`_ALIGN_ALPHABET`). No date normalization, currency parsing, or numeric equivalence.
* **6. Does it support multiple regions?**
  Yes, returns a tuple of `BBox`es, one per visual line covered by the quote.
* **7. Does it identify ambiguity?**
  Has a uniqueness threshold and min alignment score (`_MIN_ALIGNMENT_SCORE = 15`), but cannot distinguish between two legitimately identical candidates with different semantic labels.
* **8. Does it verify evidence independently?**
  Yes, independent of the LLM generation step (operates on the PDF byte stream).
* **9. Does it use an LLM during grounding?**
  No. Grounding is 100% deterministic (PDF atom extraction + sequence alignment).
* **10. What does it fail on?**
  - Short scalar values (e.g. `$50.00`, numbers, dates), which fall below `_MIN_ALIGNMENT_SCORE = 15` or match redundantly across pages.
  - Value format shifts (e.g. `1450.0` extracted vs `$1,450.00` printed).
  - Distinguishing identical numbers across table cells or invoices without field context.

---

### 2. Groundmark (`populationgenomics/groundmark`) [Deprecated]
* **1. What does it already solve?**
  First-generation quote-to-bbox resolver for PDF pages, using Pydantic AI for Markdown extraction and `seq_smith` for quote localization.
* **2. How does it represent evidence?**
  `BBox(top, left, bottom, right)` in 0–1000 normalized space, mapped from character indices.
* **3. How does it generate coordinates?**
  `pypdfium2` character box extraction followed by global Smith-Waterman alignment.
* **4. How does it resolve duplicates?**
  Global Smith-Waterman single best alignment; no duplicate resolution.
* **5. Does it normalize values?**
  Simple alphanumeric lowercase character filtering.
* **6. Does it support multiple regions?**
  Yes, line grouping via character bounding box bounding hulls.
* **7. Does it identify ambiguity?**
  No. Returns best alignment score or raises if below threshold.
* **8. Does it verify evidence independently?**
  Yes, decoupled from LLM inference.
* **9. Does it use an LLM during grounding?**
  No during grounding (LLM used only for optional Markdown generation).
* **10. What does it fail on?**
  Merged tables, page-level boundary hops, short words, dirty OCR text, and lack of field-level context. Superseded by Anchorite to add OCR anchor fusion and chained alignment.

---

### 3. LlamaExtract / LlamaParse (`run-llama`)
* **1. What does it already solve?**
  End-to-end schema extraction from complex PDFs with built-in bounding box citations (`FieldCitation`) via agentic LLM loops and LlamaParse layout engine.
* **2. How does it represent evidence?**
  `FieldCitation`: `field_path`, `page` (1-indexed), `bbox: [x, y, w, h]` (COCO normalized `[0, 1]`), `reference_text`.
* **3. How does it generate coordinates?**
  LlamaParse parses document layout and OCR word boxes (`granular_bboxes`). During extraction, the LLM maps extracted field outputs to chunk/word references generated by LlamaParse.
* **4. How does it resolve duplicates?**
  The LLM prompt passes chunk/word metadata and asks the model to output the matching citation reference.
* **5. Does it normalize values?**
  Yes, the extraction model converts text to schema-typed values (e.g. strings to floats, dates to ISO 8601).
* **6. Does it support multiple regions?**
  Limited. Emits individual citations or bounding boxes, but often collapses multi-line entities into single oversized bounding boxes.
* **7. Does it identify ambiguity?**
  No. The LLM commits to a single citation or omits it. Does not emit `ambiguous` or `derived`.
* **8. Does it verify evidence independently?**
  No. The LLM generates both the value and selects the citation simultaneously. No secondary verification step checks whether the bounding box actually encloses the evidence text.
* **9. Does it use an LLM during grounding?**
  Yes, grounding is tightly coupled to LLM extraction.
* **10. What does it fail on?**
  - Caps at **46.43%** word grounding F1 on ExtractBench.
  - OCR bounding box drift and jitter.
  - LLM hallucination of citation references.
  - Severe latency (>110s to 580s per document).

---

### 4. Reducto (Reducto Deep Extract)
* **1. What does it already solve?**
  High-performance proprietary document ingestion and extraction pipeline using vision models and layout segmentation.
* **2. How does it represent evidence?**
  JSON citation tree mapping extracted keys to source page and bounding boxes.
* **3. How does it generate coordinates?**
  Proprietary vision-layout model detects document blocks, tables, and words, projecting extracted tokens back to visual layout masks.
* **4. How does it resolve duplicates?**
  Uses visual block hierarchy (field labels adjacent to values in spatial layout).
* **5. Does it normalize values?**
  Yes, produces typed JSON matching extraction schemas.
* **6. Does it support multiple regions?**
  Yes, through multi-block citation lists.
* **7. Does it identify ambiguity?**
  No. Always outputs a point estimate or omits citation.
* **8. Does it verify evidence independently?**
  Partially; internal layout matching checks token positions, but closed-source.
* **9. Does it use an LLM during grounding?**
  Yes, hybrid layout-VLM.
* **10. What does it fail on?**
  - **43.30%** word grounding F1 on ExtractBench.
  - Very expensive ($0.34/page).
  - High latency (206s–591s per document).
  - Closed-source proprietary API.

---

### 5. Docling (IBM Granite Document AI)
* **1. What does it already solve?**
  Open-source layout parsing, OCR integration, and table structure recovery (TEDS) into rich document trees (`DoclingDocument`).
* **2. How does it represent evidence?**
  Hierarchical node tree with `BoundingBox(l, t, r, b)` in original document page coordinates.
* **3. How does it generate coordinates?**
  Computer vision models (DocLayNet, TableFormer) detect bounding boxes for paragraphs, tables, headings, and cells.
* **4. How does it resolve duplicates?**
  Maintains exact structural location (table cell `(row_idx, col_idx)`, section hierarchy), but Docling is a *parser*, not a *schema resolver*. It does not take an external `{field: "tax", value: 50}` and locate it.
* **5. Does it normalize values?**
  Extracts raw text content inside detected layout elements.
* **6. Does it support multiple regions?**
  Yes, hierarchical elements with constituent word/line bounding boxes.
* **7. Does it identify ambiguity?**
  No (not an extraction resolver).
* **8. Does it verify evidence independently?**
  N/A.
* **9. Does it use an LLM during grounding?**
  No, uses layout detection CNNs/transformers.
* **10. What does it fail on?**
  Does not provide quote/value-to-box resolution for arbitrary extracted schemas. Requires a downstream resolver like TonerHound to map extracted values to Docling's bounding boxes.

---

### 6. Google Document AI (Form Parser / Custom Extractor)
* **1. What does it already solve?**
  Enterprise form/table parsing and schema entity extraction with native bounding poly geometry.
* **2. How does it represent evidence?**
  `Document.entities`: text, type, confidence, and `pageAnchor.pageRefs.boundingPoly` (normalized vertices `x, y` in `[0, 1]`).
* **3. How does it generate coordinates?**
  Combines proprietary Google OCR token coordinates with sequence-tagging models.
* **4. How does it resolve duplicates?**
  Trained on document layout priors (form fields, key-value pairs).
* **5. Does it normalize values?**
  Normalizes known types (dates, currencies, money amounts) in `normalizedValue`.
* **6. Does it support multiple regions?**
  Yes, `boundingPoly` can contain multiple segments or polygons.
* **7. Does it identify ambiguity?**
  Emits confidence scores (0.0–1.0) per entity, but does not explicitly flag semantic ambiguity or derived status.
* **8. Does it verify evidence independently?**
  No, extraction and grounding are a single joint ML model.
* **9. Does it use an LLM during grounding?**
  Uses multimodal transformers / LLMs in recent versions.
* **10. What does it fail on?**
  - Vendor lock-in (GCP only).
  - Rigid schema configurations.
  - Cannot ground extractions produced by external VLMs (e.g. OpenAI, Anthropic, Gemini, local models).
  - High error rates on arbitrary unstructured documents or non-standard tables.

---

### 7. AWS Textract (Analyze Document / Queries)
* **1. What does it already solve?**
  OCR, table extraction, form key-value pair detection, and query-based field extraction.
* **2. How does it represent evidence?**
  `Block` objects (`PAGE`, `LINE`, `WORD`, `KEY_VALUE_SET`, `CELL`) with `Geometry.BoundingBox` (`Left`, `Top`, `Width`, `Height` normalized 0–1).
* **3. How does it generate coordinates?**
  Proprietary AWS OCR engine.
* **4. How does it resolve duplicates?**
  Key-Value matching based on visual distance between detected `KEY` and `VALUE` blocks.
* **5. Does it normalize values?**
  Queries API attempts to extract answers; minimal normalization.
* **6. Does it support multiple regions?**
  A Line or Key-Value block points to child Word IDs.
* **7. Does it identify ambiguity?**
  Confidence scores on blocks; no formal ambiguity resolution.
* **8. Does it verify evidence independently?**
  No.
* **9. Does it use an LLM during grounding?**
  Textract Queries uses NLP models; Core Textract is OCR/heuristic.
* **10. What does it fail on?**
  - Closed AWS ecosystem.
  - Rigid key-value pairing fails when label is remote or non-standard.
  - Cannot verify arbitrary downstream LLM structured outputs.

---

### 8. Unstructured (Unstructured.io)
* **1. What does it already solve?**
  Document partitioning (PDF, DOCX, HTML) into semantic chunks (`Title`, `NarrativeText`, `Table`, `ListItem`).
* **2. How does it represent evidence?**
  `Element`: `text`, `type`, `metadata.coordinates.points` in pixel space with `system="PixelSpace"`.
* **3. How does it generate coordinates?**
  Via underlying engines: `pdfminer`, `tesseract`, or `yolox` layout models.
* **4. How does it resolve duplicates?**
  Does not do value resolution. Simply parses elements in reading order.
* **5. Does it normalize values?**
  No.
* **6. Does it support multiple regions?**
  Coordinates are element-level bounding boxes.
* **7. Does it identify ambiguity?**
  No.
* **8. Does it verify evidence independently?**
  N/A.
* **9. Does it use an LLM during grounding?**
  No.
* **10. What does it fail on?**
  Unstructured is a document chunker/partitioner, not a field-level grounding resolver.

---

## 3. What Remains Genuinely Difficult After Existing Solutions Are Applied?

Our analysis reveals a sharp divide:
- **Parsers** (Docling, Unstructured, pypdfium2) extract layout and coordinates, but know nothing about what fields an application extracted.
- **Quote Aligners** (Anchorite, Groundmark) match long verbatim text strings to PDF characters, but break completely on short numbers, normalized dates, currencies, and duplicate table values.
- **End-to-End Extraction APIs** (LlamaExtract, Reducto, Google DocAI) generate bounding boxes within their black-box pipelines, but plateau at **43%–46% Word Grounding F1** because they conflate semantic extraction with physical spatial grounding and do not perform independent verification.

### The Remaining Hard Problems (TonerHound's Exact Focus):

1. **Short Scalar Disambiguation**:
   Resolving whether `$50.00` belongs to `subtotal`, `tax`, or `shipping` using spatial label proximity (`field_context`), reading order, and table row/column associations.
2. **Reversible Value Normalization**:
   Mapping an extracted `1450.0` or `2026-03-15` back to the original physical glyphs (`$1,450.00`, `March 15, 2026`) without losing the coordinate bounding boxes of each underlying character.
3. **Multi-line Segment Grouping**:
   Reconstructing exact line-by-line bounding boxes for wrapped text (addresses, terms) rather than emitting a noisy, single rectangular bounding hull with low IoU.
4. **Independent Evidence Verification & Calibration**:
   Rejecting hallucinations, declaring `derived` when no literal evidence exists, and declaring `ambiguous` when two occurrences cannot be distinguished, achieving 100% precision on accepted groundings.
5. **Provider Independence**:
   Grounding extractions from ANY model (OpenAI GPT-6, Gemini 3.8 Flash, Claude Code, Qwen, local open weights) against the raw PDF bytes or OCR geometry.
