# TonerHound 🐕🔎

**TonerHound** is a high-precision, model-agnostic **document evidence grounding and provenance engine**.

AI systems are often good at extracting the correct value from a document, but much worse at proving exactly where that value came from. Leading Vision-Language Models (such as OpenAI Codex, GPT-4o/Astra, Gemini 1.5/2.0 Flash, and Qwen2.5-VL) score **0.00% Word Grounding F1** on the official [ExtractBench](https://github.com/run-llama/ExtractBench) benchmark because they lack pixel-level token spatial indexation.

TonerHound acts as an independent evidence-resolution layer: it takes extracted structured data from any LLM, VLM, or heuristic parser, indexes the physical document character and token geometry via `pypdfium2`, resolves canonical normalizations (dates, currencies, numbers), and disambiguates candidate regions using 2D spatial context and strict ambiguity gating.

---

## 🏆 ExtractBench Official Benchmark Results

Evaluated directly with ExtractBench's official evaluator (`compute_unified_evidence_metrics`):

| System | Word Grounding F1 | Word Precision | Word Recall | Page Grounding F1 | Value F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ungrounded VLM Baseline (Codex / Flash / Astra)** | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| **LlamaExtract Agentic Plus (#1 Public Leader)** | 46.43% | — | — | 84.92% | 84.77% |
| **TonerHound (`real_wyo_Goshen_2024`)** | **88.51%** | **88.97%** | **88.05%** | **99.65%** | **100.00%** |
| **TonerHound (`real_pueblo_oct_2025`)** | **53.81%** | **55.93%** | **51.85%** | **96.27%** | **100.00%** |
| **TonerHound (`real_sm0801_eco_full`)** | **50.93%** | **53.53%** | **48.57%** | **94.34%** | **100.00%** |
| **TonerHound (Digital PDF Average)** | **64.42%** | **66.14%** | **62.82%** | **96.75%** | **100.00%** |

*TonerHound outperforms the #1 public ExtractBench leader by **+17.99 percentage points** on digital document benchmarks, reaching up to **88.51% Word Grounding F1** on election records with **99.65% Page Grounding F1** and **100% Page Precision**.*

---

## 📐 Architecture & Key Innovations

```
                          ┌──────────────────────────┐
                          │   Extracted JSON Data    │
                          │   (from any LLM / VLM)   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
┌──────────────────┐      ┌──────────────────────────┐
│   Source PDF     │ ───► │  TonerHound DocumentIndex│
│  (Vector/Raster) │      │  (Character & Line BBox) │
└──────────────────┘      └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   EvidenceResolver       │
                          │ 1. Quote Exact Match     │
                          │ 2. Canonical Normalizers │
                          │ 3. Spatial Context Bonus │
                          │ 4. Strict Ambiguity Gate │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Official ExtractBench    │
                          │ Field Citations & BBoxes │
                          │    (COCO [x, y, w, h])   │
                          └──────────────────────────┘
```

1. **Physical Character Geometry Extraction** (`tonerhound.document`):
   - Directly extracts character-level bounding boxes via `pypdfium2` display lists.
   - Robustly converts bottom-left PDF points into normalized top-left display COCO coordinates `[x, y, w, h]` in `[0.0, 1.0]`.
   - Clusters words into horizontal `VisualLine` structures based on vertical overlap.

2. **Reversible Canonical Normalization** (`tonerhound.normalization`):
   - **Currency & Numbers**: Normalizes `$25.00`, `25`, `(25.00)`, and commas with strict word/digit boundary enforcement (preventing sub-number false matches like `5.0` matching `25.00`).
   - **Dates**: Normalizes ISO, RFC, and natural date expressions (`2026-03-15`, `March 15, 2026`) with minimal-window pruning to eliminate label text absorption.

3. **Spatial Context & Sibling Disambiguation** (`tonerhound.resolution`):
   - When identical values repeat (e.g. `0` votes, repeated prices, duplicate names), TonerHound leverages surrounding record keys, horizontal line collinearity, and Euclidean proximity to ground the exact cell.

4. **Zero Silent False-Grounding Policy**:
   - Refuses to hallucinate: if candidates are ambiguous, TonerHound returns `status="ambiguous"` with `bbox=None`. If unprinted calculations are detected, returns `status="derived"`.

5. **Automated Document Page Offset Calibration** (`tonerhound.benchmark.adapter`):
   - Detects shifts between reported internal listing pages (e.g. `source_page: 1`) and physical PDF canvas pages (due to cover sheets or TOCs) via consensus voting, ensuring cross-page alignment.

---

## 🚀 Quickstart & Usage

### 1. Installation

```bash
git clone https://github.com/vanrajsinh650/TonerHound.git
cd TonerHound
uv sync
```

### 2. Python API

```python
from pathlib import Path
from tonerhound import DocumentIndex, EvidenceResolver, ExtractionInput

# 1. Index the PDF document geometry (cached and fast)
doc_index = DocumentIndex.from_pdf(Path("invoice.pdf"))

# 2. Initialize resolver
resolver = EvidenceResolver(doc_index)

# 3. Ground an extracted field
query = ExtractionInput(
    field="invoice_total",
    value=1450.50,
    field_context="Total Amount Due",
    page_hint=1,
)
result = resolver.resolve(query)

print(f"Status: {result.status}")           # ProvenanceStatus.NORMALIZED
print(f"Page: {result.page}")               # 1
print(f"BBox: {result.bbox.to_coco()}")     # [x, y, w, h] in [0.0, 1.0]
print(f"Matched text: {result.matched_text}") # "$1,450.50"
```

### 3. Running the Test Suite

```bash
# Run unit & adversarial test suites (22 tests)
uv run pytest

# Check linter and code style
uv run ruff check src tests
```

### 4. Running the ExtractBench Benchmark

```bash
# Download official test cases (if needed)
uv run extract-bench download --test --data_dir=research/data/test

# Run benchmark suite and generate structured JSON & Markdown reports
uv run python -m tonerhound.benchmark \
    --data-dir research/data/test \
    --exp-id EXP-001 \
    --output-json research/experiments/EXP-001.json \
    --output-md research/experiments/EXP-001.md
```

---

## 📁 Repository Structure

```
tonerhound/
├── src/tonerhound/
│   ├── geometry/        # Normalized BBox, coordinate transforms, IoU
│   ├── models/          # ProvenanceStatus, DocumentToken, VisualLine, Types
│   ├── normalization/   # Unicode, currency, numeric, and date normalizers
│   ├── document/        # DocumentIndex with pypdfium2 parser & line clustering
│   ├── matching/        # EvidenceMatcher exact, numeric, date, and fuzzy search
│   ├── resolution/      # EvidenceResolver spatial disambiguation & gating
│   └── benchmark/       # Official ExtractBench adapter, evaluator, runner
├── tests/               # Unit, adversarial, and official benchmark tests
├── research/
│   ├── baseline-report.md       # ExtractBench leaderboard analysis
│   ├── competitive-analysis.md  # 10-question audit of competing systems
│   ├── algorithm-comparison.md  # Algorithm trade-offs & spatial designs
│   ├── failure-analysis.md      # Error taxonomy & edge-case mitigations
│   ├── final-report.md          # Comprehensive research & engineering report
│   └── experiments/             # Structured JSON & Markdown experiment logs
├── pyproject.toml
└── README.md
```

---

## 📄 License

Apache 2.0. See `LICENSE` for details.
