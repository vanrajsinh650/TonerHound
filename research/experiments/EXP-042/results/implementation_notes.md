# EXP-042 IMPLEMENTATION NOTES
## Maximum Deterministic Recovery

### Architecture Overview
EXP-042 implemented the complete suite of deterministic grounding tools across 8 distinct failure classes:

1. **HyphenationJoinerV3 (`hyphen_joiner_v3.py`):**
   - Exact prefix + line-end hyphen stripping + lowercase/digit start check.
   - Emits union bbox spanning line wrap.

2. **DateLiteralVariants (`date_variants.py`):**
   - Parses dates into flexible `(y, m, d)` components and expands to 18 literal formats.
   - Exact matching against page tokens without semantic loss.

3. **MultiLineAssembler (`multiline_assembler.py`):**
   - Sequential token stream alignment with reading-order line gap validation.

4. **VisualDetectorV2 (`visual_detector_v2.py`):**
   - Morphological opening with horizontal/vertical kernels `(25, 1)` and `(1, 25)` to eliminate table borders before contour extraction.
   - Aspect ratio and ink-density verification for signatures.

5. **CellGrounder (`cell_grounder.py`):**
   - PyMuPDF `page.find_tables(strategy="lines_strict")` cell text containment.
   - Vector drawing vertical line clamping.

6. **RegionAndRotation (`region_and_rotation.py`):**
   - Spatial proximity clustering for cross-column addresses.
   - Hough transform rotation angle calculation.

7. **CharAndWordMatcher (`char_and_word_matcher.py`):**
   - Character stream substring search on `rawdict` character streams.
   - Stop-word skipping multi-token sequence matching.
