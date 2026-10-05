# EXP-041 IMPLEMENTATION NOTES
## Pure Deterministic Geometric Recovery

### Architecture Overview
EXP-041 implemented five independent geometric and visual recovery modules operating in a strict priority cascade:

1. **ColumnRailGrounder (`column_rail_grounder.py`):**
   - Utilizes `fitz.Page.find_tables(strategy="lines_strict")`.
   - Extracts vertical rails by analyzing cell bounds.
   - Maps field paths to columns using header keywords (`shares`, `par`, `coupon`, `maturity`, etc.).
   - Emits candidate bbox bounded within column rail `[x_left, x_right]`.

2. **HyphenationJoinerV2 (`hyphen_joiner_v2.py`):**
   - Extracts word tokens via `page.get_text("words")`.
   - Detects trailing hyphen tokens (`-`, `—`, `–`).
   - Checks subsequent line first word with vertical gap < 2× line height.
   - Joins prefix + lowercase token and emits union bbox.

3. **MultiTokenSequenceMatcherV2 (`multi_token_matcher_v2.py`):**
   - Primary: Token sequence matching with stop-word skipping (`of`, `and`, `&`, `the`, `in`, `for`).
   - Numeric normalization: removes commas for exact token alignment.
   - Fallback: RapidFuzz `fuzz.partial_ratio_alignment` on sliding window (size: length .. length+5).

4. **MultiRegionAssemblerV2 (`multi_region_assembler_v2.py`):**
   - Spatial proximity clustering with vertical gap dy <= 0.05 and horizontal gap dx <= 0.15.
   - Hough transform rotation detection and correction.

5. **Punctuation Variants (`punct_variants.py`):**
   - Strips trailing punctuation (`.,;:`) without semantic distortion.
   - Dash-as-zero mapping for financial tables.
