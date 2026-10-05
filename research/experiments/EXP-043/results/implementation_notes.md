# EXP-043: Implementation Notes

### Algorithms Implemented:
1. **Needleman-Wunsch Alignment (`nw_aligner.py`):**
   - Scoring parameters: match=+2, mismatch=-1, gap=-1.
   - Normalized score range: [0, 1]. Length gating rejects candidates with >50% length difference.
2. **Multi-Pass OCR Voting (`multi_pass_ocr.py`):**
   - Preprocessing: Niblack (k=-0.2), Sauvola (k=0.2, R=128), CLAHE, adaptive thresholding.
   - Character consensus across PSM passes.
3. **Convention Inference (`convention_inference.py`):**
   - Minimum sample requirement: 5 verified pairs.
   - Computes mean and standard deviation of spatial box shifts (dx, dy, dw, dh).
4. **Recursive XY-Cut (`recursive_xy_cut.py`):**
   - Projection profile valley detection with recursive subdivision.
   - Morphological line opening (kernel sizes 25x1 and 1x25).
5. **NW Token Sequence Matcher (`nw_token_sequence.py`):**
   - 2-level dynamic programming matching sequences with gap penalty -0.3.
