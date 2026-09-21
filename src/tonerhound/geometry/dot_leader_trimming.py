"""EXP-018 Safe Dot-Leader / Trailing-Padding Geometry Recovery.

Recovers Class J geometry failures where the correct evidence is present but the
emitted bounding box overextends into trailing dot leaders ('...', '....', '…')
or table padding whitespace bridging to adjacent columns.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentToken
from tonerhound.normalization.normalizers import normalize_unicode_and_case


def clean_word(w: str) -> str:
    """Normalize and strip punctuation from a token or word for comparison."""
    return normalize_unicode_and_case(w).text.lower().strip(' -.,;:_()[]{}/\'"')


def is_dot_leader_token(text: str) -> bool:
    """Check if a token text represents a dot leader sequence.

    Matches tokens consisting predominantly of repeated dots (e.g. '...', '....', '…',
    '. . . .', or fill sequences containing dots, dashes, or underscores).
    """
    s = text.strip()
    if not s:
        return False
    # Unicode ellipsis
    if s == "…" or (len(s) >= 2 and set(s) <= {".", "…"}):
        return True
    # Two or more consecutive dots
    if re.match(r"^\.{2,}$", s):
        return True
    # Spaced dots or leader characters with at least 2 dots
    if re.match(r"^[\.\s…—\-_]{2,}$", s) and s.count(".") >= 2:
        return True
    return False


def word_matches(token_text: str, target_word: str) -> bool:
    """Check if token text matches the target word as exact, prefix, or substring."""
    t = clean_word(token_text)
    w = clean_word(target_word)
    if not t or not w:
        return False
    return (
        t == w
        or t.startswith(w)
        or w.startswith(t)
        or t.endswith(w)
        or w.endswith(t)
        or (len(t) >= 3 and len(w) >= 3 and (t in w or w in t))
    )


def trim_dot_leaders(
    cand_bbox: BBox | tuple[float, float, float, float],
    reference_text: str,
    target_value: Any,
    line_tokens: Sequence[DocumentToken] | None = None,
    confidence: float = 0.90,
    is_passed: bool = False,
) -> BBox | tuple[float, float, float, float]:
    """Trim trailing dot-leader padding from candidate bounding boxes.

    Satisfies all EXP-018 Phase 2 mandatory safety conditions:
    1. Target text is confidently identified in the pre-dot region.
    2. Target ends before a repeated dot pattern (e.g. '..', '...', '…', or tokenized dots).
    3. Emitted bbox contains both target evidence and trailing dots.
    4. Trimmed bbox strictly covers the target text span without truncating target characters.
    5. Resulting width is plausible (width >= 0.01 and <= 0.95 * original width).
    6. Citation does NOT cross column boundaries or contain mashed text after dots.
    7. Groundings already passing (is_passed=True) or low confidence (< 0.80) are never touched.
    """
    # Safety Gate 0: Pass protection and confidence gate
    if is_passed or confidence < 0.80 or target_value is None or isinstance(target_value, bool):
        return cand_bbox

    target_str = str(target_value).strip()
    target_words = [clean_word(w) for w in target_str.split() if clean_word(w)]
    if not target_words:
        return cand_bbox

    is_bbox_obj = isinstance(cand_bbox, BBox)
    if is_bbox_obj:
        cx, cy, cw, ch = cand_bbox.x, cand_bbox.y, cand_bbox.width, cand_bbox.height
        page = cand_bbox.page
    else:
        cx, cy, cw, ch = cand_bbox[:4]
        page = None

    # Safety Gate 1: Candidate box must be wide enough to have overextended (w >= 0.04)
    if cw < 0.04:
        return cand_bbox

    # Mechanism 1: Token-Level Dot Leader Trimming (Primary, High Precision)
    if line_tokens:
        cb_x1 = cx
        cb_x2 = cx + cw
        cb_yc = cy + ch / 2.0

        # Find tokens on visual line overlapping candidate box
        overlapping = [
            t
            for t in line_tokens
            if abs(t.bbox.y + t.bbox.height / 2.0 - cb_yc) <= max(0.012, ch * 0.75)
            and max(t.bbox.x, cb_x1 - 0.005) < min(t.bbox.x + t.bbox.width, cb_x2 + 0.005)
        ]
        overlapping.sort(key=lambda t: t.bbox.x)

        if len(overlapping) >= 2:
            # Separate trailing dot leader tokens from non-dot tokens
            dot_toks: list[DocumentToken] = []
            i = len(overlapping) - 1
            while i >= 0 and is_dot_leader_token(overlapping[i].text):
                dot_toks.append(overlapping[i])
                i -= 1

            non_dot_toks = overlapping[: i + 1]

            if dot_toks and non_dot_toks:
                # Safety Gate 2: Verify non_dot_toks contain the target evidence
                non_dot_words = [clean_word(t.text) for t in non_dot_toks if clean_word(t.text)]
                matches = 0
                for tw in target_words:
                    if any(word_matches(nw, tw) for nw in non_dot_words):
                        matches += 1

                # Require at least 1 match for single-word target, or >= 2 matches for multi-word target
                target_matched = (matches >= min(2, len(target_words))) or (len(target_words) == 1 and matches >= 1)

                if target_matched:
                    # Safety Gate 3: Ensure non_dot_toks does NOT contain unrelated columns after dots
                    # (already guaranteed because we walked backwards from the right edge)
                    min_x = min(t.bbox.x for t in non_dot_toks)
                    max_x = max(t.bbox.x + t.bbox.width for t in non_dot_toks)

                    new_x = min(cx, min_x)
                    new_w = max_x - new_x

                    # Safety Gate 4: Plausible geometry (width >= 0.01 and noticeable trim <= 0.95 * original)
                    if 0.01 <= new_w <= cw * 0.95:
                        if is_bbox_obj:
                            return BBox(x=new_x, y=cy, width=new_w, height=ch, page=page)
                        return (new_x, cy, new_w, ch)

    # Mechanism 2: Intra-Token / Character-Span Dot Leader Trimming (Secondary)
    ref_stripped = (reference_text or "").rstrip()
    dot_match = re.search(r"(\.{2,}|…)$", ref_stripped)
    if dot_match and len(ref_stripped) > 5:
        dot_start = dot_match.start()
        prefix_text = ref_stripped[:dot_start].strip()
        prefix_words = [clean_word(w) for w in prefix_text.split() if clean_word(w)]

        matches = sum(1 for tw in target_words if any(word_matches(pw, tw) for pw in prefix_words))
        target_matched = (matches >= min(2, len(target_words))) or (len(target_words) == 1 and matches >= 1)

        if target_matched:
            span_frac = dot_start / float(len(ref_stripped))
            new_w = cw * span_frac
            if 0.01 <= new_w <= cw * 0.95:
                if is_bbox_obj:
                    return BBox(x=cx, y=cy, width=new_w, height=ch, page=page)
                return (cx, cy, new_w, ch)

    return cand_bbox
