"""EXP-042 Phase B: Hyphenation Fix V3.

Targets HYPHENATION (7,623 fields, realistic gain +0.45 pp).
Identifies hyphenated word pairs split across lines using exact clean matching,
end-of-line verification, lowercase/digit initial check, and union bounding box generation.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
import fitz


def join_hyphenated_pairs(
    page: fitz.Page,
    gold_value: str,
) -> tuple[float, float, float, float] | None:
    """Find hyphenated word pairs split across lines.

    Algorithm:
    1. Get all words via page.get_text("words")
    2. Find words ending in -, —, or –
    3. For each, find the NEXT line's FIRST word (vertical gap < 2x line height)
    4. Concatenate: prefix (strip trailing hyphen) + next word
    5. Compare against gold_value after removing hyphens from gold
    6. Emit union bbox in normalized [x, y, w, h] COCO coordinates if match
    """
    if not gold_value:
        return None

    gold_str = str(gold_value)
    if len(gold_str) > 100:
        return None

    words = page.get_text("words")
    if not words:
        return None

    pw = float(page.rect.width) if page.rect.width > 0 else 1.0
    ph = float(page.rect.height) if page.rect.height > 0 else 1.0

    # Strip hyphens and spaces from gold for robust visual comparison
    gold_clean = gold_str.replace("-", "").replace("—", "").replace("–", "")
    gold_clean = re.sub(r"\s+", "", gold_clean).lower()
    if not gold_clean:
        return None

    n_words = len(words)
    for i, w in enumerate(words):
        x0, y0, x1, y1, text, block, line, word_no = w

        if not text.endswith(("-", "—", "–")):
            continue

        line_h = max(y1 - y0, 1.0)

        # Find next line's first word
        for j in range(i + 1, min(i + 8, n_words)):
            nx0, ny0, nx1, ny1, ntext, nblock, nline, nword = words[j]

            # Must be different line and BELOW
            if nline == line:
                continue
            if ny0 - y1 > 2.0 * line_h:
                break  # Too far vertically

            # Must be the FIRST word of next line (or word_no == 0, or lowest x on that line)
            if nword != 0:
                continue

            # Safety: next word starts with lowercase OR digit
            if not ntext or not (ntext[0].islower() or ntext[0].isdigit()):
                continue

            # Concatenate
            prefix = text.rstrip("-—–").strip()
            joined = prefix + ntext.strip()
            joined_clean = re.sub(r"\s+", "", joined).lower()

            if joined_clean == gold_clean:
                ux0 = min(x0, nx0) / pw
                uy0 = min(y0, ny0) / ph
                ux1 = max(x1, nx1) / pw
                uy1 = max(y1, ny1) / ph
                uw = max(0.0001, ux1 - ux0)
                uh = max(0.0001, uy1 - uy0)
                return (round(ux0, 6), round(uy0, 6), round(uw, 6), round(uh, 6))

            break  # Only try first word of next line

    return None
