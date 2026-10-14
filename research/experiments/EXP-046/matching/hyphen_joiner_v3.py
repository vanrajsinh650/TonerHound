"""Stage 33: Cross-Line Hyphen Joining v3.

Identifies hyphenated word pairs split across visual lines:
- Detects hyphens at line ends: standard hyphen (-), en-dash (–), em-dash (—), soft hyphen (\xad)
- Joins if next line starts with a lowercase letter
- Strictly rejects if next line starts with a digit/number
- Returns normalized COCO bounding box [x, y, w, h] or BBox
"""

from __future__ import annotations

import re
from typing import Sequence
import fitz

from tonerhound.geometry.coordinates import BBox


HYPHEN_CHARS = ("-", "—", "–", "\xad", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014")


def join_crossline_hyphens(
    page: fitz.Page,
    target_value: str,
) -> tuple[float, float, float, float] | None:
    """Finds cross-line hyphenated pair matching target_value on page."""
    if not target_value:
        return None

    target_str = str(target_value).strip()
    if len(target_str) > 120:
        return None

    words = page.get_text("words")
    if not words:
        return None

    pw = float(page.rect.width) if page.rect.width > 0 else 1.0
    ph = float(page.rect.height) if page.rect.height > 0 else 1.0

    target_clean = target_str
    for h in HYPHEN_CHARS:
        target_clean = target_clean.replace(h, "")
    target_clean = re.sub(r"\s+", "", target_clean).lower()
    if not target_clean:
        return None

    n_words = len(words)
    for i, w in enumerate(words):
        x0, y0, x1, y1, text, block, line, word_no = w

        ends_with_hyphen = any(text.endswith(h) for h in HYPHEN_CHARS)
        if not ends_with_hyphen:
            continue

        line_h = max(y1 - y0, 1.0)

        for j in range(i + 1, min(i + 10, n_words)):
            nx0, ny0, nx1, ny1, ntext, nblock, nline, nword = words[j]

            # Must be a subsequent line
            if nline == line:
                continue
            if ny0 - y1 > 2.0 * line_h:
                break  # Too far vertically

            # Must be first word of next line
            if nword != 0:
                continue

            # Reject if next line starts with a number
            if not ntext or ntext[0].isdigit():
                break

            # Join if next line starts with lowercase
            if not ntext[0].islower():
                break

            prefix = text
            for h in HYPHEN_CHARS:
                prefix = prefix.rstrip(h)
            prefix = prefix.strip()

            joined = prefix + ntext.strip()
            joined_clean = joined
            for h in HYPHEN_CHARS:
                joined_clean = joined_clean.replace(h, "")
            joined_clean = re.sub(r"\s+", "", joined_clean).lower()

            if joined_clean == target_clean:
                ux0 = min(x0, nx0) / pw
                uy0 = min(y0, ny0) / ph
                ux1 = max(x1, nx1) / pw
                uy1 = max(y1, ny1) / ph
                uw = max(0.0001, ux1 - ux0)
                uh = max(0.0001, uy1 - uy0)
                return (round(ux0, 6), round(uy0, 6), round(uw, 6), round(uh, 6))

            break

    return None
