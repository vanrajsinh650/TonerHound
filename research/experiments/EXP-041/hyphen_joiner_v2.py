"""EXP-041 Phase B: Trailing Line Hyphen Joiner V2.

Target class: HYPHENATION (7,623 fields)
Realistic gain: +0.68 pp

Words split across lines by trailing hyphens (e.g. Consoli-\ndated) are indexed as
two separate tokens in standard pipelines. This module identifies trailing hyphenated
tokens at line ends, stitches them with the subsequent line's leading lowercase token,
and returns their unified bounding box.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
import fitz


def normalize_for_match(text: str) -> str:
    """Normalize text for exact visual/token matching."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\s\.\,\-\$\%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def union_bbox(bboxes: Sequence[Sequence[float]]) -> list[float] | None:
    """Compute the union bounding box [x, y, w, h] in normalized COCO coordinates."""
    if not bboxes:
        return None
    valid_boxes = [b for b in bboxes if b and len(b) == 4 and b[2] > 0 and b[3] > 0]
    if not valid_boxes:
        return None

    min_x = min(b[0] for b in valid_boxes)
    min_y = min(b[1] for b in valid_boxes)
    max_x = max(b[0] + b[2] for b in valid_boxes)
    max_y = max(b[1] + b[3] for b in valid_boxes)

    w = max(0.0001, max_x - min_x)
    h = max(0.0001, max_y - min_y)
    return [round(min_x, 6), round(min_y, 6), round(w, 6), round(h, 6)]


class HyphenationJoinerV2:
    """Stitches trailing hyphenated words across line boundaries with strict safety rules."""

    def __init__(self, page: fitz.Page) -> None:
        self.page = page
        self.pw = float(page.rect.width) if page.rect.width > 0 else 1.0
        self.ph = float(page.rect.height) if page.rect.height > 0 else 1.0

        # words: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        raw_words = page.get_text("words")
        self.words = raw_words

    def join_hyphenated_tokens(
        self,
        gold_value: str,
    ) -> list[float] | None:
        """
        Join hyphenated words split across line breaks.

        Algorithm:
        1. Extract all words via page.get_text("words")
        2. For each word ending with '-', '—', or '–':
           a. Find the next line's first word (vertical gap < 2× line height)
           b. Concatenate: prefix (strip trailing hyphen) + next word
           c. If concatenated matches gold_value, emit union bbox
        """
        if not gold_value:
            return None

        gold_norm = normalize_for_match(gold_value)
        if not gold_norm or len(gold_norm) > 100:
            return None

        words = self.words
        n_words = len(words)

        for i, word in enumerate(words):
            x0, y0, x1, y1, text, block, line, word_no = word

            if not text.endswith(('-', '—', '–')):
                continue

            # Height of the current line
            line_height = max(y1 - y0, 1.0)

            # Find words on next line (different line_no, same or next block)
            for j in range(i + 1, min(i + 10, n_words)):
                nx0, ny0, nx1, ny1, ntext, nblock, nline, nword = words[j]

                # Must be on a different line but below
                if nline == line:
                    continue
                if ny0 - y1 > 2.0 * line_height:
                    # Too far vertically
                    break

                # Safety: Only join if next word starts with lowercase
                if not ntext or not ntext[0].islower():
                    continue

                prefix = text[:-1].strip()
                joined = prefix + ntext.strip()
                joined_with_hyphen = prefix + "-" + ntext.strip()

                if len(joined) > 100:
                    break

                joined_norm = normalize_for_match(joined)
                joined_hyphen_norm = normalize_for_match(joined_with_hyphen)

                if joined_norm == gold_norm or joined_hyphen_norm == gold_norm:
                    # Convert to normalized COCO bbox
                    box1 = [
                        max(0.0, min(1.0, x0 / self.pw)),
                        max(0.0, min(1.0, y0 / self.ph)),
                        max(0.0, min(1.0, (x1 - x0) / self.pw)),
                        max(0.0, min(1.0, (y1 - y0) / self.ph)),
                    ]
                    box2 = [
                        max(0.0, min(1.0, nx0 / self.pw)),
                        max(0.0, min(1.0, ny0 / self.ph)),
                        max(0.0, min(1.0, (nx1 - nx0) / self.pw)),
                        max(0.0, min(1.0, (ny1 - ny0) / self.ph)),
                    ]
                    return union_bbox([box1, box2])

                # Only check the first word of the next line
                break

        return None
