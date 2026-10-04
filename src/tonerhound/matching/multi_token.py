"""Multi-token sequence matcher for contiguous and near-contiguous tokens.

Target: Solves multi-word string matching split across multiple token atoms
on a document page.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import fitz


def normalize_for_match(text: str) -> str:
    """Normalize text for robust multi-token sequence matching."""
    if not text:
        return ""
    t = text.lower()
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

    w = max(0.001, max_x - min_x)
    h = max(0.001, max_y - min_y)
    return [round(min_x, 6), round(min_y, 6), round(w, 6), round(h, 6)]


class MultiTokenSequenceMatcher:
    """Matches multi-word values to sequences of page tokens."""

    def __init__(self, page: fitz.Page) -> None:
        self.page = page
        self.pw = float(page.rect.width) if page.rect.width > 0 else 1.0
        self.ph = float(page.rect.height) if page.rect.height > 0 else 1.0

        raw_words = page.get_text("words")
        self.tokens: list[dict[str, Any]] = []

        for w in raw_words:
            x0, y0, x1, y1, text, bno, lno, wno = w
            norm_box = [
                max(0.0, min(1.0, x0 / self.pw)),
                max(0.0, min(1.0, y0 / self.ph)),
                max(0.0, min(1.0, (x1 - x0) / self.pw)),
                max(0.0, min(1.0, (y1 - y0) / self.ph)),
            ]
            self.tokens.append({
                "text": text,
                "text_norm": normalize_for_match(text),
                "bbox": norm_box,
                "block": bno,
                "line": lno,
            })

    def match_sequence(
        self,
        value: str,
        max_gap: int = 2,
    ) -> list[float] | None:
        """Find token sequence matching the target multi-token value."""
        if not value or not self.tokens:
            return None

        val_norm = normalize_for_match(value)
        val_words = val_norm.split()
        if not val_words:
            return None

        first_word = val_words[0]
        val_word_count = len(val_words)
        n_tokens = len(self.tokens)

        for i in range(n_tokens):
            tok_norm = self.tokens[i]["text_norm"]
            if not tok_norm:
                continue

            tok_words = tok_norm.split()
            if not tok_words:
                continue

            if tok_words[0] != first_word and first_word not in tok_words:
                continue

            max_window = min(n_tokens, i + val_word_count + max_gap * 2 + 5)
            window = self.tokens[i:max_window]

            candidate_texts: list[str] = []
            candidate_boxes: list[list[float]] = []

            for t in window:
                candidate_texts.append(t["text_norm"])
                candidate_boxes.append(t["bbox"])

                joined_text = " ".join(candidate_texts)
                joined_norm = re.sub(r"\s+", " ", joined_text).strip()

                if joined_norm == val_norm:
                    return union_bbox(candidate_boxes)

                if len(joined_norm) > len(val_norm) + 15:
                    break

        return None
