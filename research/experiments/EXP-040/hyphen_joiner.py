"""EXP-040 Phase E.1: Trailing Hyphen Joiner.

Targets HYPHENATION (7,623 fields, theoretical ceiling +0.98 pp, estimated gain +0.68 pp).
Detects end-of-line tokens with trailing hyphens and stitches them with the subsequent
line's leading token, computing their unified bounding box.
"""

from __future__ import annotations

import re
import fitz
from typing import Any, Sequence


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


class HyphenationJoiner:
    """Stitches broken hyphenated words across line boundaries."""

    def __init__(self, page: fitz.Page) -> None:
        self.page = page
        self.pw = float(page.rect.width) if page.rect.width > 0 else 1.0
        self.ph = float(page.rect.height) if page.rect.height > 0 else 1.0

        # Extract words: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        raw_words = page.get_text("words")
        self.lines: dict[tuple[int, int], list[dict[str, Any]]] = {}

        for w in raw_words:
            x0, y0, x1, y1, text, bno, lno, wno = w
            norm_box = [
                max(0.0, min(1.0, x0 / self.pw)),
                max(0.0, min(1.0, y0 / self.ph)),
                max(0.0, min(1.0, (x1 - x0) / self.pw)),
                max(0.0, min(1.0, (y1 - y0) / self.ph)),
            ]
            key = (bno, lno)
            if key not in self.lines:
                self.lines[key] = []
            self.lines[key].append({
                "text": text,
                "bbox": norm_box,
                "wno": wno,
            })

        # Pre-compute joined hyphen pairs
        self.hyphen_pairs: list[dict[str, Any]] = []
        sorted_keys = sorted(self.lines.keys())

        for idx, k in enumerate(sorted_keys[:-1]):
            line1 = self.lines[k]
            if not line1:
                continue
            last_tok = line1[-1]
            last_text = last_tok["text"].strip()

            if last_text.endswith(("-", "—", "–")):
                # Check next line in same block or next block
                next_k = sorted_keys[idx + 1]
                line2 = self.lines[next_k]
                if not line2:
                    continue

                first_tok = line2[0]
                first_text = first_tok["text"].strip()

                # Vertical pitch check: line2 should be directly beneath line1
                y1 = last_tok["bbox"][1] + last_tok["bbox"][3]
                y2 = first_tok["bbox"][1]
                if 0.0 <= (y2 - y1) <= max(last_tok["bbox"][3], 0.01) * 2.0:
                    prefix = last_text.rstrip("-—–")
                    joined_word = f"{prefix}{first_text}"
                    joined_hyphen = f"{prefix}-{first_text}"
                    u_box = union_bbox([last_tok["bbox"], first_tok["bbox"]])

                    self.hyphen_pairs.append({
                        "joined_word": joined_word.lower(),
                        "joined_hyphen": joined_hyphen.lower(),
                        "bbox": u_box,
                    })

    def match_hyphenated(self, value: str) -> list[float] | None:
        """Check if target value matches a stitched hyphenated pair."""
        if not value or not self.hyphen_pairs:
            return None

        val_clean = value.lower().strip()
        val_no_hyphen = val_clean.replace("-", "").replace("—", "").replace(" ", "")

        for pair in self.hyphen_pairs:
            pair_word = pair["joined_word"]
            pair_hyphen = pair["joined_hyphen"]

            if val_clean == pair_word or val_clean == pair_hyphen:
                return pair["bbox"]

            if val_no_hyphen == pair_word.replace("-", ""):
                return pair["bbox"]

        return None
