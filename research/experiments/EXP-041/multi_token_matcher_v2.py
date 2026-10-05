"""EXP-041 Phase C: Multi-Token Sequence Matcher V2.

Target class: REAL_INDEXING_MISS / SUB_MULTI_TOKEN (65,486 fields).
Realistic gain: +0.3–0.5 pp.

Finds multi-word sequences with stop-word skipping, numeric normalization (1,000 vs 1000),
punctuation-agnostic matching (U.S.D. vs USD), bounded token gap <= 3, and RapidFuzz
partial_ratio_alignment fallback.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
import fitz
from rapidfuzz import fuzz

STOP_WORDS = {"of", "and", "&", "the", "a", "an", "for", "in", "on", "at", "to"}


def normalize_token_text(text: str) -> str:
    """Normalize a token string: lowercase, strip punctuation and extra whitespace."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_numeric(text: str) -> str:
    """Remove commas and dollar signs for numeric comparisons."""
    if not text:
        return ""
    return str(text).replace(",", "").replace("$", "").strip()


def normalize_for_match(text: str) -> str:
    """Normalize full string for matching."""
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


class DocumentTokenV2:
    def __init__(self, text: str, bbox: list[float], block: int = 0, line: int = 0) -> None:
        self.text = text
        self.text_norm = normalize_token_text(text)
        self.text_num = normalize_numeric(self.text_norm)
        self.bbox = bbox
        self.block = block
        self.line = line


class MultiTokenSequenceMatcherV2:
    """Matches multi-token strings across page tokens with tolerance and fuzzy fallback."""

    def __init__(self, page: fitz.Page) -> None:
        self.page = page
        self.pw = float(page.rect.width) if page.rect.width > 0 else 1.0
        self.ph = float(page.rect.height) if page.rect.height > 0 else 1.0

        raw_words = page.get_text("words")
        self.tokens: list[DocumentTokenV2] = []

        for w in raw_words:
            x0, y0, x1, y1, text, bno, lno, wno = w
            norm_box = [
                max(0.0, min(1.0, x0 / self.pw)),
                max(0.0, min(1.0, y0 / self.ph)),
                max(0.0, min(1.0, (x1 - x0) / self.pw)),
                max(0.0, min(1.0, (y1 - y0) / self.ph)),
            ]
            self.tokens.append(DocumentTokenV2(text, norm_box, bno, lno))

    def match_sequence(
        self,
        gold_value: str,
        max_gap: int = 3,
        min_similarity: float = 85.0,
    ) -> list[float] | None:
        """
        Match multi-token sequences with gap tolerance and fuzzy fallback.

        Primary path: exact token-by-token matching with gap skipping and numeric tolerance.
        Fallback path: RapidFuzz partial_ratio_alignment on concatenated window.
        """
        if not gold_value or not self.tokens:
            return None

        gold_norm = normalize_for_match(gold_value)
        gold_words = [normalize_token_text(w) for w in gold_norm.split() if normalize_token_text(w)]
        if len(gold_words) < 2:
            return None

        gold_words_num = [normalize_numeric(w) for w in gold_words]
        n_tokens = len(self.tokens)

        # --- Primary: exact token-by-token with gap tolerance and numeric normalization ---
        for i in range(n_tokens):
            first_tok = self.tokens[i]
            if first_tok.text_norm != gold_words[0] and first_tok.text_num != gold_words_num[0]:
                continue

            matched_boxes = []
            gold_idx = 0
            t_idx = i
            skipped = 0

            while gold_idx < len(gold_words) and t_idx < n_tokens:
                tok = self.tokens[t_idx]
                target_word = gold_words[gold_idx]
                target_num = gold_words_num[gold_idx]

                if tok.text_norm == target_word or tok.text_num == target_num:
                    matched_boxes.append(tok.bbox)
                    gold_idx += 1
                    t_idx += 1
                elif tok.text_norm in STOP_WORDS and target_word not in STOP_WORDS:
                    # Skip stop words in document
                    t_idx += 1
                elif target_word in STOP_WORDS:
                    # Skip stop word in gold query
                    gold_idx += 1
                elif skipped < max_gap:
                    skipped += 1
                    t_idx += 1
                else:
                    break

            if gold_idx == len(gold_words) and len(matched_boxes) >= 2:
                return union_bbox(matched_boxes)

        # --- Fallback: RapidFuzz alignment on sliding windows ---
        val_clean = normalize_token_text(gold_value)
        min_win = len(gold_words)
        max_win = min(min_win + 5, 15)

        for window_size in range(min_win, max_win + 1):
            if window_size > n_tokens:
                break
            for i in range(n_tokens - window_size + 1):
                window = self.tokens[i : i + window_size]
                win_text = " ".join(t.text for t in window)
                win_clean = normalize_token_text(win_text)

                if not win_clean:
                    continue

                alignment = fuzz.partial_ratio_alignment(val_clean, win_clean)
                if alignment and alignment.score >= min_similarity:
                    return union_bbox([t.bbox for t in window])

        return None
