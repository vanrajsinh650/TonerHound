"""EXP-046 Stage 14: Character-Level Multi-Scale N-Gram Index V2.

Unions 2-gram, 3-gram, and 4-gram character inverted indices to handle severe OCR noise,
ranking candidates by composite Jaccard overlap (threshold >= 0.40).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def get_ngrams(text: str, n: int) -> set[str]:
    cleaned = text.lower().strip()
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


class NGramIndexV2:
    """Multi-scale character n-gram inverted index over page tokens."""

    def __init__(self, jaccard_threshold: float = 0.40) -> None:
        self.jaccard_threshold = jaccard_threshold

    def build_index(self, tokens: Sequence[DocumentToken]) -> dict[int, dict[str, list[DocumentToken]]]:
        """Index tokens by 2-grams, 3-grams, and 4-grams."""
        index: dict[int, dict[str, list[DocumentToken]]] = {2: defaultdict(list), 3: defaultdict(list), 4: defaultdict(list)}
        for t in tokens:
            for n in [2, 3, 4]:
                for ng in get_ngrams(t.text, n):
                    index[n][ng].append(t)
        return index

    def query_candidate(
        self, target_value: Any, page_obj: DocumentPage
    ) -> tuple[BBox | None, str | None]:
        """Find best token match via multi-scale n-gram overlap."""
        target_str = str(target_value).strip().lower()
        if len(target_str) < 2:
            return None, None

        tokens = page_obj.tokens
        if not tokens:
            return None, None

        target_ng2 = get_ngrams(target_str, 2)
        target_ng3 = get_ngrams(target_str, 3)
        target_ng4 = get_ngrams(target_str, 4)

        best_score = 0.0
        best_tok: DocumentToken | None = None

        for t in tokens:
            t_txt = t.text.strip().lower()
            if not t_txt:
                continue

            tok_ng2 = get_ngrams(t_txt, 2)
            tok_ng3 = get_ngrams(t_txt, 3)
            tok_ng4 = get_ngrams(t_txt, 4)

            j2 = len(target_ng2 & tok_ng2) / max(1, len(target_ng2 | tok_ng2))
            j3 = len(target_ng3 & tok_ng3) / max(1, len(target_ng3 | tok_ng3))
            j4 = len(target_ng4 & tok_ng4) / max(1, len(target_ng4 | tok_ng4))

            # Composite Jaccard with higher weight on longer n-grams
            score = 0.2 * j2 + 0.4 * j3 + 0.4 * j4
            if score > best_score:
                best_score = score
                best_tok = t

        if best_score >= self.jaccard_threshold and best_tok:
            b = best_tok.bbox
            return BBox(b.x, b.y, b.width, b.height, page_obj.page_number), best_tok.text

        return None, None
