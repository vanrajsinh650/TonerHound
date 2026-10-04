"""OCR noise-tolerant inverted index for degraded or OCR-noisy pages.

Target: Addresses Tesseract OCR single-character substitutions (e.g. 0<->O, 1<->l<->I)
on scanned pages via bounded character 3-gram indexing and Levenshtein search.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import Levenshtein

from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import DocumentToken


class OCRNoiseTolerantIndex:
    """Secondary character 3-gram index for OCR-noisy tokens."""

    def __init__(self, doc_index: DocumentIndex) -> None:
        self.doc_index = doc_index
        self._ngram_index_by_page: dict[int, dict[str, list[DocumentToken]]] = defaultdict(lambda: defaultdict(list))
        self._tokens_by_page: dict[int, list[DocumentToken]] = defaultdict(list)
        self._ocr_pages: set[int] = set()
        self._build_index()

    def _build_index(self) -> None:
        """Construct character 3-gram index exclusively for OCR-generated pages."""
        page_modes = getattr(self.doc_index, "page_modes", {})
        for page in self.doc_index.pages:
            p_num = page.page_number
            is_ocr = (
                page_modes.get(p_num) == "ocr"
                or getattr(page, "page_mode", "digital") == "ocr"
                or (len(page.tokens) > 0 and getattr(page.tokens[0], "confidence", 1.0) < 0.99)
                or getattr(page, "is_scanned", False)
            )
            if not is_ocr:
                continue

            self._ocr_pages.add(p_num)
            for token in page.tokens:
                text_clean = token.text.strip().lower()
                if len(text_clean) < 4:
                    continue

                self._tokens_by_page[p_num].append(token)
                for i in range(len(text_clean) - 2):
                    ngram = text_clean[i : i + 3]
                    self._ngram_index_by_page[p_num][ngram].append(token)

    def is_ocr_page(self, page_num: int) -> bool:
        """Return True if page was detected as OCR-scanned."""
        return page_num in self._ocr_pages

    def query(
        self,
        query_token: str,
        page_hint: int | None = None,
        max_distance: int = 1,
        min_similarity: float = 0.75,
        max_candidates: int = 3,
    ) -> list[tuple[DocumentToken, float]]:
        """Query bounded fuzzy candidates for query_token on OCR pages.

        Returns:
            list of (DocumentToken, similarity) sorted by distance / similarity.
        """
        query_clean = query_token.strip().lower()
        if len(query_clean) < 4:
            return []

        target_pages = [page_hint] if (page_hint is not None and page_hint in self._ocr_pages) else list(self._ocr_pages)
        if not target_pages:
            return []

        query_ngrams = set()
        for i in range(len(query_clean) - 2):
            query_ngrams.add(query_clean[i : i + 3])

        candidate_tokens: dict[int, DocumentToken] = {}
        for p in target_pages:
            p_ngrams = self._ngram_index_by_page[p]
            for ng in query_ngrams:
                for tok in p_ngrams.get(ng, []):
                    candidate_tokens[id(tok)] = tok

        results: list[tuple[DocumentToken, float, int]] = []
        for tok in candidate_tokens.values():
            tok_clean = tok.text.strip().lower()
            dist = Levenshtein.distance(query_clean, tok_clean)
            if dist <= max_distance:
                max_len = max(len(query_clean), len(tok_clean))
                sim = 1.0 - (dist / max(1, max_len))
                if sim >= min_similarity:
                    results.append((tok, sim, dist))

        results.sort(key=lambda x: (x[2], -x[1]))
        return [(tok, sim) for tok, sim, _ in results[:max_candidates]]
