"""EXP-039 Phase C: OCR Noise-Tolerant Inverted Index.

Target class: REAL_INDEXING_MISS (89,022 fields, 229 documents)
Theoretical ceiling: +11.40 pp
Realistic target: +0.7 to +1.5 pp

Addresses Tesseract OCR single-character substitutions (e.g., 0<->O, 1<->l<->I,
S<->5, 8<->B, Comsolidated<->Consolidated) on scanned pages via bounded
character 3-gram indexing and Levenshtein distance <= 1 search.
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
        # (page_num) -> 3-gram -> list of DocumentTokens
        self._ngram_index_by_page: dict[int, dict[str, list[DocumentToken]]] = defaultdict(lambda: defaultdict(list))
        self._tokens_by_page: dict[int, list[DocumentToken]] = defaultdict(list)
        self._ocr_pages: set[int] = set()
        self._build_index()

    def _build_index(self) -> None:
        """Construct character 3-gram index exclusively for OCR-generated pages."""
        for page in self.doc_index.pages:
            p_num = page.page_number
            # Check if page was OCR generated
            is_ocr = (
                getattr(page, "page_mode", "digital") == "ocr"
                or (len(page.tokens) > 0 and getattr(page.tokens[0], "confidence", 1.0) < 0.99)
                or getattr(page, "is_scanned", False)
            )
            if not is_ocr:
                continue

            self._ocr_pages.add(p_num)
            for token in page.tokens:
                text_clean = token.text.strip().lower()
                if len(text_clean) < 4:
                    continue  # Short tokens (< 4 chars) are too prone to false positives

                self._tokens_by_page[p_num].append(token)
                for i in range(len(text_clean) - 2):
                    ngram = text_clean[i : i + 3]
                    self._ngram_index_by_page[p_num][ngram].append(token)

    def is_ocr_page(self, page_num: int) -> bool:
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

        # Generate query 3-grams
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

        # Sort by edit distance ascending, then similarity descending
        results.sort(key=lambda x: (x[2], -x[1]))
        return [(tok, sim) for tok, sim, _ in results[:max_candidates]]
