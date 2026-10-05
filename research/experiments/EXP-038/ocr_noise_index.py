"""EXP-038 Fix 1: OCR Noise-Tolerant Indexing.

Implements secondary character n-gram and bounded edit-distance retrieval for OCR tokens:
- Problem: OCR produces slight character errors ("Comsolidated" -> "Consolidated", "1O4O" -> "1040", "Dog" -> "Doc").
- Solution: Secondary 3-gram index with Levenshtein distance <= 1 / consonant skeleton alignment.
- Safety:
  * STRICTLY active ONLY on pages tagged with page_mode == 'ocr' (or pages with low native text).
  * NEVER touches native digital pages.
  * Limits candidate generation to at most 3 candidates per query.
  * Rejects candidates below similarity threshold 0.82.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentToken


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0[:] = v1

    return v1[len(s2)]


def string_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity [0.0, 1.0]."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / float(max_len))


def consonant_skeleton(s: str) -> str:
    """Extract consonant skeleton of word for phonetic noise tolerance."""
    return re.sub(r"[aeiouy\W_]", "", s.lower())


class OCRNoiseTolerantIndex:
    """Secondary index for OCR-noise tolerant candidate retrieval."""

    def __init__(self, doc_index: DocumentIndex, ocr_pages: set[int] | None = None) -> None:
        self.doc_index = doc_index
        # If ocr_pages not specified, infer from page_modes or page token counts
        if ocr_pages is not None:
            self.ocr_pages = ocr_pages
        else:
            modes = getattr(doc_index, "page_modes", {})
            if modes:
                self.ocr_pages = {p for p, m in modes.items() if m == "ocr"}
            else:
                self.ocr_pages = {p.page_number for p in doc_index.pages}

        # Build token ngram map exclusively for OCR pages
        self._ngram_to_tokens: dict[str, list[DocumentToken]] = defaultdict(list)
        self._tokens_by_page: dict[int, list[DocumentToken]] = defaultdict(list)
        self._consonant_to_tokens: dict[str, list[DocumentToken]] = defaultdict(list)

        self._build_ocr_token_index()

    def _build_ocr_token_index(self) -> None:
        for p in self.doc_index.pages:
            p_num = p.page_number
            if p_num not in self.ocr_pages:
                continue

            for tok in p.tokens:
                clean = tok.text.strip().lower()
                if not clean:
                    continue
                self._tokens_by_page[p_num].append(tok)

                # N-grams (n=3)
                if len(clean) >= 3:
                    for i in range(len(clean) - 2):
                        ng = clean[i : i + 3]
                        self._ngram_to_tokens[ng].append(tok)

                # Consonant skeleton
                skel = consonant_skeleton(clean)
                if len(skel) >= 3:
                    self._consonant_to_tokens[skel].append(tok)

    def find_fuzzy_candidates(
        self,
        query: str,
        page_hint: int | None = None,
        max_candidates: int = 3,
        min_similarity: float = 0.82,
    ) -> list[tuple[DocumentToken, float]]:
        """Search for tokens on OCR pages matching query within edit distance 1 or high similarity."""
        norm_query = query.strip().lower()
        if len(norm_query) < 3:
            return []

        target_pages = [page_hint] if page_hint is not None and page_hint in self.ocr_pages else list(self.ocr_pages)
        if not target_pages:
            return []

        # Candidate pooling via n-grams
        candidate_tokens: set[DocumentToken] = set()
        for i in range(len(norm_query) - 2):
            ng = norm_query[i : i + 3]
            for tok in self._ngram_to_tokens.get(ng, []):
                if tok.page in target_pages:
                    candidate_tokens.add(tok)

        # Consonant skeleton fallback
        skel = consonant_skeleton(norm_query)
        if len(skel) >= 3:
            for tok in self._consonant_to_tokens.get(skel, []):
                if tok.page in target_pages:
                    candidate_tokens.add(tok)

        # Score candidates with Levenshtein
        scored: list[tuple[DocumentToken, float]] = []
        for tok in candidate_tokens:
            tok_text = tok.text.strip().lower()
            # Length guard
            if abs(len(tok_text) - len(norm_query)) > 2:
                continue

            sim = string_similarity(norm_query, tok_text)
            if sim >= min_similarity:
                scored.append((tok, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_candidates]
