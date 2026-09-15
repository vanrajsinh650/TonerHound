"""OCR-aware text, numeric, and character normalization with exact position tracking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from tonerhound.normalization.normalizers import (
    _CHAR_REPLACEMENTS,
    is_number_equal,
    parse_numeric_value,
)

# Common OCR noise characters that appear as leading/trailing specks or column borders
OCR_NOISE_CHARS = set(r"*“\"'’‘`_~|#.-:;,\()[]{}<>=+/!?%$—–§•")

# OCR confusable characters mapping (digits and letters)
OCR_CONFUSIONS = str.maketrans({
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "L": "1",
    "i": "1",
    "|": "1",
    "S": "5",
    "s": "5",
    "$": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
    "z": "2",
})


@dataclass(frozen=True, slots=True)
class OCRNormalizedText:
    """A normalized string accompanied by a strict 1-to-1 mapping to original character offsets."""

    text: str
    char_map: list[int]
    original_text: str

    def span_to_original_range(self, start: int, end: int) -> tuple[int, int]:
        """Convert a [start, end) span in normalized text to [orig_start, orig_end)."""
        if not self.char_map or start >= len(self.char_map):
            return (0, 0)
        orig_start = self.char_map[max(0, start)]
        last_norm_idx = min(end - 1, len(self.char_map) - 1)
        orig_end = self.char_map[last_norm_idx] + 1
        return (orig_start, max(orig_start, orig_end))


class OCRNormalizer:
    """Deterministic OCR normalizer preserving exact source character geometry."""

    @staticmethod
    def normalize(source: str) -> OCRNormalizedText:
        """Normalize Unicode ligatures, smart quotes, and casefold, preserving char_map."""
        norm_chars: list[str] = []
        char_map: list[int] = []

        for orig_idx, ch in enumerate(source):
            replaced = _CHAR_REPLACEMENTS.get(ch, ch)
            decomposed = unicodedata.normalize("NFKC", replaced)
            folded = decomposed.casefold()

            for folded_ch in folded:
                norm_chars.append(folded_ch)
                char_map.append(orig_idx)

        return OCRNormalizedText(text="".join(norm_chars), char_map=char_map, original_text=source)


def clean_ocr_noise_span(text: str) -> tuple[str, int, int]:
    """Find valid alphanumeric content boundaries within token, returning (cleaned, start_idx, end_idx)."""
    if not text:
        return "", 0, 0

    start = 0
    while start < len(text) and (text[start] in OCR_NOISE_CHARS or text[start].isspace()):
        start += 1

    end = len(text)
    while end > start and (text[end - 1] in OCR_NOISE_CHARS or text[end - 1].isspace()):
        end -= 1

    cleaned = text[start:end]
    return cleaned, start, end


def ocr_canonical_numeric(text: str) -> str:
    """Map confusable OCR digits and strip non-numeric punctuation."""
    cleaned = re.sub(r"^[$€£¥§]+", "", text.strip())
    cleaned = cleaned.replace("·", ".").replace("•", ".").translate(OCR_CONFUSIONS)
    return re.sub(r"[^0-9.]", "", cleaned)


def ocr_consonant_skeleton(text: str) -> str:
    """Extract consonant skeleton for phonetic / typo-tolerant name comparison."""
    norm = text.upper().strip()
    return "".join(c for c in norm if c in "BCDFGHJKLMNPQRSTVWXYZ0123456789")


def ocr_similarity(s1: str, s2: str) -> float:
    """Compute OCR-aware string similarity accounting for space elision, punctuation, and glyph confusions."""
    if s1 == s2:
        return 1.0

    # Strip non-alphanumeric for space-elided comparison (e.g. TARGET EXTERMINATING vs TARGETEXTERMINATING)
    c1 = re.sub(r"[^A-Za-z0-9]+", "", s1).upper()
    c2 = re.sub(r"[^A-Za-z0-9]+", "", s2).upper()
    if not c1 or not c2:
        return 0.0

    if c1 == c2:
        return 1.0

    # Check confusable mapping (e.g. 2093 vs 2003, BIGMIKE'SAC vs BIG MIKE'S A/C)
    m1 = c1.translate(OCR_CONFUSIONS)
    m2 = c2.translate(OCR_CONFUSIONS)
    if m1 == m2:
        return 0.98

    # Fast check: substring containment (minimum 3 characters and >= 80% coverage)
    if len(c1) >= 3 and len(c2) >= 3 and (c1 in c2 or c2 in c1):
        coverage = min(len(c1), len(c2)) / max(len(c1), len(c2))
        if coverage >= 0.80:
            return 0.85 + 0.15 * coverage

    # Rapidfuzz ratio on confusable mapped strings
    base_ratio = fuzz.ratio(m1, m2) / 100.0

    # Consonant skeleton bonus for long words
    if len(c1) >= 6 and len(c2) >= 6:
        sk1 = ocr_consonant_skeleton(c1)
        sk2 = ocr_consonant_skeleton(c2)
        if sk1 and sk2:
            sk_ratio = fuzz.ratio(sk1, sk2) / 100.0
            return max(base_ratio, sk_ratio * 0.92)

    return base_ratio
