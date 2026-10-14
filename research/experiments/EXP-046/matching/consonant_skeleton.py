"""EXP-046 Stage 15: Consonant Skeleton Matcher.

Reduces words to consonant skeletons by stripping vowels, punctuation, and duplicate consonants,
recovering heavily distorted OCR tokens (similarity >= 0.85).
"""

from __future__ import annotations

import re
from typing import Any
from rapidfuzz import fuzz

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage


def extract_consonant_skeleton(text: str) -> str:
    """Extract ordered lowercase consonant skeleton from word."""
    cleaned = re.sub(r"[^a-zA-Z]", "", text).lower()
    # Remove standard English vowels
    consonants = re.sub(r"[aeiouy]", "", cleaned)
    # Collapse adjacent duplicate consonants
    collapsed = re.sub(r"(.)\1+", r"\1", consonants)
    return collapsed


class ConsonantSkeletonMatcher:
    """Matches words via consonant skeletons to bypass vowel OCR substitutions."""

    def __init__(self, skeleton_threshold: float = 85.0) -> None:
        self.skeleton_threshold = skeleton_threshold

    def match_skeleton(
        self, target_value: Any, page_obj: DocumentPage
    ) -> tuple[BBox | None, str | None]:
        """Search page tokens by consonant skeleton similarity."""
        target_str = str(target_value).strip()
        target_skel = extract_consonant_skeleton(target_str)
        if len(target_skel) < 3:
            return None, None

        best_score = 0.0
        best_tok = None

        for t in page_obj.tokens:
            t_skel = extract_consonant_skeleton(t.text)
            if len(t_skel) < 2:
                continue

            sim = fuzz.ratio(target_skel, t_skel)
            if sim > best_score:
                best_score = sim
                best_tok = t

        if best_score >= self.skeleton_threshold and best_tok:
            b = best_tok.bbox
            return BBox(b.x, b.y, b.width, b.height, page_obj.page_number), best_tok.text

        return None, None
