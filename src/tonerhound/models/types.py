"""Core data contracts for TonerHound."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tonerhound.geometry.coordinates import BBox


class ProvenanceStatus(str, Enum):
    """Provenance status of a resolved value."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    MULTI_REGION = "multi_region"
    AMBIGUOUS = "ambiguous"
    DERIVED = "derived"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class DocumentToken:
    """Atomic token (word or punctuation atom) with exact physical geometry."""

    text: str
    bbox: BBox
    page: int
    char_index_in_page: int
    line_index: int = 0
    block_index: int = 0

    @property
    def clean_text(self) -> str:
        return self.text.strip()


@dataclass(slots=True)
class VisualLine:
    """A visual line containing ordered tokens."""

    tokens: list[DocumentToken]
    page: int
    line_index: int
    bbox: BBox

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)


@dataclass(slots=True)
class DocumentPage:
    """A document page with physical geometry and tokens."""

    page_number: int  # 1-indexed
    width: float
    height: float
    tokens: list[DocumentToken] = field(default_factory=list)
    lines: list[VisualLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(frozen=True, slots=True)
class ExtractionInput:
    """Input payload describing an extracted field to resolve."""

    field: str
    value: Any
    evidence_text: str | None = None
    field_context: str | None = None
    page_hint: int | None = None


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """The resolved evidence and provenance for an extracted field."""

    field: str
    value: Any
    status: ProvenanceStatus
    page: int | None = None
    bbox: BBox | None = None
    regions: list[BBox] | None = None
    confidence: float = 0.0
    matched_text: str | None = None
    explanation: str = ""

    @property
    def is_grounded(self) -> bool:
        """True only if accepted with a physical coordinate box."""
        return self.status in {
            ProvenanceStatus.EXACT,
            ProvenanceStatus.NORMALIZED,
            ProvenanceStatus.FUZZY,
            ProvenanceStatus.MULTI_REGION,
        } and self.bbox is not None
