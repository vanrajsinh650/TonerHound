"""TonerHound OCR-Aware Evidence Retrieval & Alignment Module."""

from tonerhound.ocr.aligner import OCRSequenceAligner
from tonerhound.ocr.matcher import OCRMatcher
from tonerhound.ocr.normalizer import (
    OCRNormalizedText,
    OCRNormalizer,
    clean_ocr_noise_span,
    ocr_canonical_numeric,
    ocr_similarity,
)

__all__ = [
    "OCRMatcher",
    "OCRNormalizedText",
    "OCRNormalizer",
    "OCRSequenceAligner",
    "clean_ocr_noise_span",
    "ocr_canonical_numeric",
    "ocr_similarity",
]
