"""Test for microscope boolean vs numeric distinction regression protection (Section 9)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
from research.observer.field_classifier import (
    FailureMicroscopeClassifier,
    PRODUCTION_FAILURE_CLASSES,
    AUDIT_ONLY_CLASSES,
    ALL_TAXONOMY_CLASSES,
)


def test_boolean_distinction_no_numeric_confusion():
    """Verify bool, int, float are distinctly recognized and never confused."""
    classifier = FailureMicroscopeClassifier()

    # Booleans MUST be identified as boolean
    assert classifier.is_boolean_target(True) is True
    assert classifier.is_boolean_target(False) is True
    assert classifier.is_boolean_target("True") is True
    assert classifier.is_boolean_target("false") is True
    assert classifier.is_boolean_target("yes", field_path="eligible_yes_box") is True

    # Ints MUST NOT be identified as boolean
    assert classifier.is_boolean_target(0) is False
    assert classifier.is_boolean_target(1) is False
    assert classifier.is_boolean_target(100) is False
    assert classifier.is_boolean_target(-5) is False

    # Floats MUST NOT be identified as boolean
    assert classifier.is_boolean_target(0.0) is False
    assert classifier.is_boolean_target(1.0) is False
    assert classifier.is_boolean_target(99.4) is False
    assert classifier.is_boolean_target(-12.34) is False


def test_taxonomy_completeness_and_separation():
    """Verify taxonomy structure and separation between production and audit classes."""
    # Ensure audit-only classes are segregated
    for audit_cls in AUDIT_ONLY_CLASSES:
        assert audit_cls not in PRODUCTION_FAILURE_CLASSES

    assert "AUDIT_CLASSIFICATION_ERROR" in AUDIT_ONLY_CLASSES
    assert "ALREADY_RESOLVED" in AUDIT_ONLY_CLASSES

    # Ensure required production classes are present
    required_prod = [
        "NON_TEXT_BOOLEAN_GROUNDING",
        "NO_TEXT_AT_GOLD_REGION",
        "DATE_INDEX_MISS",
        "REAL_INDEXING_MISS",
        "NORMALIZATION_MISMATCH",
        "VERIFICATION_REJECTION",
        "BBOX_RECONSTRUCTION",
        "HYPHENATION",
        "PAGE_ROUTING",
        "WRONG_ROW",
        "WRONG_COLUMN",
        "WRONG_PAGE",
        "BBOX_TOO_NARROW",
        "BBOX_TOO_WIDE",
        "TOKEN_SLICING",
        "TOP_K_TRUNCATION",
        "DEDUPLICATION_COLLAPSE",
        "MULTI_LINE_SPLIT",
        "OTHER",
    ]
    for req in required_prod:
        assert req in PRODUCTION_FAILURE_CLASSES
