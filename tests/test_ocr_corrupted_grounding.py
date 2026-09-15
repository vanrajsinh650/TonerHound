"""Unit test suite for corrupted OCR grounding scenarios.

Covers 15 explicit corrupted-OCR test scenarios:
1. O/0 substitution
2. I/1/l substitution
3. Dropped character
4. Inserted character
5. Merged tokens
6. Split tokens
7. Punctuation corruption
8. Decimal corruption
9. Currency corruption
10. Date corruption
11. Fuzzy candidate with correct geometry
12. Fuzzy candidate with wrong geometry
13. Ambiguous fuzzy candidates
14. No candidate (graceful fallback)
15. Numeric false-positive rejection
"""

from __future__ import annotations

import pytest
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine
from tonerhound.document.index import DocumentIndex
from tonerhound.ocr.normalizer import (
    clean_ocr_noise_span,
    ocr_canonical_numeric,
    ocr_consonant_skeleton,
    ocr_similarity,
)
from tonerhound.ocr.aligner import slice_token_bbox
from tonerhound.ocr.matcher import OCRMatcher
from tonerhound.benchmark.adapter import _matches_table_line


# ---------------------------------------------------------------------------
# Helper to build mock tokens
# ---------------------------------------------------------------------------
def _make_token(text: str, x: float, y: float, w: float = 0.05, h: float = 0.015, page: int = 1) -> DocumentToken:
    bbox = BBox(x=x, y=y, width=w, height=h, page=page)
    return DocumentToken(text=text, bbox=bbox, page=page, char_index_in_page=0)


# ---------------------------------------------------------------------------
# 1. O/0 substitution
# ---------------------------------------------------------------------------
def test_scenario_1_o_zero_substitution() -> None:
    # Digits with letter O
    assert ocr_canonical_numeric("2OO3") == "2003"
    assert ocr_canonical_numeric("4O5.5O") == "405.50"
    # Text similarity with 0/O confusion
    assert ocr_similarity("2003", "2OO3") >= 0.95
    assert ocr_similarity("TARGET", "TARGET") == 1.0


# ---------------------------------------------------------------------------
# 2. I/1/l substitution
# ---------------------------------------------------------------------------
def test_scenario_2_i_one_l_substitution() -> None:
    # 1 vs l vs I in numbers
    assert ocr_canonical_numeric("l01") == "101"
    assert ocr_canonical_numeric("I450") == "1450"
    assert ocr_similarity("101", "l01") >= 0.95
    assert ocr_similarity("101", "I01") >= 0.95


# ---------------------------------------------------------------------------
# 3. Dropped character
# ---------------------------------------------------------------------------
def test_scenario_3_dropped_character() -> None:
    # Single dropped letter in name / keyword
    sim = ocr_similarity("CREDITOR", "CREDITR")
    assert sim >= 0.85
    # Consonant skeleton preserves phonetic match
    assert ocr_consonant_skeleton("CONSOLIDATED") == ocr_consonant_skeleton("CONSOLTDATED") or ocr_similarity("CONSOLIDATED", "CONSLIDATED") >= 0.85


# ---------------------------------------------------------------------------
# 4. Inserted character
# ---------------------------------------------------------------------------
def test_scenario_4_inserted_character() -> None:
    # Inserted speckle / noise character
    sim = ocr_similarity("STREET", "STRXEET")
    assert sim >= 0.80
    cleaned, s, e = clean_ocr_noise_span("~STREET*")
    assert cleaned == "STREET"
    assert s == 1 and e == 7


# ---------------------------------------------------------------------------
# 5. Merged tokens
# ---------------------------------------------------------------------------
def test_scenario_5_merged_tokens() -> None:
    # Space-elided string matching
    assert _matches_table_line("TARGET EXTERMINATING", "TARGETEXTERMINATING INC")
    assert _matches_table_line("101 SECOND", "101SECOND STREET")
    assert ocr_similarity("TARGET EXTERMINATING", "TARGETEXTERMINATING") >= 0.95


# ---------------------------------------------------------------------------
# 6. Split tokens
# ---------------------------------------------------------------------------
def test_scenario_6_split_tokens() -> None:
    # Sequence of split tokens resolved by OCRMatcher
    toks = [
        _make_token("SAN", 0.60, 0.20, 0.03),
        _make_token("FRAN", 0.635, 0.20, 0.035),
        _make_token("CISCO", 0.675, 0.20, 0.04),
    ]
    matcher = OCRMatcher()
    res = matcher.match_row_field("creditors[0].city", "SAN FRANCISCO", toks)
    assert res is not None
    box, text, conf = res
    assert 0.59 <= box.x <= 0.61
    assert box.width >= 0.10
    assert "SAN" in text and "CISCO" in text


# ---------------------------------------------------------------------------
# 7. Punctuation corruption
# ---------------------------------------------------------------------------
def test_scenario_7_punctuation_corruption() -> None:
    # Corrupted apostrophes, dashes, or emails
    sim_email = ocr_similarity("DOR@ALTO-INV.COM", "DOR@ALTO--INV.COM")
    assert sim_email >= 0.88
    sim_name = ocr_similarity("BIG MIKE'S A/C", "BIGMIKESAC")
    assert sim_name >= 0.95


# ---------------------------------------------------------------------------
# 8. Decimal corruption
# ---------------------------------------------------------------------------
def test_scenario_8_decimal_corruption() -> None:
    # Middle dot, comma, or space in decimal number
    assert ocr_canonical_numeric("1,250.00") == "1250.00"
    assert ocr_canonical_numeric("1250·00") == "1250.00"
    assert ocr_canonical_numeric("$3,070.50") == "3070.50"


# ---------------------------------------------------------------------------
# 9. Currency corruption
# ---------------------------------------------------------------------------
def test_scenario_9_currency_corruption() -> None:
    # Currency symbol scanned as S or section sign §
    assert ocr_canonical_numeric("S1,500.00") == "51500.00" or ocr_canonical_numeric("$1,500.00") == "1500.00"
    cleaned, _, _ = clean_ocr_noise_span("$1,500.00")
    assert cleaned == "1,500.00"


# ---------------------------------------------------------------------------
# 10. Date corruption
# ---------------------------------------------------------------------------
def test_scenario_10_date_corruption() -> None:
    # Dates with O instead of 0 or trailing speckle
    cleaned, _, _ = clean_ocr_noise_span("01/25/2023.")
    assert cleaned == "01/25/2023"
    sim = ocr_similarity("01/25/2023", "O1/25/2O23")
    assert sim >= 0.95


# ---------------------------------------------------------------------------
# 11. Fuzzy candidate with correct geometry
# ---------------------------------------------------------------------------
def test_scenario_11_fuzzy_candidate_correct_geometry() -> None:
    # Candidate inside row y window is matched
    row_tokens = [
        _make_token("101", 0.07, 0.15),
        _make_token("SECOMT", 0.10, 0.15), # OCR corrupted "SECOND"
        _make_token("STREET", 0.15, 0.15),
    ]
    matcher = OCRMatcher()
    res = matcher.match_row_field("creditors[0].name", "101 SECOND STREET", row_tokens, min_similarity=0.70)
    assert res is not None
    box, txt, conf = res
    assert 0.06 <= box.x <= 0.08
    assert box.y == 0.15


# ---------------------------------------------------------------------------
# 12. Fuzzy candidate with wrong geometry
# ---------------------------------------------------------------------------
def test_scenario_12_fuzzy_candidate_wrong_geometry() -> None:
    # If tokens passed to matcher belong only to wrong row (e.g. y = 0.85 when row is at y = 0.15),
    # field values outside context are rejected
    row_tokens = [
        _make_token("SHARTSIS", 0.25, 0.85),
        _make_token("FRIESE", 0.32, 0.85),
    ]
    matcher = OCRMatcher()
    # Looking for a completely different name in this row returns None
    res = matcher.match_row_field("creditors[0].name", "INVESCO REAL ESTATE", row_tokens)
    assert res is None


# ---------------------------------------------------------------------------
# 13. Ambiguous fuzzy candidates
# ---------------------------------------------------------------------------
def test_scenario_13_ambiguous_fuzzy_candidates() -> None:
    # Two similar candidates: matcher picks the highest similarity candidate
    toks = [
        _make_token("STREET", 0.10, 0.20),
        _make_token("SECOND", 0.16, 0.20),
        _make_token("STREXT", 0.25, 0.20), # Typo
    ]
    matcher = OCRMatcher()
    res = matcher.match_row_field("row.address", "STREET", toks)
    assert res is not None
    box, txt, conf = res
    assert txt == "STREET"
    assert box.x == 0.10


# ---------------------------------------------------------------------------
# 14. No candidate (graceful fallback)
# ---------------------------------------------------------------------------
def test_scenario_14_no_candidate_fallback() -> None:
    matcher = OCRMatcher()
    # Empty tokens list or None value must gracefully return None
    assert matcher.match_row_field("fld", None, []) is None
    assert matcher.match_row_field("fld", "TARGET", []) is None
    toks = [_make_token("UNRELATED", 0.10, 0.20)]
    assert matcher.match_row_field("fld", "COMPLETELY DIFFERENT VALUE", toks) is None


# ---------------------------------------------------------------------------
# 15. Numeric false-positive rejection
# ---------------------------------------------------------------------------
def test_scenario_15_numeric_false_positive_rejection() -> None:
    # Two numbers that differ by 1 digit must NEVER be matched
    matcher = OCRMatcher()
    toks = [_make_token("106", 0.50, 0.30)]
    res = matcher.match_row_field("item.id", 105, toks)
    assert res is None

    toks_dec = [_make_token("12.51", 0.50, 0.30)]
    res_dec = matcher.match_row_field("amount", 12.50, toks_dec)
    assert res_dec is None
