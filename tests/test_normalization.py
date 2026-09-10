"""Tests for TonerHound reversible normalization."""

from tonerhound.normalization.normalizers import (
    clean_currency_and_numbers,
    is_date_equal,
    is_number_equal,
    normalize_unicode_and_case,
    normalize_whitespace,
    parse_numeric_value,
)


def test_unicode_and_case_char_mapping():
    # "ﬁ" decomposes to "fi" in NFKC (1 char -> 2 chars)
    # smart quotes become standard quotes
    source = "The ﬁnal ‘quote’"
    norm = normalize_unicode_and_case(source)
    assert norm.text == "the final 'quote'"
    # 'f' at index 4 in norm came from 'ﬁ' at index 4 in source
    # 'i' at index 5 in norm came from 'ﬁ' at index 4 in source
    assert norm.char_map[4] == 4
    assert norm.char_map[5] == 4
    # Span of "final" in norm is [4, 9)
    orig_start, orig_end = norm.span_to_original_range(4, 9)
    assert orig_start == 4
    assert source[orig_start:orig_end] == "ﬁnal"


def test_whitespace_normalization():
    source = "  Total   Amount:   $1,450.00  "
    norm = normalize_unicode_and_case(source)
    cleaned = normalize_whitespace(norm)
    assert cleaned.text == "total amount: $1,450.00"
    # Map back span of "amount"
    start = cleaned.text.index("amount")
    end = start + len("amount")
    orig_start, orig_end = cleaned.span_to_original_range(start, end)
    assert source[orig_start:orig_end] == "Amount"


def test_currency_and_numeric_cleaning():
    assert clean_currency_and_numbers("$1,450.00") == "1450.00"
    assert clean_currency_and_numbers("USD 1,450.00") == "1450.00"
    assert clean_currency_and_numbers("€ 99.50") == "99.50"
    assert clean_currency_and_numbers("(50.00)") == "-50.00"
    assert parse_numeric_value("$1,450.00") == 1450.0
    assert parse_numeric_value("(50.00)") == -50.0
    assert is_number_equal("1450", "USD 1,450.00")
    assert is_number_equal("50", "$50.00")
    assert not is_number_equal("50", "$55.00")


def test_date_parsing_and_equality():
    assert is_date_equal("2026-03-15", "15 March 2026")
    assert is_date_equal("03/15/2026", "March 15, 2026")
    assert is_date_equal("2026-05-01", "May 1, 2026")
    assert not is_date_equal("2026-03-15", "2026-03-16")
