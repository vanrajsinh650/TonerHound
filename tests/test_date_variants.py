"""Unit tests for date literal variants."""

from __future__ import annotations

import fitz
import pytest
from tonerhound.matching.date_variants import (
    date_literal_variants,
    ground_date_literal,
    parse_date_flexible,
)


def test_parse_date_flexible() -> None:
    assert parse_date_flexible("2024-01-15") == (2024, 1, 15)
    assert parse_date_flexible("01/15/2024") == (2024, 1, 15)
    assert parse_date_flexible("Jan 15, 2024") == (2024, 1, 15)


def test_date_literal_variants() -> None:
    vars_list = date_literal_variants("2024-01-15")
    assert "2024-01-15" in vars_list
    assert "01/15/2024" in vars_list
    assert "Jan 15, 2024" in vars_list


def test_ground_date_literal_fitz() -> None:
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text(fitz.Point(100, 100), "01/15/2024")

    res = ground_date_literal(page, "2024-01-15")
    doc.close()

    assert res is not None
    assert len(res) == 4
    assert res[2] > 0
    assert res[3] > 0
