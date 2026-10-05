"""EXP-038 Fix 4: Multi-Format Date Normalizer and Indexer.

Extends deterministic date parsing to non-standard temporal formats:
- YYYY-MMM-DD / DD-MMM-YYYY (e.g. 2021-OCT-15, 15-OCT-2021)
- DD-MM-YYYY / DD/MM/YYYY (e.g. 15-10-2021, 15/10/2021)
- MM-DD-YY / MM/DD/YY (e.g. 10-17-08, 10/17/08)
- YYYY/MM/DD (e.g. 2021/10/15)
- Normalizes all parsed dates to canonical ISO 'YYYY-MM-DD'.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Sequence

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Regex patterns for non-standard dates
RE_YYYY_MMM_DD = re.compile(r"^(19\d\d|20\d\d)[-_/]([a-zA-Z]{3,9})[-_/](0?[1-9]|[12]\d|3[01])$", re.IGNORECASE)
RE_DD_MMM_YYYY = re.compile(r"^(0?[1-9]|[12]\d|3[01])[-_/]([a-zA-Z]{3,9})[-_/](19\d\d|20\d\d)$", re.IGNORECASE)
RE_DD_MM_YYYY = re.compile(r"^(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](19\d\d|20\d\d)$")
RE_MM_DD_YY = re.compile(r"^(0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])[-/.](\d{2})$")
RE_YYYY_MM_DD_SLASH = re.compile(r"^(19\d\d|20\d\d)/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])$")


def parse_extended_date(text: str) -> str | None:
    """Parse text into ISO YYYY-MM-DD string if it matches any extended format."""
    clean = text.strip()
    if not clean or len(clean) > 30:
        return None

    # 1. YYYY-MMM-DD (2021-OCT-15)
    m = RE_YYYY_MMM_DD.match(clean)
    if m:
        y, mon_str, d = m.groups()
        mon = MONTH_MAP.get(mon_str.lower())
        if mon:
            try:
                dt = datetime(int(y), mon, int(d))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    # 2. DD-MMM-YYYY (15-OCT-2021)
    m = RE_DD_MMM_YYYY.match(clean)
    if m:
        d, mon_str, y = m.groups()
        mon = MONTH_MAP.get(mon_str.lower())
        if mon:
            try:
                dt = datetime(int(y), mon, int(d))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    # 3. YYYY/MM/DD (2021/10/15)
    m = RE_YYYY_MM_DD_SLASH.match(clean)
    if m:
        y, mon, d = m.groups()
        try:
            dt = datetime(int(y), int(mon), int(d))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 4. DD-MM-YYYY (15-10-2021)
    m = RE_DD_MM_YYYY.match(clean)
    if m:
        d, mon, y = m.groups()
        try:
            dt = datetime(int(y), int(mon), int(d))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 5. MM-DD-YY (10-17-08) -> assume 20YY if <= 50 else 19YY
    m = RE_MM_DD_YY.match(clean)
    if m:
        mon, d, yy = m.groups()
        y_int = int(yy)
        y_full = 2000 + y_int if y_int <= 50 else 1900 + y_int
        try:
            dt = datetime(y_full, int(mon), int(d))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None
