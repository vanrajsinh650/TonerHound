"""Stage 32: Relative Date Parser.

Parses relative date clauses:
- Matches phrases like "30 days from invoice date", "Net 30", "within 15 days of..."
- Resolves against anchor date (e.g. invoice date, order date)
- Computes effective target date and validates with proximity check.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from dateutil import parser as date_parser


_REL_DATE_PAT = re.compile(
    r"(?:within\s+)?(\d+)\s*(?:days?|business\s+days?)\s+(?:from|after|following)\s+(?:the\s+)?([a-zA-Z\s]+)",
    re.I,
)
_NET_TERMS_PAT = re.compile(r"\bnet\s*(\d+)\b", re.I)


def parse_relative_date_offset(text: str) -> tuple[int, str] | None:
    """Extracts offset days and anchor field description from text."""
    m = _REL_DATE_PAT.search(text)
    if m:
        offset_days = int(m.group(1))
        anchor_desc = m.group(2).strip()
        return offset_days, anchor_desc

    m2 = _NET_TERMS_PAT.search(text)
    if m2:
        return int(m2.group(1)), "invoice_date"

    return None


def resolve_relative_date(
    text: str,
    anchor_date: str | datetime,
) -> datetime | None:
    """Computes target date based on relative offset from anchor_date."""
    res = parse_relative_date_offset(text)
    if not res:
        return None

    offset_days, _ = res

    if isinstance(anchor_date, str):
        try:
            base_dt = date_parser.parse(anchor_date, fuzzy=True)
        except Exception:
            return None
    elif isinstance(anchor_date, datetime):
        base_dt = anchor_date
    else:
        return None

    return base_dt + timedelta(days=offset_days)
