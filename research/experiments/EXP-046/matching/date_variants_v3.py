"""Stage 31: Date Literal Variants v3.

Expands date string variations to 50 canonical and international formats:
- All international numeric combinations (YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, YYYY.MM.DD)
- Textual month names (Jan, January, JAN, JANUARY, Sept, Sep, September)
- Diverse delimiters (hyphen -, slash /, dot ., space, comma)
- Ordinal suffixes (1st, 2nd, 3rd, 4th, 21st, 22nd, 31st)
- Two-digit vs four-digit years (24 vs 2024)
"""

from __future__ import annotations

import re
from datetime import datetime
from dateutil import parser as date_parser


MONTH_NAMES = [
    ("January", "Jan"),
    ("February", "Feb"),
    ("March", "Mar"),
    ("April", "Apr"),
    ("May", "May"),
    ("June", "Jun"),
    ("July", "Jul"),
    ("August", "Aug"),
    ("September", "Sep"),
    ("October", "Oct"),
    ("November", "Nov"),
    ("December", "Dec"),
]


def _ordinal_suffix(day: int) -> str:
    if 11 <= (day % 100) <= 13:
        return f"{day}th"
    last = day % 10
    if last == 1:
        return f"{day}st"
    if last == 2:
        return f"{day}nd"
    if last == 3:
        return f"{day}rd"
    return f"{day}th"


def generate_date_variants_v3(date_input: str | datetime) -> list[str]:
    """Generates up to 50 textual format variants of a given date."""
    if not date_input:
        return []

    dt: datetime | None = None
    if isinstance(date_input, datetime):
        dt = date_input
    else:
        s = str(date_input).strip()
        # Clean ordinal suffixes like 1st, 2nd
        s_clean = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", s, flags=re.I)
        try:
            dt = date_parser.parse(s_clean, fuzzy=True)
        except Exception:
            return [s]

    if dt is None:
        return [str(date_input)]

    d = dt.day
    m = dt.month
    y = dt.year
    yy = y % 100

    variants: set[str] = set()

    # Original representations
    variants.add(str(date_input))

    full_m, short_m = MONTH_NAMES[m - 1]
    ord_d = _ordinal_suffix(d)

    # 1. Numeric formats (ISO and variations)
    for sep in ["-", "/", ".", " "]:
        # YYYY-MM-DD
        variants.add(f"{y:04d}{sep}{m:02d}{sep}{d:02d}")
        variants.add(f"{y:04d}{sep}{m}{sep}{d}")
        # DD-MM-YYYY
        variants.add(f"{d:02d}{sep}{m:02d}{sep}{y:04d}")
        variants.add(f"{d}{sep}{m}{sep}{y:04d}")
        # MM-DD-YYYY
        variants.add(f"{m:02d}{sep}{d:02d}{sep}{y:04d}")
        variants.add(f"{m}{sep}{d}{sep}{y:04d}")
        # With 2-digit year
        variants.add(f"{m:02d}{sep}{d:02d}{sep}{yy:02d}")
        variants.add(f"{d:02d}{sep}{m:02d}{sep}{yy:02d}")

    # 2. Textual month formats
    for m_str in [full_m, full_m.upper(), full_m.lower(), short_m, short_m.upper(), short_m.lower()]:
        # "Jan 15, 2024", "January 15, 2024"
        variants.add(f"{m_str} {d}, {y}")
        variants.add(f"{m_str} {d:02d}, {y}")
        variants.add(f"{m_str} {d} {y}")
        variants.add(f"{m_str} {d}, {yy:02d}")
        variants.add(f"{m_str}. {d}, {y}")

        # "15 Jan 2024", "15 January 2024"
        variants.add(f"{d} {m_str} {y}")
        variants.add(f"{d:02d} {m_str} {y}")
        variants.add(f"{d} {m_str}, {y}")
        variants.add(f"{d}-{m_str}-{y}")
        variants.add(f"{d:02d}-{m_str}-{y}")
        variants.add(f"{d}/{m_str}/{y}")

        # Ordinal: "January 15th, 2024", "15th January 2024"
        variants.add(f"{m_str} {ord_d}, {y}")
        variants.add(f"{m_str} {ord_d} {y}")
        variants.add(f"{ord_d} {m_str} {y}")
        variants.add(f"{ord_d} of {m_str}, {y}")
        variants.add(f"{ord_d} of {m_str} {y}")

    return sorted(list(variants))
