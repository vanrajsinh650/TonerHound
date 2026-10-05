"""EXP-042 Phase C: Date Literal Variants Fix.

Targets DATE_INDEX_MISS (2,646 fields, realistic gain +0.15 pp).
Generates exact literal string variants (NO semantic normalization) for dates
and matches against page tokens with exact literal fidelity.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
import fitz


def parse_date_flexible(value: str) -> tuple[int, int, int] | None:
    """Parse date into (year, month, day) from common financial formats."""
    if not value:
        return None
    val = str(value).strip()

    # ISO: 2024-01-15
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", val)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # US: 01/15/2024 or 1/15/2024
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", val)
    if m:
        y = int(m.group(3))
        if y < 100:
            y += 2000 if y < 50 else 1900
        return (y, int(m.group(1)), int(m.group(2)))

    # EU: 15-01-2024
    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{2,4})$", val)
    if m:
        y = int(m.group(3))
        if y < 100:
            y += 2000 if y < 50 else 1900
        return (y, int(m.group(2)), int(m.group(1)))

    # Month name: Jan 15, 2024 or 15 Jan 2024
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    m = re.match(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$", val)
    if m:
        mon = month_map.get(m.group(1).lower()[:3])
        if mon:
            return (int(m.group(3)), mon, int(m.group(2)))

    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})$", val)
    if m:
        mon = month_map.get(m.group(2).lower()[:3])
        if mon:
            return (int(m.group(3)), mon, int(m.group(1)))

    return None


def date_literal_variants(value: str) -> list[str]:
    """Generate literal string variants for a date value without semantic normalization."""
    if not value:
        return []

    val_str = str(value).strip()
    variants = {val_str}

    parsed = parse_date_flexible(val_str)
    if parsed is None:
        return [val_str]

    y, m, d = parsed
    if not (1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
        return [val_str]

    # Generate common format variants
    variants.add(f"{y:04d}-{m:02d}-{d:02d}")
    variants.add(f"{m:02d}/{d:02d}/{y:04d}")
    variants.add(f"{m}/{d}/{y:04d}")
    variants.add(f"{d:02d}/{m:02d}/{y:04d}")
    variants.add(f"{d}/{m}/{y:04d}")
    variants.add(f"{d}-{m:02d}-{y:04d}")
    variants.add(f"{d:02d}-{m:02d}-{y:04d}")
    variants.add(f"{m:02d}-{d:02d}-{y:04d}")
    variants.add(f"{m}-{d}-{y:04d}")
    variants.add(f"{y:04d}/{m:02d}/{d:02d}")
    variants.add(f"{y%100:02d}-{m:02d}-{d:02d}")
    variants.add(f"{m:02d}/{d:02d}/{y%100:02d}")
    variants.add(f"{m}/{d}/{y%100:02d}")
    variants.add(f"{d:02d}/{m:02d}/{y%100:02d}")
    variants.add(f"{d}/{m}/{y%100:02d}")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_full = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    if 1 <= m <= 12:
        m_short = month_names[m - 1]
        m_long = month_full[m - 1]
        variants.add(f"{m_short} {d}, {y:04d}")
        variants.add(f"{m_long} {d}, {y:04d}")
        variants.add(f"{m_short}. {d}, {y:04d}")
        variants.add(f"{d} {m_short} {y:04d}")
        variants.add(f"{d} {m_short.upper()} {y:04d}")
        variants.add(f"{d} {m_long} {y:04d}")
        variants.add(f"{m_short} {d} {y:04d}")

    return sorted(list(variants))


def ground_date_literal(
    page: fitz.Page,
    gold_value: str,
    gold_bbox: Sequence[float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Find exact literal date match on page tokens."""
    if not gold_value or not page:
        return None

    variants = date_literal_variants(gold_value)
    if not variants:
        return None

    pw = float(page.rect.width) if page.rect.width > 0 else 1.0
    ph = float(page.rect.height) if page.rect.height > 0 else 1.0

    raw_words = page.get_text("words")
    candidates = []

    # Check single tokens
    for w in raw_words:
        x0, y0, x1, y1, text, bno, lno, wno = w
        t_clean = text.strip()
        for v in variants:
            if t_clean == v or t_clean.rstrip(".,;:") == v:
                b = (x0 / pw, y0 / ph, (x1 - x0) / pw, (y1 - y0) / ph)
                candidates.append((b, len(v)))

    # Also check 2-word or 3-word spans for dates like "Jan 15, 2024"
    if not candidates:
        n_w = len(raw_words)
        for i in range(n_w):
            for span_len in (2, 3, 4):
                if i + span_len > n_w:
                    break
                span_words = raw_words[i : i + span_len]
                # Same line check
                if any(w[6] != span_words[0][6] for w in span_words):
                    continue
                span_text = " ".join(w[4].strip() for w in span_words)
                for v in variants:
                    if span_text == v or span_text.rstrip(".,;:") == v:
                        x0 = min(w[0] for w in span_words)
                        y0 = min(w[1] for w in span_words)
                        x1 = max(w[2] for w in span_words)
                        y1 = max(w[3] for w in span_words)
                        b = (x0 / pw, y0 / ph, (x1 - x0) / pw, (y1 - y0) / ph)
                        candidates.append((b, len(v)))

    if not candidates:
        return None

    if len(candidates) == 1:
        return (
            round(candidates[0][0][0], 6),
            round(candidates[0][0][1], 6),
            round(candidates[0][0][2], 6),
            round(candidates[0][0][3], 6),
        )

    # If multiple candidates, pick closest to gold_bbox if available, else longest match
    if gold_bbox and len(gold_bbox) == 4:
        gx, gy = gold_bbox[0], gold_bbox[1]
        best = min(candidates, key=lambda c: (c[0][0] - gx) ** 2 + (c[0][1] - gy) ** 2)
        return (
            round(best[0][0], 6),
            round(best[0][1], 6),
            round(best[0][2], 6),
            round(best[0][3], 6),
        )

    best = max(candidates, key=lambda c: c[1])
    return (
        round(best[0][0], 6),
        round(best[0][1], 6),
        round(best[0][2], 6),
        round(best[0][3], 6),
    )
