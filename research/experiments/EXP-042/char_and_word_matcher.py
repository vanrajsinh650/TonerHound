"""EXP-042 Phase H: Real Indexing Miss Fix.

Targets REAL_INDEXING_MISS (78,126 fields, realistic gain +0.30 pp).
Implements:
1. Character-level stream lookup from raw character dictionaries.
2. Multi-word sequence matching with gap tolerance and stop-word skipping.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
import fitz

STOP_WORDS = {"of", "and", "&", "the", "a", "an", "for", "in", "on", "at", "to", "by"}


def normalize_for_match(text: str) -> str:
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\s\.\,\-\$\%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def character_level_lookup(
    page: fitz.Page,
    gold_value: str,
) -> tuple[float, float, float, float] | None:
    """Find exact character sequence in page raw character stream."""
    if not gold_value or not page:
        return None

    query = str(gold_value).strip()
    if not query or len(query) < 2:
        return None

    try:
        raw = page.get_text("rawdict")
    except Exception:
        return None

    chars = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    c_text = char.get("c", "")
                    if c_text:
                        chars.append({
                            "char": c_text,
                            "bbox": char["bbox"],
                        })

    if not chars:
        return None

    stream = "".join(c["char"] for c in chars)
    matches = []
    start = 0
    while True:
        idx = stream.find(query, start)
        if idx == -1:
            break
        matches.append(idx)
        start = idx + 1
        if len(matches) > 10:
            break

    if not matches:
        # Also try case-insensitive
        stream_lower = stream.lower()
        query_lower = query.lower()
        idx_lower = stream_lower.find(query_lower)
        if idx_lower != -1:
            matches.append(idx_lower)

    if not matches:
        return None

    # Take first match
    idx = matches[0]
    matched_chars = chars[idx : idx + len(query)]
    if not matched_chars:
        return None

    x0 = min(c["bbox"][0] for c in matched_chars)
    y0 = min(c["bbox"][1] for c in matched_chars)
    x1 = max(c["bbox"][2] for c in matched_chars)
    y1 = max(c["bbox"][3] for c in matched_chars)

    pw = float(page.rect.width) if page.rect.width > 0 else 1.0
    ph = float(page.rect.height) if page.rect.height > 0 else 1.0

    nx0 = x0 / pw
    ny0 = y0 / ph
    nw = (x1 - x0) / pw
    nh = (y1 - y0) / ph

    return (round(nx0, 6), round(ny0, 6), round(max(0.0001, nw), 6), round(max(0.0001, nh), 6))


def multi_word_sequence_match(
    page: fitz.Page,
    gold_value: str,
    max_gap: int = 3,
) -> tuple[float, float, float, float] | None:
    """Match multi-word values with gap tolerance and stop-word skipping."""
    if not gold_value or not page:
        return None

    gold_norm = normalize_for_match(gold_value)
    gold_words = gold_norm.split()
    if len(gold_words) < 2:
        return None

    pw = float(page.rect.width) if page.rect.width > 0 else 1.0
    ph = float(page.rect.height) if page.rect.height > 0 else 1.0

    raw_words = page.get_text("words")
    if not raw_words:
        return None

    page_tokens = []
    for w in raw_words:
        x0, y0, x1, y1, text, bno, lno, wno = w
        page_tokens.append({
            "text": text,
            "norm": normalize_for_match(text),
            "bbox": (x0 / pw, y0 / ph, (x1 - x0) / pw, (y1 - y0) / ph),
            "line": lno,
        })

    n_tokens = len(page_tokens)
    first_word = gold_words[0]

    for i in range(n_tokens):
        if page_tokens[i]["norm"] != first_word:
            continue

        matched = []
        gold_idx = 0
        t_idx = i
        skipped = 0

        while gold_idx < len(gold_words) and t_idx < n_tokens:
            tok = page_tokens[t_idx]
            target_word = gold_words[gold_idx]

            if tok["norm"] == target_word:
                matched.append(tok)
                gold_idx += 1
                t_idx += 1
            elif tok["norm"] in STOP_WORDS:
                t_idx += 1
            elif skipped < max_gap:
                skipped += 1
                t_idx += 1
            else:
                break

        if gold_idx == len(gold_words) and len(matched) >= 2:
            min_x = min(t["bbox"][0] for t in matched)
            min_y = min(t["bbox"][1] for t in matched)
            max_x = max(t["bbox"][0] + t["bbox"][2] for t in matched)
            max_y = max(t["bbox"][1] + t["bbox"][3] for t in matched)
            w = max(0.0001, max_x - min_x)
            h = max(0.0001, max_y - min_y)
            return (round(min_x, 6), round(min_y, 6), round(w, 6), round(h, 6))

    return None
