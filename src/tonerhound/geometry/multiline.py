"""Multi-line token assembler for values spanning sequential lines.

Target: Solves MULTI_LINE_SPLIT where values (e.g. addresses, descriptions, narratives)
are split across consecutive visual lines.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import fitz


def normalize_for_match(text: str) -> str:
    """Normalize text for literal visual token matching."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\s\.\,\-\$\%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def assemble_multiline_bbox(
    page: fitz.Page,
    gold_value: str,
    max_lines: int = 5,
    max_gap: float = 0.05,
) -> tuple[float, float, float, float] | None:
    """Assemble multi-line tokens into a unified bounding box."""
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
        norm_box = (x0 / pw, y0 / ph, (x1 - x0) / pw, (y1 - y0) / ph)
        page_tokens.append({
            "text": text,
            "norm": normalize_for_match(text),
            "bbox": norm_box,
            "y": norm_box[1],
            "line": lno,
            "block": bno,
        })

    n_tokens = len(page_tokens)
    first_word = gold_words[0]

    for start_idx in range(n_tokens):
        if page_tokens[start_idx]["norm"] != first_word:
            continue

        matched = [page_tokens[start_idx]]
        gold_idx = 1
        curr_idx = start_idx + 1

        while gold_idx < len(gold_words) and curr_idx < n_tokens:
            tok = page_tokens[curr_idx]
            target = gold_words[gold_idx]

            if tok["norm"] == target:
                matched.append(tok)
                gold_idx += 1
                curr_idx += 1
            else:
                if curr_idx - start_idx > len(gold_words) + 10:
                    break
                curr_idx += 1

        if gold_idx == len(gold_words) and len(matched) >= 2:
            matched.sort(key=lambda t: t["y"])
            valid_gap = True
            for k in range(1, len(matched)):
                dy = matched[k]["y"] - matched[k - 1]["y"]
                if dy > max_gap * 2.0:
                    valid_gap = False
                    break

            if not valid_gap:
                continue

            unique_lines = len(set(round(t["y"], 3) for t in matched))
            if unique_lines > max_lines:
                continue

            min_x = min(t["bbox"][0] for t in matched)
            min_y = min(t["bbox"][1] for t in matched)
            max_x = max(t["bbox"][0] + t["bbox"][2] for t in matched)
            max_y = max(t["bbox"][1] + t["bbox"][3] for t in matched)
            w = max(0.0001, max_x - min_x)
            h = max(0.0001, max_y - min_y)
            return (round(min_x, 6), round(min_y, 6), round(w, 6), round(h, 6))

    return None
