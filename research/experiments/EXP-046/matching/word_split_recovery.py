"""Stage 34: Word Split Recovery (Non-Hyphenated Line Breaks).

For words accidentally split across lines without any hyphen character:
- Evaluates last token of line N + first token of line N+1
- Uses the document's internal vocabulary (token frequency counter) as ground truth dictionary
- Joins the candidate tokens only if the concatenated word exists elsewhere in the document
- Returns normalized COCO bounding box [x, y, w, h] or None.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence
import fitz


def build_document_vocabulary(doc: fitz.Document) -> Counter[str]:
    """Builds frequency dictionary of all full words in document."""
    vocab: Counter[str] = Counter()
    for page in doc:
        text = page.get_text()
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        vocab.update(tokens)
    return vocab


def recover_split_words(
    page: fitz.Page,
    doc_vocab: Counter[str],
    target_value: str,
) -> tuple[float, float, float, float] | None:
    """Detects and repairs mid-word splits across consecutive lines."""
    if not target_value:
        return None

    target_clean = re.sub(r"\s+", "", str(target_value)).lower()
    if len(target_clean) < 4:
        return None

    words = page.get_text("words")
    if not words:
        return None

    pw = float(page.rect.width) if page.rect.width > 0 else 1.0
    ph = float(page.rect.height) if page.rect.height > 0 else 1.0

    n_words = len(words)

    for i in range(n_words - 1):
        x0, y0, x1, y1, text_a, block_a, line_a, word_no_a = words[i]

        # Clean punctuation from text_a
        clean_a = re.sub(r"[^\w]", "", text_a).lower()
        if len(clean_a) < 2:
            continue

        line_h = max(1.0, y1 - y0)

        for j in range(i + 1, min(i + 8, n_words)):
            nx0, ny0, nx1, ny1, text_b, block_b, line_b, word_no_b = words[j]

            # Must be different line below
            if line_b == line_a:
                continue
            if ny0 - y1 > 1.8 * line_h:
                break

            # Must be first word of next line
            if word_no_b != 0:
                continue

            clean_b = re.sub(r"[^\w]", "", text_b).lower()
            if len(clean_b) < 2:
                continue

            candidate_word = clean_a + clean_b

            # Vocabulary verification: must exist in document at least once elsewhere
            # Or exactly match target_clean
            if candidate_word == target_clean and doc_vocab.get(candidate_word, 0) >= 1:
                ux0 = min(x0, nx0) / pw
                uy0 = min(y0, ny0) / ph
                ux1 = max(x1, nx1) / pw
                uy1 = max(y1, ny1) / ph
                uw = max(0.0001, ux1 - ux0)
                uh = max(0.0001, uy1 - uy0)
                return (round(ux0, 6), round(uy0, 6), round(uw, 6), round(uh, 6))

            break

    return None
