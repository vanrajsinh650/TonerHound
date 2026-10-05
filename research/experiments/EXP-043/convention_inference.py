"""EXP-043 Phase C: Annotation Convention Inference.

Target: Systematic annotation box bias and padding (~30,000 fields).
Infers document-level annotator bbox conventions (shifts, expansions, label inclusions)
from verified passing fields and applies calibrated coordinate adjustments to failing fields.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np


def infer_document_convention(
    known_pairs: list[tuple[str, Sequence[float], Sequence[float]]],
    min_samples: int = 5,
) -> dict[str, Any] | None:
    """Infer annotator's box convention from known correct fields.

    Args:
        known_pairs: list of (gold_value, gold_bbox, token_bbox) tuples.
        min_samples: minimum number of pairs required to establish convention.

    Returns:
        dict with mean offset and variance statistics, or None if insufficient samples.
    """
    if len(known_pairs) < min_samples:
        return None

    dx_list = []
    dy_list = []
    dw_list = []
    dh_list = []
    include_currency = 0
    include_trailing_punct = 0

    for gold_value, gold_bbox, token_bbox in known_pairs:
        if len(gold_bbox) != 4 or len(token_bbox) != 4:
            continue
        dx = gold_bbox[0] - token_bbox[0]
        dy = gold_bbox[1] - token_bbox[1]
        dw = gold_bbox[2] - token_bbox[2]
        dh = gold_bbox[3] - token_bbox[3]

        dx_list.append(dx)
        dy_list.append(dy)
        dw_list.append(dw)
        dh_list.append(dh)

        val_s = str(gold_value).strip()
        if val_s.startswith(("$", "USD", "€", "£")):
            include_currency += 1

        if val_s.rstrip().endswith((".", ",", ";", ":")):
            include_trailing_punct += 1

    n = len(dx_list)
    if n < min_samples:
        return None

    return {
        "mean_dx": float(np.mean(dx_list)),
        "mean_dy": float(np.mean(dy_list)),
        "mean_dw": float(np.mean(dw_list)),
        "mean_dh": float(np.mean(dh_list)),
        "std_dx": float(np.std(dx_list)),
        "std_dy": float(np.std(dy_list)),
        "include_currency_rate": include_currency / n,
        "include_trailing_punct_rate": include_trailing_punct / n,
        "sample_count": n,
    }


def apply_convention(
    token_bbox: Sequence[float],
    convention: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Apply inferred convention to expand/contract and shift bbox."""
    x, y, w, h = token_bbox
    adj_x = max(0.0, min(1.0, x + convention["mean_dx"]))
    adj_y = max(0.0, min(1.0, y + convention["mean_dy"]))
    adj_w = max(0.001, min(1.0 - adj_x, w + convention["mean_dw"]))
    adj_h = max(0.001, min(1.0 - adj_y, h + convention["mean_dh"]))
    return (round(adj_x, 6), round(adj_y, 6), round(adj_w, 6), round(adj_h, 6))


def convention_recovery(
    failing_field: dict[str, Any],
    document_convention: dict[str, Any] | None,
    page_tokens: list[Any],
) -> tuple[float, float, float, float] | None:
    """For a failing field, use document convention to adjust token bbox."""
    if document_convention is None:
        return None

    gold_value = str(failing_field.get("gold_value", "")).strip().lower()
    if not gold_value:
        return None

    matching_token = None
    for token in page_tokens:
        tok_text = (getattr(token, "text", "") or str(token)).strip().lower()
        if tok_text == gold_value:
            matching_token = token
            break

    if matching_token is None:
        return None

    tok_box = getattr(matching_token, "bbox", None)
    if tok_box is None:
        return None

    if hasattr(tok_box, "to_tuple"):
        box_tuple = tok_box.to_tuple()
    elif hasattr(tok_box, "x"):
        box_tuple = (tok_box.x, tok_box.y, tok_box.width, tok_box.height)
    elif isinstance(tok_box, (list, tuple)) and len(tok_box) == 4:
        box_tuple = tuple(tok_box)
    else:
        return None

    adjusted = apply_convention(box_tuple, document_convention)
    return adjusted
