"""EXP-029 Feature Extraction & Candidate Dataset Builder.

Constructs leakage-free candidate-level features across 6 families:
1. TEXT: exact, normalized, fuzzy, token overlap, numeric similarity, length ratios
2. STRUCTURAL: table detection, field depth, peer distribution, same-line/col consensus
3. GEOMETRIC: bbox coordinates, area, aspect ratio, text density, margins, relative distance to rank-1
4. DOCUMENT: page priors, split indicators, document taxonomy
5. SOURCE: match tiers (exact, numeric, date, boolean, fuzzy, recovered, OCR)
6. RANKING: baseline rank, reciprocal rank, rank indicators, pool cardinality

Strict Leakage Prevention:
Zero gold evidence (bboxes, pages, ground-truth IoU) is exposed to the feature vector.
The label y is binary: 1 if cand['best_iou'] >= 0.50 else 0 (training only).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    # TEXT (7)
    "text_exact_match",
    "text_norm_match",
    "text_fuzz_ratio",
    "text_token_overlap",
    "text_numeric_similarity",
    "text_char_len_ratio",
    "text_len_diff",
    # STRUCTURAL (8)
    "struct_is_table",
    "struct_field_depth",
    "struct_same_page_peers",
    "struct_same_line_peers",
    "struct_same_col_peers",
    "struct_same_page_as_r1",
    "struct_same_line_as_r1",
    "struct_same_col_as_r1",
    # GEOMETRIC (15)
    "geom_bbox_x",
    "geom_bbox_y",
    "geom_bbox_w",
    "geom_bbox_h",
    "geom_bbox_area",
    "geom_aspect_ratio",
    "geom_text_density",
    "geom_dist_left",
    "geom_dist_right",
    "geom_dist_top",
    "geom_dist_bottom",
    "geom_dx_from_r1",
    "geom_dy_from_r1",
    "geom_dist_from_r1",
    "geom_area_ratio_r1",
    # DOCUMENT (12)
    "doc_page",
    "doc_reciprocal_page",
    "doc_is_page_1",
    "doc_is_page_2",
    "doc_is_page_3_plus",
    "doc_split_short",
    "doc_split_medium",
    "doc_split_long",
    "doc_is_sec",
    "doc_is_tax",
    "doc_is_regulatory",
    "doc_is_schedule_table",
    # SOURCE (7)
    "source_exact",
    "source_numeric",
    "source_date",
    "source_boolean",
    "source_fuzzy",
    "source_recovered",
    "source_is_ocr",
    # RANKING (8)
    "rank_orig",
    "rank_reciprocal",
    "rank_is_1",
    "rank_is_2",
    "rank_is_3",
    "rank_diff_from_r1",
    "pool_candidate_count",
    "pool_inv_candidate_count",
]


def _clean_num(val_str: str) -> float | None:
    s = re.sub(r"[^\d.-]", "", val_str)
    if not s or s == "-" or s == ".":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_candidate_features_and_labels(
    field_record: dict[str, Any],
) -> tuple[list[list[float]], list[int], list[dict[str, Any]]]:
    """Extract leakage-free features and training labels for all candidates in a field.
    
    Returns:
        (feature_matrix, binary_labels, candidate_metadata)
    """
    pool_raw = field_record.get("candidate_pool")
    if not pool_raw:
        return [], [], []

    if isinstance(pool_raw, str):
        try:
            pool = json.loads(pool_raw)
        except Exception:
            return [], [], []
    else:
        pool = pool_raw

    if not pool or len(pool) < 2:
        return [], [], []

    query_val = str(field_record.get("gold_value") or "")
    q_norm = query_val.strip().lower()
    q_tokens = set(q_norm.split())
    q_num = _clean_num(query_val)

    field_path = str(field_record.get("field_path") or "")
    is_table = 1.0 if ("[" in field_path and "]" in field_path) else 0.0
    field_depth = float(field_path.count(".") + field_path.count("["))

    split = str(field_record.get("split") or "").lower()
    doc_type = str(field_record.get("document_type") or "").lower()

    split_short = 1.0 if split == "short" else 0.0
    split_med = 1.0 if split == "medium" else 0.0
    split_long = 1.0 if split == "long" else 0.0

    is_sec = 1.0 if ("sec" in doc_type or "13f" in doc_type) else 0.0
    is_tax = 1.0 if any(k in doc_type for k in ("1040", "w-2", "1099", "k-1", "tax")) else 0.0
    is_reg = 1.0 if ("rrc" in doc_type or "regulatory" in doc_type) else 0.0
    is_sched = 1.0 if any(k in doc_type for k in ("schedule", "register", "matrix", "table", "listing", "creditor")) else 0.0

    total_cands = len(pool)
    inv_total_cands = 1.0 / max(1, total_cands)

    # Rank 1 anchor for relative geometry
    r1 = pool[0]
    r1_page = r1.get("page", 1)
    r1_bbox = r1.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    r1_x = float(r1_bbox[0])
    r1_y = float(r1_bbox[1])
    r1_area = float(r1_bbox[2]) * float(r1_bbox[3])

    # Pre-compute pool distribution per page for fast peer metrics
    page_counts: dict[int, int] = {}
    for c in pool:
        p = c.get("page", 1)
        page_counts[p] = page_counts.get(p, 0) + 1

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []

    for c in pool:
        label = 1 if float(c.get("best_iou", 0.0)) >= 0.50 else 0
        c_text = str(c.get("text") or "")
        c_norm = str(c.get("normalized_text") or "").strip().lower()
        c_tokens = set(c_norm.split())

        bbox = c.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        x = float(bbox[0])
        y = float(bbox[1])
        w = float(bbox[2])
        h = float(bbox[3])
        area = w * h
        aspect = w / max(1e-4, h)
        density = len(c_text) / max(1e-4, area)

        page = int(c.get("page", 1))
        rank = int(c.get("rank", 1))
        src = str(c.get("source") or "").lower()

        # 1. TEXT FEATURES
        exact_match = 1.0 if c_text == query_val else 0.0
        norm_match = 1.0 if c_norm == q_norm else 0.0
        fuzz_ratio = fuzz.ratio(c_norm, q_norm) / 100.0 if (c_norm and q_norm) else 0.0
        token_overlap = (
            len(c_tokens & q_tokens) / max(1, len(c_tokens | q_tokens))
            if (c_tokens or q_tokens)
            else 0.0
        )
        c_num = _clean_num(c_text)
        if c_num is not None and q_num is not None:
            diff = abs(c_num - q_num)
            num_sim = 1.0 if diff < 1e-4 else (1.0 / (1.0 + diff))
        else:
            num_sim = 1.0 if exact_match else 0.0

        len_c = len(c_text)
        len_q = len(query_val)
        char_len_ratio = min(len_c, len_q) / max(1, len_c, len_q)
        len_diff = float(abs(len_c - len_q))

        # 2. STRUCTURAL FEATURES
        same_page_peers = (page_counts.get(page, 1) - 1) / max(1, total_cands - 1)
        # Fast peer checks
        same_line_cnt = sum(
            1 for other in pool
            if other.get("page", 1) == page and other is not c
            and abs(float((other.get("bbox") or [0, 0, 0, 0])[1]) - y) < 0.012
        )
        same_col_cnt = sum(
            1 for other in pool
            if other.get("page", 1) == page and other is not c
            and abs(float((other.get("bbox") or [0, 0, 0, 0])[0]) - x) < 0.025
        )
        same_line_peers = same_line_cnt / max(1, total_cands - 1)
        same_col_peers = same_col_cnt / max(1, total_cands - 1)

        same_page_r1 = 1.0 if page == r1_page else 0.0
        same_line_r1 = 1.0 if (same_page_r1 and abs(y - r1_y) < 0.012) else 0.0
        same_col_r1 = 1.0 if (same_page_r1 and abs(x - r1_x) < 0.025) else 0.0

        # 3. GEOMETRIC FEATURES
        dist_left = x
        dist_right = max(0.0, 1.0 - (x + w))
        dist_top = y
        dist_bottom = max(0.0, 1.0 - (y + h))

        dx_r1 = abs(x - r1_x)
        dy_r1 = abs(y - r1_y)
        dist_r1 = math.sqrt(dx_r1 * dx_r1 + dy_r1 * dy_r1)
        area_ratio_r1 = area / max(1e-5, r1_area)

        # 4. DOCUMENT FEATURES
        recip_page = 1.0 / max(1, page)
        is_p1 = 1.0 if page == 1 else 0.0
        is_p2 = 1.0 if page == 2 else 0.0
        is_p3_plus = 1.0 if page >= 3 else 0.0

        # 5. SOURCE FEATURES
        src_exact = 1.0 if src == "exact" else 0.0
        src_numeric = 1.0 if "number" in src else 0.0
        src_date = 1.0 if "date" in src else 0.0
        src_bool = 1.0 if "bool" in src else 0.0
        src_fuzzy = 1.0 if "fuzzy" in src else 0.0
        src_recovered = 1.0 if "recovered" in src else 0.0
        src_ocr = 1.0 if "ocr" in src else 0.0

        # 6. RANKING FEATURES
        recip_rank = 1.0 / max(1, rank)
        rank_is_1 = 1.0 if rank == 1 else 0.0
        rank_is_2 = 1.0 if rank == 2 else 0.0
        rank_is_3 = 1.0 if rank == 3 else 0.0
        rank_diff_r1 = float(rank - 1)

        row = [
            exact_match,
            norm_match,
            fuzz_ratio,
            token_overlap,
            num_sim,
            char_len_ratio,
            len_diff,
            is_table,
            field_depth,
            same_page_peers,
            same_line_peers,
            same_col_peers,
            same_page_r1,
            same_line_r1,
            same_col_r1,
            x,
            y,
            w,
            h,
            area,
            aspect,
            density,
            dist_left,
            dist_right,
            dist_top,
            dist_bottom,
            dx_r1,
            dy_r1,
            dist_r1,
            area_ratio_r1,
            float(page),
            recip_page,
            is_p1,
            is_p2,
            is_p3_plus,
            split_short,
            split_med,
            split_long,
            is_sec,
            is_tax,
            is_reg,
            is_sched,
            src_exact,
            src_numeric,
            src_date,
            src_bool,
            src_fuzzy,
            src_recovered,
            src_ocr,
            float(rank),
            recip_rank,
            rank_is_1,
            rank_is_2,
            rank_is_3,
            rank_diff_r1,
            float(total_cands),
            inv_total_cands,
        ]

        feature_rows.append(row)
        labels.append(label)
        meta.append({
            "rank": rank,
            "page": page,
            "bbox": bbox,
            "source": src,
            "best_iou": float(c.get("best_iou", 0.0)),
        })

    return feature_rows, labels, meta
