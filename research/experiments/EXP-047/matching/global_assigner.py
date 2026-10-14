"""EXP-047 Phase D: Global Assigner — Document-Level Constraint Assignment.

Implements the core paradigm shift of EXP-047:
Instead of solving each field independently, solve ALL failing fields as one
global constraint problem via block-decomposed linear sum assignment (Hungarian
algorithm via scipy.optimize.linear_sum_assignment).

For each block of fields:
1. Generate candidates from char_index and numeric_index
2. Filter candidates by table page range (from table_localizer)
3. Compute cost matrix: value_mismatch + position_prior_error + type_mismatch + page_penalty
4. Solve: scipy.optimize.linear_sum_assignment
5. Enforce monotonicity for array fields (bubble sort on y-position)

Usage:
    assigner = GlobalAssigner(pdf_path, char_index, numeric_index, table_localizer)
    assignments = assigner.assign(failing_fields)
    # Returns dict[field_path → AssignedCandidate | None]
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from char_index import CharIndex, CharMatch
from numeric_index import NumericIndex, NumericMatch, parse_numeric_value
from table_localizer import TableLocalizer, PositionPrior, parse_array_field


# ---------------------------------------------------------------------------
# Field descriptor (input to global assigner)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FieldDescriptor:
    """Describes a single failing field to be resolved."""
    field_path: str       # e.g. "holdings[42].shares"
    expected_value: str   # primary string representation of the expected value
    expected_type: str    # "number", "date", "text", "boolean"
    page_hint: int | None # page hint from extraction (1-indexed), if available
    alt_values: list[str] = None  # alternative representations to also search


@dataclass(slots=True)
class AssignedCandidate:
    """An assignment of a field to a document location."""
    field_path: str
    page: int              # 1-indexed
    x0: float
    y0: float
    x1: float
    y1: float
    matched_text: str
    cost: float
    source: str            # "char", "numeric", "baseline"
    hungarian_rank: int    # rank in the Hungarian solution (0 = best)

    @property
    def bbox_xywh(self) -> list[float]:
        return [self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0]


# ---------------------------------------------------------------------------
# Cost components
# ---------------------------------------------------------------------------

def _value_mismatch_cost(expected: str, matched: str) -> float:
    """Cost [0, 5] based on how well matched_text matches expected value."""
    if not expected or not matched:
        return 5.0

    exp_clean = expected.strip().lower()
    mat_clean = matched.strip().lower()

    if exp_clean == mat_clean:
        return 0.0

    # Numeric equivalence
    exp_num = parse_numeric_value(expected)
    mat_num = parse_numeric_value(matched)
    if exp_num is not None and mat_num is not None:
        rel_err = abs(exp_num - mat_num) / (abs(exp_num) + 1e-9)
        if rel_err < 1e-6:
            return 0.0
        if rel_err < 0.01:
            return 0.1
        if rel_err < 0.1:
            return 0.5

    # Substring match
    if exp_clean in mat_clean or mat_clean in exp_clean:
        return 0.3

    # Character overlap ratio
    exp_set = set(exp_clean.replace(" ", ""))
    mat_set = set(mat_clean.replace(" ", ""))
    if exp_set and mat_set:
        overlap = len(exp_set & mat_set) / len(exp_set | mat_set)
        if overlap > 0.9:
            return 0.5
        if overlap > 0.7:
            return 1.0

    return 5.0


def _position_prior_error(
    prior: PositionPrior | None,
    candidate_y: float,
    candidate_x: float,
    candidate_page: int,
) -> float:
    """Cost [0, 5] based on distance from expected position."""
    if prior is None:
        return 0.0  # No prior → no penalty

    # Check if candidate is within table page range
    if not (prior.page_start <= candidate_page <= prior.page_end):
        return 3.0

    # Compute normalized y distance within table
    table_y_range = prior.body_y_end - 0.0 if prior.page_end == prior.page_start else 1.0
    if table_y_range <= 0:
        table_y_range = 1.0

    dy = abs(candidate_y - prior.expected_y)
    dx = abs(candidate_x - prior.expected_x)

    # Weight by confidence
    pos_err = (dy + 0.5 * dx) * prior.confidence
    return min(5.0, pos_err * 10)


def _type_mismatch_cost(expected_type: str, matched_text: str) -> float:
    """Cost [0, 5] based on whether matched text matches expected type."""
    if expected_type == "number":
        val = parse_numeric_value(matched_text)
        return 0.0 if val is not None else 3.0
    elif expected_type == "date":
        # Simple date pattern check
        if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", matched_text):
            return 0.0
        if re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", matched_text, re.I):
            return 0.0
        return 2.0
    elif expected_type == "boolean":
        text_lower = matched_text.strip().lower()
        if text_lower in ("yes", "no", "true", "false", "x", "✓", "☑", "☐"):
            return 0.0
        return 3.0
    return 0.0  # text — no type cost


def _page_penalty(expected_page: int | None, candidate_page: int) -> float:
    """Cost [0, 5] based on page distance."""
    if expected_page is None:
        return 0.0
    diff = abs(expected_page - candidate_page)
    if diff == 0:
        return 0.0
    if diff == 1:
        return 1.0
    if diff <= 3:
        return 2.0
    return 5.0


# Weights from spec: value(0.40) + position(0.30) + type(0.20) + page(0.10)
_W_VALUE = 0.40
_W_POSITION = 0.30
_W_TYPE = 0.20
_W_PAGE = 0.10


def compute_cost(
    field: FieldDescriptor,
    prior: PositionPrior | None,
    candidate_y: float,
    candidate_x: float,
    candidate_page: int,
    matched_text: str,
) -> float:
    """Compute total cost for a (field, candidate) pair."""
    c_val = _value_mismatch_cost(field.expected_value, matched_text)
    c_pos = _position_prior_error(prior, candidate_y, candidate_x, candidate_page)
    c_type = _type_mismatch_cost(field.expected_type, matched_text)
    c_page = _page_penalty(field.page_hint, candidate_page)

    return (
        _W_VALUE * c_val
        + _W_POSITION * c_pos
        + _W_TYPE * c_type
        + _W_PAGE * c_page
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Candidate:
    """Internal candidate for a field."""
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    matched_text: str
    source: str  # "char" or "numeric"


def _generate_candidates(
    field: FieldDescriptor,
    char_idx: CharIndex,
    num_idx: NumericIndex,
    prior: PositionPrior | None,
    max_candidates: int = 50,
) -> list[_Candidate]:
    """Generate candidates for a field from char and numeric indexes.

    Searches expected_value AND all alt_values to maximize recall.
    """
    candidates: list[_Candidate] = []
    seen_locs: set[tuple[int, float, float]] = set()

    # Collect all values to search (primary + alternates)
    values_to_search: list[str] = [str(field.expected_value).strip()]
    if field.alt_values:
        for av in field.alt_values:
            sv = str(av).strip()
            if sv and sv not in values_to_search:
                values_to_search.append(sv)

    for expected_val in values_to_search:
        if not expected_val or expected_val == "None":
            continue

        # --- Text candidates from char index ---
        if len(expected_val) >= 1:
            try:
                char_matches = char_idx.find_all(expected_val)
            except Exception:
                char_matches = []

            for m in char_matches[:max_candidates]:
                loc_key = (m.page, round(m.x0, 3), round(m.y0, 3))
                if loc_key in seen_locs:
                    continue
                seen_locs.add(loc_key)

                # Filter by page range if prior available
                if prior is not None:
                    if not (prior.page_start <= m.page <= prior.page_end):
                        continue

                candidates.append(_Candidate(
                    page=m.page,
                    x0=m.x0, y0=m.y0, x1=m.x1, y1=m.y1,
                    matched_text=m.matched_text,
                    source="char",
                ))

        # --- Numeric candidates from numeric index ---
        if field.expected_type == "number" or _looks_numeric(expected_val):
            try:
                num_matches = num_idx.find_numeric_value(expected_val, tolerance=0.01)
            except Exception:
                num_matches = []

            for m in num_matches[:max_candidates]:
                loc_key = (m.page, round(m.x0, 3), round(m.y0, 3))
                if loc_key in seen_locs:
                    continue
                seen_locs.add(loc_key)

                if prior is not None:
                    if not (prior.page_start <= m.page <= prior.page_end):
                        continue

                candidates.append(_Candidate(
                    page=m.page,
                    x0=m.x0, y0=m.y0, x1=m.x1, y1=m.y1,
                    matched_text=m.original_text,
                    source="numeric",
                ))

        if len(candidates) >= max_candidates:
            break

    return candidates[:max_candidates]


def _looks_numeric(s: str) -> bool:
    """Quick check if string looks like a number."""
    return bool(re.search(r"^\s*[$€£¥]?\s*\(?\s*-?\d[\d,.]*%?\s*\)?\s*$", s))


# ---------------------------------------------------------------------------
# Global Assigner
# ---------------------------------------------------------------------------

class GlobalAssigner:
    """Document-level global field assignment via block-decomposed Hungarian algorithm.

    Solves all failing fields jointly:
    1. Generates candidates for each field via char_index + numeric_index
    2. Filters by table page range (from table_localizer)
    3. Computes cost matrix
    4. Solves linear sum assignment per block
    5. Enforces monotonicity for array fields

    Parameters:
        block_size: Number of fields per Hungarian block. 100 recommended.
        max_candidates_per_field: Max candidates to consider per field.
        unresolved_cost_threshold: Fields with best cost > threshold get no assignment.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        char_idx: CharIndex,
        num_idx: NumericIndex,
        table_loc: TableLocalizer,
        block_size: int = 100,
        max_candidates_per_field: int = 20,
        unresolved_cost_threshold: float = 4.0,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.char_idx = char_idx
        self.num_idx = num_idx
        self.table_loc = table_loc
        self.block_size = block_size
        self.max_candidates = max_candidates_per_field
        self.unresolved_threshold = unresolved_cost_threshold

        # Statistics
        self.stats: dict[str, Any] = {
            "total_fields": 0,
            "fields_zero_candidates": 0,
            "fields_one_candidate": 0,
            "fields_2_10_candidates": 0,
            "fields_gt10_candidates": 0,
            "total_candidate_pairs": 0,
            "hungarian_solve_times": [],
            "monotonicity_corrections": 0,
            "assignments_made": 0,
            "fields_unresolved": 0,
        }

    def assign(
        self,
        fields: list[FieldDescriptor],
    ) -> dict[str, AssignedCandidate | None]:
        """Assign candidates to all fields using block-decomposed Hungarian.

        Returns:
            Dict mapping field_path → AssignedCandidate (or None if unresolved).
        """
        self.stats["total_fields"] = len(fields)
        result: dict[str, AssignedCandidate | None] = {}

        if not fields:
            return result

        # Step 1: Generate candidates for all fields
        field_candidates: dict[str, list[_Candidate]] = {}
        field_priors: dict[str, PositionPrior | None] = {}

        for f in fields:
            prior = self.table_loc.get_position_prior(f.field_path)
            field_priors[f.field_path] = prior
            cands = _generate_candidates(
                f, self.char_idx, self.num_idx, prior, self.max_candidates
            )
            field_candidates[f.field_path] = cands

            n = len(cands)
            self.stats["total_candidate_pairs"] += n
            if n == 0:
                self.stats["fields_zero_candidates"] += 1
            elif n == 1:
                self.stats["fields_one_candidate"] += 1
            elif n <= 10:
                self.stats["fields_2_10_candidates"] += 1
            else:
                self.stats["fields_gt10_candidates"] += 1

        # Step 2: Block decomposition
        for block_start in range(0, len(fields), self.block_size):
            block = fields[block_start:block_start + self.block_size]
            block_result = self._solve_block(
                block, field_candidates, field_priors
            )
            result.update(block_result)

        # Step 3: Monotonicity enforcement for array fields
        self._enforce_monotonicity(fields, result)

        # Update stats
        self.stats["assignments_made"] = sum(1 for v in result.values() if v is not None)
        self.stats["fields_unresolved"] = sum(1 for v in result.values() if v is None)

        return result

    def _solve_block(
        self,
        block: list[FieldDescriptor],
        field_candidates: dict[str, list[_Candidate]],
        field_priors: dict[str, PositionPrior | None],
    ) -> dict[str, AssignedCandidate | None]:
        """Solve assignment for one block of fields using Hungarian algorithm."""
        result: dict[str, AssignedCandidate | None] = {}

        # Filter to fields with at least one candidate
        active_fields = [f for f in block if field_candidates[f.field_path]]
        no_cand_fields = [f for f in block if not field_candidates[f.field_path]]

        for f in no_cand_fields:
            result[f.field_path] = None

        if not active_fields:
            return result

        # Collect all unique candidates for this block
        all_candidates: list[_Candidate] = []
        seen_cand_keys: set[tuple[int, float, float, str]] = set()
        field_to_cand_indices: dict[str, list[int]] = {}

        for f in active_fields:
            cands = field_candidates[f.field_path]
            indices = []
            for c in cands:
                key = (c.page, round(c.x0, 4), round(c.y0, 4), c.matched_text)
                if key not in seen_cand_keys:
                    seen_cand_keys.add(key)
                    all_candidates.append(c)
                idx = next(
                    (i for i, ac in enumerate(all_candidates)
                     if (ac.page, round(ac.x0, 4), round(ac.y0, 4), ac.matched_text) == key),
                    len(all_candidates) - 1
                )
                indices.append(idx)
            field_to_cand_indices[f.field_path] = indices

        n_fields = len(active_fields)
        n_cands = len(all_candidates)

        if n_cands == 0:
            for f in active_fields:
                result[f.field_path] = None
            return result

        # Build cost matrix: n_fields × n_cands
        BIG_COST = 100.0  # cost for "not assigned"
        cost_matrix = np.full((n_fields, n_cands), BIG_COST, dtype=np.float32)

        for fi, f in enumerate(active_fields):
            prior = field_priors[f.field_path]
            for ci_idx in field_to_cand_indices[f.field_path]:
                c = all_candidates[ci_idx]
                cost = compute_cost(
                    f,
                    prior,
                    candidate_y=c.y0,
                    candidate_x=(c.x0 + c.x1) / 2,
                    candidate_page=c.page,
                    matched_text=c.matched_text,
                )
                if cost < cost_matrix[fi, ci_idx]:
                    cost_matrix[fi, ci_idx] = cost

        # Hungarian algorithm
        t0 = time.perf_counter()
        if n_fields <= n_cands:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            # More fields than candidates — pad cost matrix
            padding = np.full((n_fields, n_fields - n_cands), BIG_COST, dtype=np.float32)
            padded = np.hstack([cost_matrix, padding])
            row_ind, col_ind = linear_sum_assignment(padded)
            # Filter out padded assignments
            col_ind = np.where(col_ind < n_cands, col_ind, -1)

        dt = time.perf_counter() - t0
        self.stats["hungarian_solve_times"].append(dt)

        # Record assignments
        for ri, ci in zip(row_ind, col_ind):
            f = active_fields[ri]
            if ci < 0 or ci >= n_cands:
                result[f.field_path] = None
                continue
            c = all_candidates[ci]
            assigned_cost = cost_matrix[ri, ci]

            if assigned_cost >= self.unresolved_threshold:
                result[f.field_path] = None
                continue

            result[f.field_path] = AssignedCandidate(
                field_path=f.field_path,
                page=c.page,
                x0=c.x0, y0=c.y0, x1=c.x1, y1=c.y1,
                matched_text=c.matched_text,
                cost=float(assigned_cost),
                source=c.source,
                hungarian_rank=0,
            )

        # Fields not in row_ind get None
        assigned_rows = set(row_ind)
        for fi, f in enumerate(active_fields):
            if fi not in assigned_rows and f.field_path not in result:
                result[f.field_path] = None

        return result

    def _enforce_monotonicity(
        self,
        fields: list[FieldDescriptor],
        result: dict[str, AssignedCandidate | None],
    ) -> None:
        """Enforce that array field assignments are monotonically increasing in y.

        For holdings[i].shares and holdings[i+1].shares, the y-position of
        holdings[i].shares must be <= that of holdings[i+1].shares.
        If violated, swap the assignments.
        """
        # Group by (table_name, subfield)
        groups: dict[tuple[str, str], list[tuple[int, FieldDescriptor]]] = {}
        for f in fields:
            parsed = parse_array_field(f.field_path)
            if parsed is None:
                continue
            table_name, idx, subfield = parsed
            key = (table_name, subfield)
            if key not in groups:
                groups[key] = []
            groups[key].append((idx, f))

        for key, entries in groups.items():
            entries.sort(key=lambda x: x[0])  # sort by array index

            # Bubble sort until monotone
            changed = True
            max_iters = len(entries)
            iters = 0
            while changed and iters < max_iters:
                changed = False
                iters += 1
                for i in range(len(entries) - 1):
                    idx_i, f_i = entries[i]
                    idx_j, f_j = entries[i + 1]
                    a_i = result.get(f_i.field_path)
                    a_j = result.get(f_j.field_path)
                    if a_i is None or a_j is None:
                        continue
                    # Check: a_i.y0 should be <= a_j.y0 (or same page)
                    if a_i.page == a_j.page and a_i.y0 > a_j.y0 + 0.005:
                        # Violation → swap
                        result[f_i.field_path] = AssignedCandidate(
                            field_path=f_i.field_path,
                            page=a_j.page, x0=a_j.x0, y0=a_j.y0,
                            x1=a_j.x1, y1=a_j.y1,
                            matched_text=a_j.matched_text,
                            cost=a_j.cost,
                            source=a_j.source,
                            hungarian_rank=a_j.hungarian_rank,
                        )
                        result[f_j.field_path] = AssignedCandidate(
                            field_path=f_j.field_path,
                            page=a_i.page, x0=a_i.x0, y0=a_i.y0,
                            x1=a_i.x1, y1=a_i.y1,
                            matched_text=a_i.matched_text,
                            cost=a_i.cost,
                            source=a_i.source,
                            hungarian_rank=a_i.hungarian_rank,
                        )
                        changed = True
                        self.stats["monotonicity_corrections"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Return aggregated statistics."""
        times = self.stats["hungarian_solve_times"]
        return {
            **self.stats,
            "avg_hungarian_time_ms": (
                sum(times) / len(times) * 1000 if times else 0.0
            ),
            "max_hungarian_time_ms": (max(times) * 1000 if times else 0.0),
        }
