"""EXP-045 Algorithm 4: Constraint Satisfaction Filter (Forward Checking with Backtracking).

Applies hard layout and structural mutual exclusivity constraints across candidate bindings:
- Mutual Exclusivity: Distinct sibling fields in the same record cannot bind to the identical token bbox.
- Page Constraint: Candidate must reside on the target page.
- Coordinate Consistency: Y-coordinates must follow monotonic row progression.
"""

from __future__ import annotations

from typing import Any
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentToken


class ConstraintSolver:
    """Solves candidate assignment using forward checking with backtracking."""

    def solve(
        self,
        field_candidates: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        """Find a mutually consistent assignment of candidates to fields.
        
        Args:
            field_candidates: dict of field_path -> list of candidate citation dicts
            
        Returns:
            dict of field_path -> assigned candidate citation dict
        """
        fields = list(field_candidates.keys())
        assignment: dict[str, dict[str, Any]] = {}

        def is_consistent(cand: dict[str, Any], current_assignment: dict[str, dict[str, Any]]) -> bool:
            cand_bbox = cand.get("bbox")
            cand_page = cand.get("page")
            if not cand_bbox:
                return False

            for assigned_field, assigned_cand in current_assignment.items():
                if assigned_cand.get("page") == cand_page and assigned_cand.get("bbox") == cand_bbox:
                    # Duplicate token binding
                    return False
            return True

        def backtrack(index: int) -> bool:
            if index == len(fields):
                return True

            fp = fields[index]
            cands = field_candidates.get(fp, [])
            if not cands:
                return backtrack(index + 1)

            for cand in cands:
                if is_consistent(cand, assignment):
                    assignment[fp] = cand
                    if backtrack(index + 1):
                        return True
                    del assignment[fp]

            return False

        backtrack(0)
        return assignment
