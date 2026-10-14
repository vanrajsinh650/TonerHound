"""EXP-045 Algorithm 5: Optimal Transport (Sinkhorn Matrix Scaling).

Solves soft spatial matching for fields with many repeated candidates (> 5 occurrences)
using entropic-regularized optimal transport (Sinkhorn-Knopp algorithm).
Treats expected row coordinates and candidate token coordinates as discrete distributions.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def sinkhorn(
    cost_matrix: np.ndarray,
    reg: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    """Compute optimal transport plan via Sinkhorn-Knopp algorithm."""
    n, m = cost_matrix.shape
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Kernel matrix K = exp(-cost / reg)
    K = np.exp(-cost_matrix / reg)
    K = np.maximum(K, 1e-12)

    u = np.ones(n)
    for _ in range(max_iter):
        v = b / (K.T @ u + 1e-12)
        u_new = a / (K @ v + 1e-12)
        if np.max(np.abs(u_new - u)) < tol:
            u = u_new
            break
        u = u_new

    # Transport plan P = diag(u) * K * diag(v)
    P = np.diag(u) @ K @ np.diag(v)
    return P


def match_optimal_transport(
    target_val: Any,
    page_obj: DocumentPage,
    expected_y_ratio: float = 0.5,
) -> tuple[BBox | None, str | None]:
    """Find the best candidate token using optimal transport cost minimization."""
    if target_val is None or not page_obj or not page_obj.tokens:
        return None, None

    val_str = str(target_val).strip().lower()
    cands: list[DocumentToken] = [
        t for t in page_obj.tokens
        if val_str == t.text.lower() or (len(val_str) >= 3 and val_str in t.text.lower())
    ]

    if not cands:
        return None, None

    if len(cands) == 1:
        return cands[0].bbox, cands[0].text

    # Compute vertical distance cost from expected position
    page_h = max(1.0, page_obj.height)
    exp_y = expected_y_ratio * page_h

    costs = np.array([abs(t.bbox.y - exp_y) / page_h for t in cands], dtype=np.float64)
    cost_matrix = costs.reshape((1, len(cands)))

    plan = sinkhorn(cost_matrix, reg=0.05)
    best_idx = int(np.argmax(plan[0]))
    best_cand = cands[best_idx]
    return best_cand.bbox, best_cand.text
