"""Realistic Gain Estimator for Failure Microscope V4.

Implements the three-metric calibration required by Section 3 of the Directive:
1. THEORETICAL_CEILING: Macro-weighted opportunity if 100% of fields are recovered perfectly.
2. REALISTIC_RECOVERY_RATE: Empirical recovery rate derived from historical benchmark experiments.
3. REALISTIC_EXPECTED_GAIN: THEORETICAL_CEILING * REALISTIC_RECOVERY_RATE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RATE_TABLE_PATH = Path(__file__).resolve().parent / "recovery_rate_table.json"


@dataclass(slots=True)
class RealisticClassEstimate:
    failure_class: str
    field_count: int
    document_count: int
    document_breadth_pct: float
    theoretical_ceiling_pp: float
    realistic_recovery_rate: float
    realistic_expected_gain_pp: float
    fix_type: str
    confidence: str
    source_evidence: str
    recommended_next_step: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class,
            "field_count": self.field_count,
            "document_count": self.document_count,
            "document_breadth_pct": round(self.document_breadth_pct, 2),
            "theoretical_ceiling_pp": round(self.theoretical_ceiling_pp, 4),
            "realistic_recovery_rate": round(self.realistic_recovery_rate, 4),
            "realistic_recovery_rate_pct": f"{self.realistic_recovery_rate * 100:.2f}%",
            "realistic_expected_gain_pp": round(self.realistic_expected_gain_pp, 4),
            "fix_type": self.fix_type,
            "confidence": self.confidence,
            "source_evidence": self.source_evidence,
            "recommended_next_step": self.recommended_next_step,
        }

    def format_text_block(self) -> str:
        """Format as required by Section 3.4."""
        return (
            f"FAILURE CLASS: {self.failure_class}\n"
            f"  Field count:              {self.field_count}\n"
            f"  Document breadth:         {self.document_count} ({self.document_breadth_pct:.2f}% of grounded docs)\n"
            f"  THEORETICAL_CEILING:      +{self.theoretical_ceiling_pp:.4f} pp  (upper bound, not achievable)\n"
            f"  REALISTIC_RECOVERY_RATE:  {self.realistic_recovery_rate * 100:.2f}%      (empirical, source: {self.source_evidence})\n"
            f"  REALISTIC_EXPECTED_GAIN:  +{self.realistic_expected_gain_pp:.4f} pp  (theoretical × recovery)\n"
            f"  Fix type:                 {self.fix_type}\n"
            f"  Historical confidence:    {self.confidence}\n"
            f"  Source evidence:          {self.source_evidence}\n"
            f"  Recommended next step:    {self.recommended_next_step}"
        )


class RealisticEstimator:
    """Calculates realistic expected benchmark gains based on empirical historical rates."""

    def __init__(self, rate_table_path: Path | str | None = None) -> None:
        p = Path(rate_table_path) if rate_table_path else RATE_TABLE_PATH
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.rates: dict[str, dict[str, Any]] = data.get("rates", {})
        self.default_fallback: dict[str, Any] = data.get("default_fallback", {
            "fix_type": "General heuristic",
            "historical_recovery_rate": 0.10,
            "confidence": "LOW",
            "source": "Conservative prior (10%)",
            "recommended_next_step": "Targeted forensic audit",
        })

    def estimate_class(
        self,
        failure_class: str,
        field_count: int,
        document_count: int,
        total_benchmark_docs: int,
        theoretical_ceiling_pp: float,
    ) -> RealisticClassEstimate:
        info = self.rates.get(failure_class, self.default_fallback)
        rate = float(info.get("historical_recovery_rate", 0.10))
        expected_gain = theoretical_ceiling_pp * rate
        breadth_pct = (document_count / max(1, total_benchmark_docs)) * 100.0

        return RealisticClassEstimate(
            failure_class=failure_class,
            field_count=field_count,
            document_count=document_count,
            document_breadth_pct=breadth_pct,
            theoretical_ceiling_pp=theoretical_ceiling_pp,
            realistic_recovery_rate=rate,
            realistic_expected_gain_pp=expected_gain,
            fix_type=str(info.get("fix_type", "General heuristic")),
            confidence=str(info.get("confidence", "LOW")),
            source_evidence=str(info.get("source", "Conservative prior")),
            recommended_next_step=str(info.get("recommended_next_step", "Targeted audit")),
        )
