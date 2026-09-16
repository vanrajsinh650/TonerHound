"""Failure Correlation Framework for TonerHound Benchmark Citations.

Provides rigorous attribution and correlation of accuracy gains to specific failure
taxonomies (G1, G2, G3, G4, T1), enabling validation that targeted fixes (e.g.,
Agent A's G1 slot drift fix or Agent B's G3 multi-line fix) resolve the intended
failure mechanisms without causing cross-category regressions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh


class ErrorTaxonomy(str, Enum):
    """Failure error categories for long-document tabular grounding."""

    G1 = "G1"  # Row Slot Drift (anchor >= 1 slot off, |dy| > 0.006)
    G2 = "G2"  # Cell Dimension Mismatch (IoU between 0.30 and 0.499, correct row but box width/height off)
    G3 = "G3"  # Multi-Line Line Choice (|dy| ~ 0.010 - 0.013, line 2 vs line 1 within 2-slot cells)
    G4 = "G4"  # Column Boundary Offset (horizontal displacement, |dx| > 0.01)
    T1 = "T1"  # Severe OCR Glyph Corruption (fragmented characters, speckle noise, IoU < 0.30)
    CORRECT = "CORRECT"  # Ground truth bounding box matches with IoU >= 0.50


@dataclass
class CitationRecord:
    """Baseline record for a single gradeable field citation."""

    path: str
    status: str  # "PASS" or "FAIL"
    category: str  # "CORRECT", "G1", "G2", "G3", "G4", "T1"
    iou: float
    gt_page: int
    gt_box: tuple[float, float, float, float]
    pred_page: int | None = None
    pred_box: tuple[float, float, float, float] | None = None
    dy: float | None = None
    dx: float | None = None
    row_idx: int | None = None
    field: str = ""
    is_multi_line_row: bool = False


@dataclass
class CorrelationReport:
    """Detailed correlation report evaluating a proposed algorithmic fix against baseline."""

    target_category: str
    baseline_total_gradeable: int
    baseline_passing_count: int
    baseline_failing_count: int
    candidate_passing_count: int
    candidate_failing_count: int
    newly_passing_count: int
    regressions_count: int
    net_gain: int
    target_fixed_count: int
    purity: float  # target_fixed_count / newly_passing_count
    target_recall: float  # target_fixed_count / baseline_target_category_count
    fixed_by_category: dict[str, int] = field(default_factory=dict)
    regressed_by_category: dict[str, int] = field(default_factory=dict)
    newly_passing_paths: list[str] = field(default_factory=list)
    regressed_paths: list[str] = field(default_factory=list)
    is_verified: bool = False
    verification_notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Format a comprehensive markdown summary table of the correlation results."""
        verdict = "**VERIFIED ON-TARGET**" if self.is_verified else "**UNVERIFIED / REGRESSION DETECTED**"
        lines = [
            f"### Failure Correlation Report: Target Category `{self.target_category}`",
            f"- **Overall Verdict**: {verdict}",
            f"- **Baseline Passing**: {self.baseline_passing_count} / {self.baseline_total_gradeable} "
            f"({self.baseline_passing_count / self.baseline_total_gradeable * 100:.2f}%)",
            f"- **Candidate Passing**: {self.candidate_passing_count} / {self.baseline_total_gradeable} "
            f"({self.candidate_passing_count / self.baseline_total_gradeable * 100:.2f}%)",
            f"- **Net Gain**: **{self.net_gain:+d} citations** "
            f"(+{self.newly_passing_count} fixed, -{self.regressions_count} regressed)",
            f"- **Attribution Purity**: **{self.purity * 100:.2f}%** "
            f"({self.target_fixed_count} of {self.newly_passing_count} newly fixed belong to `{self.target_category}`)",
            f"- **Target Category Recall**: **{self.target_recall * 100:.2f}%** "
            f"({self.target_fixed_count} fixed out of baseline target pool)",
            "",
            "#### Newly Fixed Citations Breakdown by Baseline Taxonomy:",
            "| Taxonomy Category | Newly Fixed Count | % of All Fixes | Status |",
            "| :--- | :---: | :---: | :--- |",
        ]

        for cat, cnt in sorted(self.fixed_by_category.items(), key=lambda x: x[1], reverse=True):
            pct = cnt / self.newly_passing_count * 100 if self.newly_passing_count > 0 else 0.0
            tag = "TARGET" if cat == self.target_category else "COLLATERAL"
            lines.append(f"| **{cat}** | {cnt} | {pct:.2f}% | {tag} |")

        if self.regressions_count > 0:
            lines.extend([
                "",
                "#### Regressions Breakdown:",
                "| Taxonomy Category | Regressed Count | Sample Path |",
                "| :--- | :---: | :--- |",
            ])
            for cat, cnt in self.regressed_by_category.items():
                sample = next((p for p in self.regressed_paths), "N/A")
                lines.append(f"| **{cat}** | {cnt} | `{sample}` |")
        else:
            lines.extend(["", "- **Regressions**: **0 citations** (Zero regression verified)."])

        if self.verification_notes:
            lines.append("")
            lines.append("#### Verification Notes:")
            for note in self.verification_notes:
                lines.append(f"- {note}")

        return "\n".join(lines)


class FailureRegistry:
    """Manages baseline groundings and provides verification correlation services."""

    DEFAULT_REGISTRY_PATH = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "experiments"
        / "EXP-005-ftx-failure-registry.json"
    )

    def __init__(self, records: dict[str, CitationRecord]) -> None:
        self.records = records
        self.category_counts = Counter(r.category for r in records.values())
        self.passing_paths = {p for p, r in records.items() if r.status == "PASS"}
        self.failing_paths = {p for p, r in records.items() if r.status == "FAIL"}

    @classmethod
    def load(cls, path: Path | str | None = None) -> FailureRegistry:
        """Load registry from JSON file."""
        file_path = Path(path) if path is not None else cls.DEFAULT_REGISTRY_PATH
        if not file_path.exists():
            raise FileNotFoundError(f"Failure registry file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records: dict[str, CitationRecord] = {}
        for p, d in data.items():
            records[p] = CitationRecord(
                path=d["path"],
                status=d["status"],
                category=d["category"],
                iou=d["iou"],
                gt_page=d["gt_page"],
                gt_box=tuple(d["gt_box"]),  # type: ignore[arg-type]
                pred_page=d.get("pred_page"),
                pred_box=tuple(d["pred_box"]) if d.get("pred_box") else None,  # type: ignore[arg-type]
                dy=d.get("dy"),
                dx=d.get("dx"),
                row_idx=d.get("row_idx"),
                field=d.get("field", ""),
                is_multi_line_row=d.get("is_multi_line_row", False),
            )
        return cls(records)

    def get_category_counts(self) -> dict[str, int]:
        """Return counts of all categories in the registry."""
        return dict(self.category_counts)

    def correlate(
        self,
        candidate_citations: Sequence[Mapping[str, Any]],
        target_category: str,
        *,
        iou_threshold: float = 0.50,
        min_purity: float = 0.60,
    ) -> CorrelationReport:
        """Correlate candidate citations against baseline registry for a targeted fix."""
        # Index candidate citations by field_path
        cand_map: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
        for cit in candidate_citations:
            path = cit.get("field_path")
            page = cit.get("page")
            bbox = cit.get("bbox")
            if path and page is not None and bbox and len(bbox) >= 4:
                cand_map[path].append((int(page), (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))

        newly_passing: list[str] = []
        regressions: list[str] = []
        fixed_by_cat: Counter[str] = Counter()
        regressed_by_cat: Counter[str] = Counter()
        total_candidate_passing = 0

        for path, rec in self.records.items():
            cand_boxes = cand_map.get(path, [])
            passes_now = any(
                cp == rec.gt_page and iou_xywh(rec.gt_box, cb) >= iou_threshold
                for cp, cb in cand_boxes
            )

            if passes_now:
                total_candidate_passing += 1
                if rec.status == "FAIL":
                    newly_passing.append(path)
                    fixed_by_cat[rec.category] += 1
            else:
                if rec.status == "PASS":
                    regressions.append(path)
                    regressed_by_cat[rec.category] += 1

        total_gradeable = len(self.records)
        baseline_passing = len(self.passing_paths)
        baseline_failing = len(self.failing_paths)
        candidate_failing = total_gradeable - total_candidate_passing
        net_gain = len(newly_passing) - len(regressions)

        target_fixed = fixed_by_cat.get(target_category, 0)
        purity = target_fixed / len(newly_passing) if newly_passing else 0.0
        baseline_target_total = self.category_counts.get(target_category, 0)
        target_recall = target_fixed / baseline_target_total if baseline_target_total > 0 else 0.0

        notes: list[str] = []
        is_verified = True

        if len(newly_passing) == 0:
            is_verified = False
            notes.append("No citations were fixed by the candidate proposal.")
        elif purity < min_purity:
            is_verified = False
            notes.append(
                f"Attribution purity ({purity * 100:.2f}%) is below minimum threshold ({min_purity * 100:.2f}%). "
                f"Fix predominantly impacts collateral categories instead of target `{target_category}`."
            )

        if len(regressions) > 0:
            is_verified = False
            notes.append(f"Proposal introduced {len(regressions)} regression(s) on previously passing citations.")
        else:
            notes.append("Zero regressions detected on previously passing citations.")

        if net_gain <= 0:
            is_verified = False
            notes.append(f"Non-positive net accuracy gain ({net_gain:+d}).")

        return CorrelationReport(
            target_category=target_category,
            baseline_total_gradeable=total_gradeable,
            baseline_passing_count=baseline_passing,
            baseline_failing_count=baseline_failing,
            candidate_passing_count=total_candidate_passing,
            candidate_failing_count=candidate_failing,
            newly_passing_count=len(newly_passing),
            regressions_count=len(regressions),
            net_gain=net_gain,
            target_fixed_count=target_fixed,
            purity=purity,
            target_recall=target_recall,
            fixed_by_category=dict(fixed_by_cat),
            regressed_by_category=dict(regressed_by_cat),
            newly_passing_paths=newly_passing,
            regressed_paths=regressions,
            is_verified=is_verified,
            verification_notes=notes,
        )


def verify_agent_proposal(
    candidate_citations: Sequence[Mapping[str, Any]],
    target_category: str,
    *,
    registry_path: Path | str | None = None,
    min_purity: float = 0.60,
) -> CorrelationReport:
    """High-level verification function to test whether an agent's proposal resolves its target failure category."""
    registry = FailureRegistry.load(registry_path)
    return registry.correlate(candidate_citations, target_category, min_purity=min_purity)
