"""Failure Microscope Report Comparison Tool.

Computes exhaustive, field-level before/after diffs between two microscope
runs (e.g. baseline vs experiment) according to Sections 29, 30, and 35.

Categorizes every field into:
- RESCUED
- REGRESSED
- UNCHANGED_CORRECT
- UNCHANGED_WRONG
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class ReportComparator:
    """Compares baseline vs experiment microscope reports."""

    def __init__(self, base_report_dir: Path | str, exp_report_dir: Path | str) -> None:
        self.base_dir = Path(base_report_dir)
        self.exp_dir = Path(exp_report_dir)

    def load_field_records(self, dir_path: Path) -> dict[str, dict[str, Any]]:
        """Load field records keyed by (document_id, field_path)."""
        field_file = dir_path / "field_level.json"
        if not field_file.exists():
            raise FileNotFoundError(f"Missing field_level.json in {dir_path}")

        with open(field_file, encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("fields", [])
        return {f"{r['document_id']}::{r['field_path']}": r for r in records}

    def compare(self) -> dict[str, Any]:
        """Compute complete differential comparison."""
        base_records = self.load_field_records(self.base_dir)
        exp_records = self.load_field_records(self.exp_dir)

        all_keys = sorted(set(base_records.keys()) | set(exp_records.keys()))

        rescued: list[dict[str, Any]] = []
        regressed: list[dict[str, Any]] = []
        unchanged_correct: list[dict[str, Any]] = []
        unchanged_wrong: list[dict[str, Any]] = []

        docs_improved: set[str] = set()
        docs_regressed: set[str] = set()

        for k in all_keys:
            b_rec = base_records.get(k)
            e_rec = exp_records.get(k)

            if not b_rec or not e_rec:
                continue

            doc_id = b_rec["document_id"]
            fpath = b_rec["field_path"]
            b_success = b_rec.get("grounding_success", False)
            e_success = e_rec.get("grounding_success", False)
            b_iou = b_rec.get("iou", 0.0)
            e_iou = e_rec.get("iou", 0.0)
            b_class = b_rec.get("failure_class", "UNKNOWN")
            e_class = e_rec.get("failure_class", "UNKNOWN")

            diff_entry = {
                "field_key": k,
                "document_id": doc_id,
                "field_path": fpath,
                "baseline_prediction": b_rec.get("predicted_value"),
                "exp_prediction": e_rec.get("predicted_value"),
                "baseline_bbox": b_rec.get("predicted_bbox"),
                "exp_bbox": e_rec.get("predicted_bbox"),
                "gold_bbox": b_rec.get("gold_bbox"),
                "baseline_iou": b_iou,
                "exp_iou": e_iou,
                "baseline_class": b_class,
                "exp_class": e_class,
                "reason_for_change": f"Transitioned from {b_class} to {e_class} (IoU {b_iou:.4f} -> {e_iou:.4f})",
            }

            if not b_success and e_success:
                diff_entry["category"] = "RESCUED"
                rescued.append(diff_entry)
                docs_improved.add(doc_id)
            elif b_success and not e_success:
                diff_entry["category"] = "REGRESSED"
                regressed.append(diff_entry)
                docs_regressed.add(doc_id)
            elif b_success and e_success:
                diff_entry["category"] = "UNCHANGED_CORRECT"
                unchanged_correct.append(diff_entry)
            else:
                diff_entry["category"] = "UNCHANGED_WRONG"
                unchanged_wrong.append(diff_entry)

        net_field_change = len(rescued) - len(regressed)

        # Failure class changes
        base_class_counts: dict[str, int] = defaultdict(int)
        for r in base_records.values():
            if not r.get("grounding_success", False):
                base_class_counts[r.get("failure_class", "UNKNOWN")] += 1

        exp_class_counts: dict[str, int] = defaultdict(int)
        for r in exp_records.values():
            if not r.get("grounding_success", False):
                exp_class_counts[r.get("failure_class", "UNKNOWN")] += 1

        classes_reduced: dict[str, int] = {}
        classes_increased: dict[str, int] = {}
        all_classes = set(base_class_counts.keys()) | set(exp_class_counts.keys())
        for c in sorted(all_classes):
            b_cnt = base_class_counts[c]
            e_cnt = exp_class_counts[c]
            diff = b_cnt - e_cnt
            if diff > 0:
                classes_reduced[c] = diff
            elif diff < 0:
                classes_increased[c] = -diff

        return {
            "total_fields": len(all_keys),
            "rescued_count": len(rescued),
            "regressed_count": len(regressed),
            "net_field_change": net_field_change,
            "unchanged_correct_count": len(unchanged_correct),
            "unchanged_wrong_count": len(unchanged_wrong),
            "documents_improved_count": len(docs_improved),
            "documents_regressed_count": len(docs_regressed),
            "documents_improved": sorted(docs_improved),
            "documents_regressed": sorted(docs_regressed),
            "failure_classes_reduced": classes_reduced,
            "failure_classes_increased": classes_increased,
            "rescued_fields": rescued,
            "regressed_fields": regressed,
        }

    def generate_diff_markdown(self, diff: dict[str, Any]) -> str:
        """Format comparison results as detailed markdown report."""
        lines = [
            "# TonerHound Experiment Comparison & Difference Analysis",
            "",
            "## 1. Overall Field Transitions",
            f"- **Fields Rescued**: {diff['rescued_count']}",
            f"- **Fields Regressed**: {diff['regressed_count']}",
            f"- **Net Field Change**: {diff['net_field_change']:+d}",
            f"- **Unchanged Correct**: {diff['unchanged_correct_count']}",
            f"- **Unchanged Wrong**: {diff['unchanged_wrong_count']}",
            f"- **Documents Improved**: {diff['documents_improved_count']}",
            f"- **Documents Regressed**: {diff['documents_regressed_count']}",
            "",
            "## 2. Failure Class Delta",
            "",
            "| Failure Class | Baseline Count | Experiment Count | Delta |",
            "|:--------------|---------------:|-----------------:|------:|",
        ]

        all_classes = sorted(
            set(diff["failure_classes_reduced"].keys())
            | set(diff["failure_classes_increased"].keys())
        )
        for c in all_classes:
            red = diff["failure_classes_reduced"].get(c, 0)
            inc = diff["failure_classes_increased"].get(c, 0)
            delta = red - inc
            lines.append(f"| `{c}` | - | - | {delta:+d} |")

        if diff["rescued_fields"]:
            lines.extend([
                "",
                "## 3. Rescued Fields Sample",
                "",
                "| Document | Field Path | Baseline Class -> New Class | Base IoU -> Exp IoU |",
                "|:---------|:-----------|:----------------------------|:--------------------:|",
            ])
            for r in diff["rescued_fields"][:20]:
                lines.append(
                    f"| `{r['document_id']}` | `{r['field_path']}` | `{r['baseline_class']}` -> `{r['exp_class']}` | {r['baseline_iou']:.4f} -> {r['exp_iou']:.4f} |"
                )

        if diff["regressed_fields"]:
            lines.extend([
                "",
                "## 4. Regressed Fields Sample",
                "",
                "| Document | Field Path | Baseline Class -> New Class | Base IoU -> Exp IoU |",
                "|:---------|:-----------|:----------------------------|:--------------------:|",
            ])
            for r in diff["regressed_fields"]:
                lines.append(
                    f"| `{r['document_id']}` | `{r['field_path']}` | `{r['baseline_class']}` -> `{r['exp_class']}` | {r['baseline_iou']:.4f} -> {r['exp_iou']:.4f} |"
                )

        return "\n".join(lines) + "\n"
