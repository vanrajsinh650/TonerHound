"""Failure Microscope Report Generator.

Transforms field-level classification records into authoritative, macro-weighted
diagnostic reports conforming to Sections 10, 11, 12, 13 of the directive.

Outputs:
- field_level.json
- failure_summary.json
- document_breakdown.json
- family_breakdown.json
- recommendations.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from research.observer.field_classifier import (
    AUDIT_ONLY_CLASSES,
    PRODUCTION_FAILURE_CLASSES,
    FieldClassificationResult,
)


def _classify_doc_family(doc_id: str) -> str:
    """Classify document family based on ID."""
    tid = doc_id.lower()
    if "w14" in tid or "w-14" in tid:
        return "TEXAS_RRC_W14"
    if "h12" in tid or "h-12" in tid or "h9" in tid or "h-9" in tid:
        return "TEXAS_RRC_H12"
    if any(k in tid for k in ("w-1", "w-2", "p4", "p-4", "2a", "w2-")):
        return "TEXAS_RRC_OTHER"
    if any(k in tid for k in ("arif", "bar-lev", "becerra", "1040", "8879", "8949", "8960", "8812", "k-1", "k1")):
        return "IRS_TAX_RETURNS"
    if any(k in tid for k in ("sec", "13f", "10-k", "10-q", "ishares", "nport")):
        return "SEC_FINANCIAL_REGULATORY"
    if "corrupted" in tid:
        return "OCR_CORRUPTED_DOCUMENT"
    return "OTHER_DOCUMENTS"


class FailureMicroscopeReporter:
    """Generates structured failure reports and macro-weighted opportunities."""

    def __init__(self, run_id: str, output_dir: Path | str) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(
        self,
        field_records: Sequence[FieldClassificationResult | dict[str, Any]],
        total_benchmark_docs: int = 236,
        doc_total_fields: dict[str, int] | None = None,
    ) -> dict[str, Path]:
        """Compute summaries and write all report files deterministically."""
        # Standardize records to dicts
        records: list[dict[str, Any]] = [
            r.to_dict() if isinstance(r, FieldClassificationResult) else dict(r)
            for r in field_records
        ]

        # 1. Deterministic sort of field records: document_id, field_path
        records.sort(key=lambda r: (r["document_id"], r["field_path"]))

        # Aggregate counts
        doc_fields_map: dict[str, int] = defaultdict(int)
        for r in records:
            doc_fields_map[r["document_id"]] += 1

        if doc_total_fields is None:
            doc_total_fields = dict(doc_fields_map)

        # Segregate production failures vs audit / resolved
        total_fields = len(records)
        failures = [r for r in records if not r.get("grounding_success", False)]
        resolved = [r for r in records if r.get("grounding_success", False)]

        class_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        class_docs: dict[str, set[str]] = defaultdict(set)
        doc_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for r in failures:
            cls = r["failure_class"]
            class_records[cls].append(r)
            class_docs[cls].add(r["document_id"])
            doc_class_counts[r["document_id"]][cls] += 1

        # Macro-weighted opportunity calculation (Section 12)
        # Macro Opportunity (pp) = (100 / total_benchmark_docs) * sum(k_{d, C} / T_d)
        class_macro_pp: dict[str, float] = defaultdict(float)
        denominator_docs = max(1, total_benchmark_docs)

        for doc_id, cls_counts in doc_class_counts.items():
            t_d = max(1, doc_total_fields.get(doc_id, sum(cls_counts.values())))
            for cls, k_dc in cls_counts.items():
                class_macro_pp[cls] += (100.0 / denominator_docs) * (k_dc / t_d)

        # Build failure summary ranked strictly by macro opportunity descending (Section 13)
        # Audit-only classes are excluded from production failure ranking
        prod_classes = [c for c in class_records.keys() if c not in AUDIT_ONLY_CLASSES]
        prod_classes.sort(
            key=lambda c: (
                round(class_macro_pp[c], 6),
                len(class_records[c]),
                len(class_docs[c]),
                c,
            ),
            reverse=True,
        )

        total_prod_failures = sum(len(class_records[c]) for c in prod_classes)
        total_macro_opportunity = sum(class_macro_pp[c] for c in prod_classes)

        # Instantiate RealisticEstimator for calibrated expectations (Microscope V4)
        from research.observer.realistic_estimator import RealisticEstimator
        estimator = RealisticEstimator()

        summary_rows = []
        for rank, cls in enumerate(prod_classes, start=1):
            f_count = len(class_records[cls])
            d_count = len(class_docs[cls])
            macro_pp = round(class_macro_pp[cls], 4)
            pct_failures = round((f_count / max(1, total_prod_failures)) * 100, 2)
            pct_macro = (
                round((macro_pp / max(1e-6, total_macro_opportunity)) * 100, 2)
                if total_macro_opportunity > 0
                else 0.0
            )

            # Compute realistic estimates
            est = estimator.estimate_class(
                failure_class=cls,
                field_count=f_count,
                document_count=d_count,
                total_benchmark_docs=denominator_docs,
                theoretical_ceiling_pp=macro_pp,
            )

            summary_rows.append({
                "rank": rank,
                "failure_class": cls,
                "fields_affected": f_count,
                "documents_affected": d_count,
                "percentage_of_failures": pct_failures,
                "macro_weighted_opportunity_pp": macro_pp,
                "THEORETICAL_CEILING_pp": est.theoretical_ceiling_pp,
                "REALISTIC_RECOVERY_RATE": est.realistic_recovery_rate,
                "REALISTIC_RECOVERY_RATE_pct": f"{est.realistic_recovery_rate * 100:.2f}%",
                "REALISTIC_EXPECTED_GAIN_pp": est.realistic_expected_gain_pp,
                "fix_type": est.fix_type,
                "confidence": est.confidence,
                "source_evidence": est.source_evidence,
                "recommended_next_step": est.recommended_next_step,
                "formatted_block": est.format_text_block(),
                "percentage_of_macro_opportunity": pct_macro,
            })

        # Summary dict
        failure_summary = {
            "run_id": self.run_id,
            "total_fields_evaluated": total_fields,
            "total_grounding_success": len(resolved),
            "total_grounding_failures": len(failures),
            "production_failure_count": total_prod_failures,
            "total_macro_opportunity_pp": round(total_macro_opportunity, 4),
            "benchmark_document_denominator": denominator_docs,
            "audit_only_records": {
                c: len([r for r in records if r["failure_class"] == c])
                for c in AUDIT_ONLY_CLASSES
            },
            "ranking": summary_rows,
        }

        # Document breakdown
        doc_breakdown = {}
        for doc_id, cls_counts in sorted(doc_class_counts.items()):
            doc_breakdown[doc_id] = {
                "total_fields": doc_total_fields.get(doc_id, sum(cls_counts.values())),
                "total_failures": sum(cls_counts.values()),
                "failure_classes": dict(sorted(cls_counts.items(), key=lambda x: x[1], reverse=True)),
            }

        # Family breakdown
        family_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        family_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "failures": 0, "success": 0})
        for r in records:
            fam = _classify_doc_family(r["document_id"])
            family_totals[fam]["total"] += 1
            if r.get("grounding_success", False):
                family_totals[fam]["success"] += 1
            else:
                family_totals[fam]["failures"] += 1
                family_class_counts[fam][r["failure_class"]] += 1

        family_breakdown = {}
        for fam in sorted(family_totals.keys()):
            family_breakdown[fam] = {
                "total_fields": family_totals[fam]["total"],
                "total_success": family_totals[fam]["success"],
                "total_failures": family_totals[fam]["failures"],
                "failure_classes": dict(sorted(family_class_counts[fam].items(), key=lambda x: x[1], reverse=True)),
            }

        # Recommendations markdown derived purely from macro ranking
        recs_content = self._generate_recommendations_markdown(failure_summary, summary_rows)

        # Write files
        paths: dict[str, Path] = {}

        p_field = self.output_dir / "field_level.json"
        with open(p_field, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "fields": records}, f, indent=2)
        paths["field_level"] = p_field

        p_summary = self.output_dir / "failure_summary.json"
        with open(p_summary, "w", encoding="utf-8") as f:
            json.dump(failure_summary, f, indent=2)
        paths["failure_summary"] = p_summary

        p_doc = self.output_dir / "document_breakdown.json"
        with open(p_doc, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "documents": doc_breakdown}, f, indent=2)
        paths["document_breakdown"] = p_doc

        p_fam = self.output_dir / "family_breakdown.json"
        with open(p_fam, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "families": family_breakdown}, f, indent=2)
        paths["family_breakdown"] = p_fam

        p_recs = self.output_dir / "recommendations.md"
        with open(p_recs, "w", encoding="utf-8") as f:
            f.write(recs_content)
        paths["recommendations"] = p_recs

        return paths

    def _generate_recommendations_markdown(
        self, summary: dict[str, Any], ranking: list[dict[str, Any]]
    ) -> str:
        """Deterministically format recommendations from macro data."""
        lines = [
            f"# TonerHound Failure Microscope Diagnostic Report — {self.run_id}",
            "",
            "## 1. Executive Summary",
            f"- **Evaluated Fields**: {summary['total_fields_evaluated']}",
            f"- **Grounding Successes**: {summary['total_grounding_success']}",
            f"- **Grounding Failures**: {summary['total_grounding_failures']}",
            f"- **Total Macro-Weighted Opportunity**: +{summary['total_macro_opportunity_pp']:.4f} pp",
            "",
            "## 2. Macro-Weighted Failure Ranking",
            "",
            "| Rank | Failure Class | Fields | Documents | % of Failures | Macro Opp (pp) | % of Macro Opp |",
            "|:----:|:--------------|-------:|----------:|--------------:|---------------:|---------------:|",
        ]
        for row in ranking:
            lines.append(
                f"| {row['rank']} | `{row['failure_class']}` | {row['fields_affected']} | "
                f"{row['documents_affected']} | {row['percentage_of_failures']:.2f}% | "
                f"+{row['macro_weighted_opportunity_pp']:.4f} pp | {row['percentage_of_macro_opportunity']:.2f}% |"
            )

        lines.extend([
            "",
            "## 3. Calibrated Failure Class Evaluations (Microscope V4)",
            "",
        ])

        for row in ranking:
            fb = row.get("formatted_block")
            if fb:
                lines.extend([
                    "```",
                    fb,
                    "```",
                    "",
                ])

        lines.extend([
            "## 4. Evidence-Based Research Recommendations",
            "",
        ])

        if ranking:
            # Sort by realistic expected gain descending for actual recommendation priority
            realistic_sorted = sorted(ranking, key=lambda r: r.get("REALISTIC_EXPECTED_GAIN_pp", 0.0), reverse=True)
            top1 = realistic_sorted[0]
            lines.append(
                f"### Primary Realistic Research Target: `{top1['failure_class']}`\n"
                f"- **Realistic Expected Gain**: +{top1.get('REALISTIC_EXPECTED_GAIN_pp', 0.0):.4f} pp Word F1 (Theoretical Ceiling: +{top1['macro_weighted_opportunity_pp']:.4f} pp).\n"
                f"- **Empirical Recovery Rate**: {top1.get('REALISTIC_RECOVERY_RATE_pct', '10.0%')} (Confidence: {top1.get('confidence', 'LOW')}, Source: {top1.get('source_evidence', 'N/A')}).\n"
                f"- **Action**: {top1.get('recommended_next_step', 'Targeted audit')}.\n"
            )
            if len(realistic_sorted) > 1:
                top2 = realistic_sorted[1]
                lines.append(
                    f"### Secondary Realistic Research Target: `{top2['failure_class']}`\n"
                    f"- **Realistic Expected Gain**: +{top2.get('REALISTIC_EXPECTED_GAIN_pp', 0.0):.4f} pp Word F1 (Theoretical Ceiling: +{top2['macro_weighted_opportunity_pp']:.4f} pp).\n"
                    f"- **Empirical Recovery Rate**: {top2.get('REALISTIC_RECOVERY_RATE_pct', '10.0%')} (Confidence: {top2.get('confidence', 'LOW')}, Source: {top2.get('source_evidence', 'N/A')}).\n"
                    f"- **Action**: {top2.get('recommended_next_step', 'Targeted audit')}.\n"
                )

        return "\n".join(lines) + "\n"
