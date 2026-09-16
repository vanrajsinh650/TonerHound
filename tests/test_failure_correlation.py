"""Unit tests for Failure Correlation Framework (EXP-005 Part 4 Pass 4)."""

import pytest
from pathlib import Path
from tonerhound.benchmark.correlation import (
    ErrorTaxonomy,
    FailureRegistry,
    verify_agent_proposal,
    CitationRecord,
    CorrelationReport,
)


@pytest.fixture
def registry() -> FailureRegistry:
    """Load baseline failure registry for real_ftx_full_corrupted."""
    reg_path = Path("experiments/EXP-005-ftx-failure-registry.json")
    assert reg_path.exists(), "Failure registry file must exist."
    return FailureRegistry.load(reg_path)


def test_registry_loading_and_baseline_counts(registry: FailureRegistry) -> None:
    """Verify registry loaded 26,583 gradeable citations with exact baseline numbers."""
    counts = registry.get_category_counts()

    # Total gradeable citations
    assert len(registry.records) == 26583

    # Baseline passing citations (IoU >= 0.50)
    assert len(registry.passing_paths) == 13257
    assert counts["CORRECT"] == 13257

    # Baseline failing citations (IoU < 0.50)
    assert len(registry.failing_paths) == 13326

    # Taxonomy counts
    assert counts["G1"] == 4682
    assert counts["G2"] == 6056
    assert counts["G3"] == 564
    assert counts["G4"] == 1
    assert counts["T1"] == 2023

    # Total sums to exactly 26,583
    total_sum = sum(counts.values())
    assert total_sum == 26583


def test_baseline_self_correlation_zero_deltas(registry: FailureRegistry) -> None:
    """Correlating baseline citations against baseline registry must yield 0 net gain and 0 regressions."""
    baseline_citations = []
    for p, r in registry.records.items():
        if r.pred_box is not None and r.pred_page is not None:
            baseline_citations.append({
                "field_path": p,
                "page": r.pred_page,
                "bbox": list(r.pred_box),
            })

    report = registry.correlate(baseline_citations, target_category="G1")
    assert report.newly_passing_count == 0
    assert report.regressions_count == 0
    assert report.net_gain == 0
    assert report.candidate_passing_count == 13257
    assert not report.is_verified  # No newly fixed citations


def test_agent_a_g1_fix_verification(registry: FailureRegistry) -> None:
    """Simulate Agent A proposing a true G1 fix: 50 G1 citations are fixed."""
    candidate_citations = []
    fixed_g1_count = 0

    for p, r in registry.records.items():
        if r.status == "PASS":
            # Keep passing citations passing
            candidate_citations.append({
                "field_path": p,
                "page": r.gt_page,
                "bbox": list(r.gt_box),
            })
        elif r.category == "G1" and fixed_g1_count < 50:
            # Fix this G1 citation by aligning to ground truth
            candidate_citations.append({
                "field_path": p,
                "page": r.gt_page,
                "bbox": list(r.gt_box),
            })
            fixed_g1_count += 1
        elif r.pred_box is not None and r.pred_page is not None:
            # Keep failing citations as they were
            candidate_citations.append({
                "field_path": p,
                "page": r.pred_page,
                "bbox": list(r.pred_box),
            })

    report = registry.correlate(candidate_citations, target_category="G1")
    assert report.is_verified
    assert report.newly_passing_count == 50
    assert report.target_fixed_count == 50
    assert report.regressions_count == 0
    assert report.purity == 1.0  # 100% purity
    assert report.fixed_by_category["G1"] == 50
    assert report.net_gain == 50


def test_agent_b_g3_fix_verification(registry: FailureRegistry) -> None:
    """Simulate Agent B proposing a true G3 fix: 30 G3 citations are fixed."""
    candidate_citations = []
    fixed_g3_count = 0

    for p, r in registry.records.items():
        if r.status == "PASS":
            candidate_citations.append({
                "field_path": p,
                "page": r.gt_page,
                "bbox": list(r.gt_box),
            })
        elif r.category == "G3" and fixed_g3_count < 30:
            # Fix this G3 citation
            candidate_citations.append({
                "field_path": p,
                "page": r.gt_page,
                "bbox": list(r.gt_box),
            })
            fixed_g3_count += 1
        elif r.pred_box is not None and r.pred_page is not None:
            candidate_citations.append({
                "field_path": p,
                "page": r.pred_page,
                "bbox": list(r.pred_box),
            })

    report = registry.correlate(candidate_citations, target_category="G3")
    assert report.is_verified
    assert report.newly_passing_count == 30
    assert report.target_fixed_count == 30
    assert report.regressions_count == 0
    assert report.purity == 1.0
    assert report.fixed_by_category["G3"] == 30
    assert report.net_gain == 30


def test_off_target_attribution_failure(registry: FailureRegistry) -> None:
    """If Agent A claims a G1 fix, but the fix actually only repairs G2 citations, purity fails."""
    candidate_citations = []
    fixed_g2_count = 0

    for p, r in registry.records.items():
        if r.status == "PASS":
            candidate_citations.append({
                "field_path": p,
                "page": r.gt_page,
                "bbox": list(r.gt_box),
            })
        elif r.category == "G2" and fixed_g2_count < 40:
            candidate_citations.append({
                "field_path": p,
                "page": r.gt_page,
                "bbox": list(r.gt_box),
            })
            fixed_g2_count += 1
        elif r.pred_box is not None and r.pred_page is not None:
            candidate_citations.append({
                "field_path": p,
                "page": r.pred_page,
                "bbox": list(r.pred_box),
            })

    # Test as G1 proposal
    report_g1 = registry.correlate(candidate_citations, target_category="G1", min_purity=0.60)
    assert not report_g1.is_verified
    assert report_g1.purity == 0.0  # 0 of 40 are G1
    assert report_g1.target_fixed_count == 0
    assert report_g1.fixed_by_category["G2"] == 40

    # If tested as G2 proposal, it succeeds
    report_g2 = registry.correlate(candidate_citations, target_category="G2", min_purity=0.60)
    assert report_g2.is_verified
    assert report_g2.purity == 1.0


def test_regression_detection(registry: FailureRegistry) -> None:
    """If a proposal causes previously passing citations to fail, regression is flagged."""
    candidate_citations = []
    broke_pass_count = 0

    for p, r in registry.records.items():
        if r.status == "PASS":
            if broke_pass_count < 5:
                # Deliberately corrupt this passing citation
                candidate_citations.append({
                    "field_path": p,
                    "page": r.gt_page,
                    "bbox": [0.0, 0.0, 0.01, 0.01],
                })
                broke_pass_count += 1
            else:
                candidate_citations.append({
                    "field_path": p,
                    "page": r.gt_page,
                    "bbox": list(r.gt_box),
                })
        elif r.category == "G1":
            candidate_citations.append({
                "field_path": p,
                "page": r.gt_page,
                "bbox": list(r.gt_box),
            })

    report = registry.correlate(candidate_citations, target_category="G1")
    assert not report.is_verified
    assert report.regressions_count == 5
    assert len(report.regressed_paths) == 5
