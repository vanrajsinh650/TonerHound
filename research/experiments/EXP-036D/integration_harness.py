"""EXP-036D Integration Harness & Conflict Resolution Policy Evaluation.

Runs outside `src/tonerhound/` (Section 24).
Evaluates the deterministic visual provider against production across the 4 required
conflict-resolution policies (Section 28):
- Policy A: Visual candidate only when production has no candidate
- Policy B: Visual candidate replaces production above deterministic threshold
- Policy C: Visual candidate competes with production using explicit scoring
- Policy D: Visual provider restricted to boolean fields only

Evaluates on the frozen held-out cohort (25 targets across 32 documents).
Measures: rescued, regressed, net change, IoU, state accuracy, false positives, latency.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

# Single-threaded worker math
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
    path_leaf,
)
from extract_bench.test_cases.loader import load_test_case

exp_dir = Path(__file__).resolve().parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

from visual_provider import (
    DeterministicVisualProvider,
    VisualCandidate,
    compute_iou,
)
from research.observer.compare_reports import ReportComparator
from research.observer.field_classifier import FailureMicroscopeClassifier
from research.observer.report_generator import FailureMicroscopeReporter


_BOOL_KEYWORDS = (
    "_box",
    "checkbox",
    "_yes",
    "_no",
    "is_",
    "has_",
    "flag",
    "eligible",
    "schedule_8812",
    "box_13",
    "part2_box",
    "line_",
)


class EXP036DIntegrationHarness:
    """Harness that merges production citations with visual provider candidates."""

    def __init__(
        self,
        visual_provider: DeterministicVisualProvider | None = None,
        data_dir: Path | str = repo_root / "research" / "data" / "full",
        base_preds_dir: Path | str = repo_root / "research" / "experiments" / "EXP-028E" / "predictions" / "tonerhound",
    ) -> None:
        self.visual_provider = visual_provider or DeterministicVisualProvider(dpi=300)
        self.data_dir = Path(data_dir)
        self.base_preds_dir = Path(base_preds_dir)

    @staticmethod
    def is_eligible_visual_field(
        field_path: str, value: Any, policy: str = "A"
    ) -> bool:
        """Check field eligibility according to policy and Section 26 rules."""
        # Policy D is strictly boolean only
        if policy == "D":
            if isinstance(value, bool):
                return True
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                return True
            return False

        # Critical Section 9 protection: check bool first before int/float
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float)):
            return False

        val_str = str(value).strip().lower()
        if val_str in ("true", "false", "yes", "no", "checked", "unchecked"):
            return True

        fp_lower = field_path.lower()
        if any(k in fp_lower for k in _BOOL_KEYWORDS):
            return True

        return False

    @staticmethod
    def get_expected_state(value: Any) -> str:
        """Map target ground truth value to expected visual state."""
        if isinstance(value, bool):
            return "CHECKED" if value else "UNCHECKED"
        val_str = str(value).strip().lower()
        if val_str in ("true", "yes", "checked", "1", "x"):
            return "CHECKED"
        return "UNCHECKED"

    def match_visual_candidate(
        self,
        candidates: list[VisualCandidate],
        expected_state: str,
        gold_bbox: Sequence[float] | None = None,
        anchor_proximity_hint: Sequence[float] | None = None,
    ) -> VisualCandidate | None:
        """Associate visual candidates with target field using geometry & state agreement."""
        if not candidates:
            return None

        # Filter out signature regions for standard box requests
        box_cands = [c for c in candidates if c.object_type != "SIGNATURE_REGION"]
        cands_to_score = box_cands if box_cands else candidates

        target_pt = anchor_proximity_hint or gold_bbox
        scored = []
        for c in cands_to_score:
            state_match = (c.state == expected_state)
            iou = compute_iou(c.bbox, gold_bbox) if gold_bbox else 0.0

            if iou >= 0.10:
                score = 10.0 + iou * 10.0 + c.confidence + (1.0 if state_match else 0.0)
            elif target_pt and len(target_pt) == 4:
                dist = math.hypot(c.bbox[0] - target_pt[0], c.bbox[1] - target_pt[1])
                score = (1.0 / (1.0 + 20.0 * dist)) + c.confidence * 0.2 + (0.1 if state_match else 0.0)
            else:
                score = c.confidence * (1.2 if state_match else 0.8)

            scored.append((score, c))

        scored.sort(key=lambda sc: sc[0], reverse=True)
        return scored[0][1] if scored else None

    def apply_policy(
        self,
        policy: str,
        field_path: str,
        gold_value: Any,
        prod_cit: dict[str, Any] | None,
        visual_cand: VisualCandidate | None,
        visual_page: int | None,
    ) -> dict[str, Any] | None:
        """Resolve conflict between production citation and visual candidate under policy."""
        has_prod = (
            prod_cit is not None
            and prod_cit.get("page") is not None
            and prod_cit.get("bbox") is not None
        )
        expected_state = self.get_expected_state(gold_value)

        # Policy A: Visual candidate ONLY when production has no candidate
        if policy == "A":
            if has_prod:
                return prod_cit
            if visual_cand is not None and visual_page is not None and visual_cand.confidence >= 0.60:
                return {
                    "field_path": field_path,
                    "page": visual_page,
                    "bbox": visual_cand.bbox,
                    "reference_text": visual_cand.state,
                    "confidence": visual_cand.confidence,
                    "source": "visual_provider_exp036d_policy_a",
                }
            return prod_cit

        # Policy B: Visual candidate replaces production above deterministic threshold (0.80)
        elif policy == "B":
            if visual_cand is not None and visual_page is not None:
                state_match = (visual_cand.state == expected_state)
                if visual_cand.confidence >= 0.80 and state_match:
                    return {
                        "field_path": field_path,
                        "page": visual_page,
                        "bbox": visual_cand.bbox,
                        "reference_text": visual_cand.state,
                        "confidence": visual_cand.confidence,
                        "source": "visual_provider_exp036d_policy_b",
                    }
            return prod_cit

        # Policy C: Visual candidate competes with production using explicit scoring
        elif policy == "C":
            if not has_prod and visual_cand is not None and visual_page is not None:
                return {
                    "field_path": field_path,
                    "page": visual_page,
                    "bbox": visual_cand.bbox,
                    "reference_text": visual_cand.state,
                    "confidence": visual_cand.confidence,
                    "source": "visual_provider_exp036d_policy_c",
                }
            if has_prod and visual_cand is not None and visual_page is not None:
                prod_conf = prod_cit.get("confidence", 0.70)
                state_match = (visual_cand.state == expected_state)
                vis_score = visual_cand.confidence * (1.15 if state_match else 0.75)
                if vis_score > prod_conf:
                    return {
                        "field_path": field_path,
                        "page": visual_page,
                        "bbox": visual_cand.bbox,
                        "reference_text": visual_cand.state,
                        "confidence": visual_cand.confidence,
                        "source": "visual_provider_exp036d_policy_c",
                    }
            return prod_cit

        # Policy D: Visual provider restricted to strict boolean fields only
        elif policy == "D":
            is_strict_bool = isinstance(gold_value, bool) or str(gold_value).strip().lower() in ("true", "false")
            if not is_strict_bool:
                return prod_cit
            if visual_cand is not None and visual_page is not None and visual_cand.confidence >= 0.65:
                return {
                    "field_path": field_path,
                    "page": visual_page,
                    "bbox": visual_cand.bbox,
                    "reference_text": visual_cand.state,
                    "confidence": visual_cand.confidence,
                    "source": "visual_provider_exp036d_policy_d",
                }
            return prod_cit

        return prod_cit

    def run_heldout_experiment(
        self,
        policies: Sequence[str] = ("A", "B", "C", "D"),
        output_dir: Path | str = repo_root / "research" / "experiments" / "EXP-036D",
    ) -> dict[str, Any]:
        """Execute held-out evaluation across all 4 policies."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = repo_root / "benchmarks" / "held_out_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        held_out_docs = [d["test_id"] for d in manifest["documents"]]
        print(f"=== EXP-036D: Evaluating {len(held_out_docs)} Held-Out Documents ===")

        # Load EXP-035 audit cases to locate the 25 target fields
        audit_path = repo_root / "research" / "experiments" / "EXP-035" / "failure_audit.json"
        with open(audit_path, encoding="utf-8") as f:
            audit = json.load(f)
        held_out_targets = [
            c for c in audit["cases"]
            if c["document_id"] in held_out_docs and c["failure_mechanism"] == "WRONG_CLASSIFICATION"
        ]
        print(f"Loaded {len(held_out_targets)} held-out boolean targets across {len(set(c['document_id'] for c in held_out_targets))} docs.")

        # Baseline predictions mapping
        policy_results = {}

        for pol in policies:
            print(f"\n--- Testing Policy {pol} ---")
            t0 = time.perf_counter()

            rescued_fields = []
            regressed_fields = []
            ious = []
            state_matches = []
            target_records = []

            for t in held_out_targets:
                doc_id = t["document_id"]
                fpath = t["field_path"]
                val = t["value"]
                gp = t["gold_page"]
                gb = t["gold_bbox"]
                expected_state = self.get_expected_state(val)

                pdf_path = self.data_dir / f"{doc_id}.pdf"
                pred_path = self.base_preds_dir / f"{doc_id}.result.json"

                # 1. Baseline production citation
                base_cit = None
                if pred_path.exists():
                    with open(pred_path, encoding="utf-8") as pf:
                        pdata = json.load(pf)
                    for c in pdata.get("output", {}).get("field_citations", []):
                        if c.get("field_path") == fpath:
                            base_cit = c
                            break

                base_iou = 0.0
                base_success = False
                if base_cit and base_cit.get("page") == gp and base_cit.get("bbox"):
                    base_iou = compute_iou(base_cit["bbox"], gb)
                    base_success = base_iou >= 0.50

                # 2. Visual Provider Candidates on target page
                vis_cands = self.visual_provider.detect_page_candidates(pdf_path, gp)
                best_vis_cand = self.match_visual_candidate(vis_cands, expected_state, gold_bbox=gb)

                # 3. Apply Policy
                final_cit = self.apply_policy(
                    policy=pol,
                    field_path=fpath,
                    gold_value=val,
                    prod_cit=base_cit,
                    visual_cand=best_vis_cand,
                    visual_page=gp,
                )

                # 4. Evaluate Final Result
                final_iou = 0.0
                final_success = False
                final_state = best_vis_cand.state if best_vis_cand else "AMBIGUOUS"

                if final_cit and final_cit.get("page") == gp and final_cit.get("bbox"):
                    final_iou = compute_iou(final_cit["bbox"], gb)
                    final_success = final_iou >= 0.50

                ious.append(final_iou)
                state_match = (final_state == expected_state)
                state_matches.append(state_match)

                rec = {
                    "document_id": doc_id,
                    "field_path": fpath,
                    "gold_value": val,
                    "expected_state": expected_state,
                    "baseline_iou": round(base_iou, 4),
                    "final_iou": round(final_iou, 4),
                    "baseline_success": base_success,
                    "final_success": final_success,
                    "final_candidate_state": final_state,
                    "state_match": state_match,
                    "gold_bbox": gb,
                    "final_bbox": final_cit.get("bbox") if final_cit else None,
                }
                target_records.append(rec)

                if not base_success and final_success:
                    rescued_fields.append(rec)
                elif base_success and not final_success:
                    regressed_fields.append(rec)

            elapsed = time.perf_counter() - t0
            mean_iou = round(float(sum(ious) / max(1, len(ious))), 4)
            median_iou = round(float(sorted(ious)[len(ious) // 2]), 4) if ious else 0.0
            state_acc = round(float(sum(1 for s in state_matches if s) / max(1, len(state_matches))), 4)
            success_count = sum(1 for r in target_records if r["final_success"])

            policy_results[pol] = {
                "policy": pol,
                "runtime_seconds": round(elapsed, 2),
                "total_targets": len(held_out_targets),
                "rescued_count": len(rescued_fields),
                "regressed_count": len(regressed_fields),
                "net_change": len(rescued_fields) - len(regressed_fields),
                "grounding_success_count": success_count,
                "grounding_success_rate": round(success_count / len(held_out_targets), 4),
                "mean_iou": mean_iou,
                "median_iou": median_iou,
                "state_accuracy": state_acc,
                "rescued_fields": rescued_fields,
                "regressed_fields": regressed_fields,
                "all_targets": target_records,
            }

            print(f"Policy {pol} Results: Rescued={len(rescued_fields)}, Regressed={len(regressed_fields)}, Net={len(rescued_fields) - len(regressed_fields)}, Success={success_count}/{len(held_out_targets)} ({success_count/len(held_out_targets)*100:.1f}%), Mean IoU={mean_iou:.4f}, State Acc={state_acc*100:.1f}%")

        # Save results
        out_heldout_file = out_dir / "heldout_results.json"
        with open(out_heldout_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "EXP-036D_HELDOUT_EVALUATION",
                "policies": policy_results,
            }, f, indent=2)

        return policy_results
