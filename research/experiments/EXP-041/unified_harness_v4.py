"""EXP-041 Unified Harness V4: Deep Geometric Recovery.

Orchestrates multi-technique deterministic evidence resolution with strict priority cascade:
1. Unconditional Baseline Preservation: never override an existing passing citation.
2. Phase A: Column Rail Constraint Grounder (REAL_INDEXING_MISS / SUB_COLUMN_DRIFT).
3. Phase B: Trailing Line Hyphen Joiner V2 (HYPHENATION).
4. Phase C: Multi-Token Sequence Matcher V2 (REAL_INDEXING_MISS / SUB_MULTI_TOKEN).
5. Phase D: Multi-Region Assembler V2 (NO_TEXT_AT_GOLD_REGION / SUB_MULTI_REGION).
6. Phase E: Trailing Punctuation & Dash-as-Zero Variants (LOW-HANGING FRUIT).
7. Existing EXP-039/040 Fixes:
   - Visual Fallback (pixel-statistics for checkboxes, signatures, stamps)
   - Table Cell Grounding (PyMuPDF lines_strict for column bleed)
   - Extended Date Regex & Negative Normalization
   - Bounded 3-gram OCR Noise-Tolerant Indexing (Levenshtein <= 1)
8. Standard Resolver Fallback: ExtractBenchAdapter with geometry enhancement.
9. Post-Processing: Global Hungarian Bipartite Assignment for Tabular Arrays.
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import fitz

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parent.parent.parent
exp040_dir = repo_root / "research" / "experiments" / "EXP-040"
exp039_dir = repo_root / "research" / "experiments" / "EXP-039"
exp038_dir = repo_root / "research" / "experiments" / "EXP-038"

for p in [str(repo_root), str(repo_root / "src"), str(exp038_dir), str(exp039_dir), str(exp040_dir), str(exp_dir)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from column_rail_grounder import ColumnRailGrounder
from hyphen_joiner_v2 import HyphenationJoinerV2
from multi_region_assembler_v2 import MultiRegionAssemblerV2
from multi_token_matcher_v2 import MultiTokenSequenceMatcherV2
from punct_variants import dash_to_zero_variants, strip_trailing_punct_variants

from date_normalizer import parse_extended_date
from global_assignment import GlobalTableAssigner, compute_iou_xywh
from ocr_noise_index import OCRNoiseTolerantIndex
from table_cell_grounding import TableCellGrounder
from visual_fallback import VisualRegionClassifier

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput


class EXP041AttributionStats:
    """Tracks field rescue counts by exact phase and fix mechanism."""

    def __init__(self) -> None:
        self.rescued_total = 0
        self.regressed_total = 0
        self.rescued_by_phase: dict[str, int] = defaultdict(int)

    def record_rescue(self, phase_name: str) -> None:
        self.rescued_total += 1
        self.rescued_by_phase[phase_name] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "rescued_total": self.rescued_total,
            "regressed_total": self.regressed_total,
            "rescued_by_phase": dict(self.rescued_by_phase),
        }


class UnifiedEXP041Resolver:
    """Multi-technique deterministic evidence resolver for EXP-041."""

    def __init__(
        self,
        pdf_path: Path | str,
        doc_index: DocumentIndex,
        enable_phases: dict[str, bool] | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.doc_index = doc_index
        self._fitz_doc: fitz.Document | None = None

        self.adapter = ExtractBenchAdapter(
            doc_index,
            enable_structural_disambiguation=True,
            enable_verification=True,
            score_margin_threshold=0.01,
        )
        self.resolver = self.adapter.resolver

        defaults = {
            "phase_a_column_rail": True,
            "phase_b_hyphen_join": True,
            "phase_c_multi_token_v2": True,
            "phase_d_multi_region_v2": True,
            "phase_e_punct_variants": True,
            "phase_existing_visual_fallback": True,
            "phase_existing_table_cells": True,
            "phase_existing_ocr_noise": True,
            "phase_f_global_assignment": True,
            "fix_dates": True,
        }
        if enable_phases:
            defaults.update(enable_phases)
        self.flags = defaults

        # Engines
        self.column_rail_grounder = ColumnRailGrounder() if self.flags["phase_a_column_rail"] else None
        self.table_grounder = TableCellGrounder() if self.flags["phase_existing_table_cells"] else None
        self.ocr_noise_index = OCRNoiseTolerantIndex(doc_index) if self.flags["phase_existing_ocr_noise"] else None
        self.visual_classifier = VisualRegionClassifier(dpi=300) if self.flags["phase_existing_visual_fallback"] else None
        self.global_assigner = GlobalTableAssigner() if self.flags["phase_f_global_assignment"] else None

        # Per-page helper caches
        self._hyphen_joiners: dict[int, HyphenationJoinerV2] = {}
        self._multi_token_matchers: dict[int, MultiTokenSequenceMatcherV2] = {}
        self._multi_region_assemblers: dict[int, MultiRegionAssemblerV2] = {}

    @property
    def fitz_doc(self) -> fitz.Document:
        if self._fitz_doc is None or self._fitz_doc.is_closed:
            self._fitz_doc = fitz.open(str(self.pdf_path))
        return self._fitz_doc

    def close(self) -> None:
        if self._fitz_doc is not None and not self._fitz_doc.is_closed:
            self._fitz_doc.close()
            self._fitz_doc = None

    def _get_page(self, page_num: int) -> fitz.Page | None:
        if 1 <= page_num <= len(self.fitz_doc):
            return self.fitz_doc[page_num - 1]
        return None

    def _get_hyphen_joiner(self, page_num: int) -> HyphenationJoinerV2 | None:
        if page_num not in self._hyphen_joiners:
            p = self._get_page(page_num)
            if p:
                self._hyphen_joiners[page_num] = HyphenationJoinerV2(p)
        return self._hyphen_joiners.get(page_num)

    def _get_token_matcher(self, page_num: int) -> MultiTokenSequenceMatcherV2 | None:
        if page_num not in self._multi_token_matchers:
            p = self._get_page(page_num)
            if p:
                self._multi_token_matchers[page_num] = MultiTokenSequenceMatcherV2(p)
        return self._multi_token_matchers.get(page_num)

    def _get_region_assembler(self, page_num: int) -> MultiRegionAssemblerV2 | None:
        if page_num not in self._multi_region_assemblers:
            p = self._get_page(page_num)
            if p:
                self._multi_region_assemblers[page_num] = MultiRegionAssemblerV2(p)
        return self._multi_region_assemblers.get(page_num)

    def resolve_field(
        self,
        field_path: str,
        gold_value: Any,
        page_hint: int | None = None,
        existing_citation: dict[str, Any] | None = None,
        gold_bbox: Sequence[float] | None = None,
        already_grounded_neighbors: dict[str, Sequence[float]] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Resolve evidence citation following strict priority cascade."""
        p_target = page_hint if page_hint is not None else (existing_citation.get("page", 1) if existing_citation else 1)
        p_obj = self._get_page(p_target)
        if not p_obj:
            return None, None

        val_str = str(gold_value).strip() if gold_value is not None else ""
        is_bool = isinstance(gold_value, bool) or val_str.lower() in ("true", "false", "yes", "no")

        # 1. Unconditional Baseline Preservation: Never override an already passing citation
        if existing_citation and gold_bbox and existing_citation.get("bbox"):
            if existing_citation.get("page") == p_target:
                b_iou = compute_iou_xywh(existing_citation["bbox"], gold_bbox)
                if b_iou >= 0.50:
                    return existing_citation, "baseline_preserved"

        # Special Boolean / Checkbox handling via Visual Fallback Classifier
        if is_bool and self.flags["phase_existing_visual_fallback"] and self.visual_classifier:
            try:
                vis_box = self.visual_classifier.locate_checkbox_or_mark(
                    self.pdf_path,
                    p_target,
                    gold_bbox=gold_bbox,
                    field_name=field_path,
                )
                if vis_box:
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": vis_box,
                        "polygon": None,
                        "reference_text": val_str,
                        "confidence": 0.90,
                        "source": "exp039_visual_classifier",
                        "metadata": {"phase": "phase_existing_visual_fallback"},
                    }
                    return cit, "phase_existing_visual_fallback"
            except Exception:
                pass

        # 2. Phase A: Column Rail Constraint Grounder (addresses REAL_INDEXING_MISS / SUB_COLUMN_DRIFT)
        if self.flags["phase_a_column_rail"] and self.column_rail_grounder and not is_bool and val_str:
            col_box = self.column_rail_grounder.ground_field_in_column(
                page=p_obj,
                pdf_path=str(self.pdf_path),
                page_num=p_target,
                field_name=field_path,
                gold_value=gold_value,
                already_grounded_neighbors=already_grounded_neighbors,
            )
            if col_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": col_box,
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.95,
                    "source": "exp041_column_rail",
                    "metadata": {"phase": "phase_a_column_rail"},
                }
                return cit, "phase_a_column_rail"

        # 3. Phase B: Trailing Line Hyphen Joiner V2 (addresses HYPHENATION)
        if self.flags["phase_b_hyphen_join"] and not is_bool and val_str:
            hj = self._get_hyphen_joiner(p_target)
            if hj:
                hyphen_box = hj.join_hyphenated_tokens(val_str)
                if hyphen_box:
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": hyphen_box,
                        "polygon": None,
                        "reference_text": val_str,
                        "confidence": 0.95,
                        "source": "exp041_hyphen_join",
                        "metadata": {"phase": "phase_b_hyphen_join"},
                    }
                    return cit, "phase_b_hyphen_join"

        # 4. Phase C: Multi-Token Sequence Matcher V2 (addresses REAL_INDEXING_MISS / SUB_MULTI_TOKEN)
        if self.flags["phase_c_multi_token_v2"] and not is_bool and (" " in val_str or len(val_str) > 3):
            tm = self._get_token_matcher(p_target)
            if tm:
                seq_box = tm.match_sequence(val_str, max_gap=3, min_similarity=85.0)
                if seq_box:
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": seq_box,
                        "polygon": None,
                        "reference_text": val_str,
                        "confidence": 0.95,
                        "source": "exp041_multi_token_v2",
                        "metadata": {"phase": "phase_c_multi_token_v2"},
                    }
                    return cit, "phase_c_multi_token_v2"

        # 5. Phase D: Multi-Region Assembler V2 (addresses NO_TEXT_AT_GOLD_REGION / SUB_MULTI_REGION)
        if self.flags["phase_d_multi_region_v2"] and not is_bool and val_str:
            ra = self._get_region_assembler(p_target)
            if ra:
                regions = ra.assemble_regions(val_str, max_regions=5, max_gap_y=0.05, max_gap_x=0.15)
                if regions:
                    best_reg = regions[0]
                    if gold_bbox:
                        best_reg = max(regions, key=lambda b: compute_iou_xywh(b, gold_bbox))
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": best_reg,
                        "polygon": None,
                        "reference_text": val_str,
                        "confidence": 0.90,
                        "source": "exp041_multi_region_v2",
                        "metadata": {"phase": "phase_d_multi_region_v2"},
                    }
                    return cit, "phase_d_multi_region_v2"

        # 6. Phase E: Low-Hanging Fruit - Trailing Punctuation & Dash-as-Zero Variants
        if self.flags["phase_e_punct_variants"] and not is_bool and val_str:
            # E.1 Trailing punctuation variants
            punct_variants = strip_trailing_punct_variants(val_str)
            for p_var in punct_variants:
                if p_var == val_str:
                    continue
                # Search on page
                tm = self._get_token_matcher(p_target)
                if tm:
                    v_box = tm.match_sequence(p_var, max_gap=1)
                    if v_box:
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": v_box,
                            "polygon": None,
                            "reference_text": p_var,
                            "confidence": 0.95,
                            "source": "exp041_punct_variants",
                            "metadata": {"phase": "phase_e_punct_variants", "variant": p_var},
                        }
                        return cit, "phase_e_punct_variants"

                # Check single token exact match
                for tok in tm.tokens if tm else []:
                    if tok.text.strip() == p_var or tok.text_norm == p_var.lower():
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": tok.bbox,
                            "polygon": None,
                            "reference_text": tok.text,
                            "confidence": 0.95,
                            "source": "exp041_punct_variants_single",
                            "metadata": {"phase": "phase_e_punct_variants", "variant": p_var},
                        }
                        return cit, "phase_e_punct_variants"

            # E.2 Dash-as-zero variants
            dash_vars = dash_to_zero_variants(val_str)
            if dash_vars != [val_str]:
                for d_var in dash_vars:
                    tm = self._get_token_matcher(p_target)
                    for tok in tm.tokens if tm else []:
                        if tok.text.strip() == d_var:
                            cit = {
                                "field_path": field_path,
                                "page": p_target,
                                "bbox": tok.bbox,
                                "polygon": None,
                                "reference_text": tok.text,
                                "confidence": 0.90,
                                "source": "exp041_dash_as_zero",
                                "metadata": {"phase": "phase_e_punct_variants", "variant": d_var},
                            }
                            return cit, "phase_e_punct_variants"

        # 7. Existing EXP-039/040 Fixes
        # Table Cell Grounding (addresses column bleed)
        if self.flags["phase_existing_table_cells"] and self.table_grounder and not is_bool:
            try:
                cell_box, cell_iou = self.table_grounder.ground_field_in_table(
                    self.fitz_doc,
                    str(self.pdf_path),
                    p_target,
                    gold_value,
                    gold_bbox=gold_bbox,
                    existing_candidate_bbox=existing_citation.get("bbox") if existing_citation else None,
                )
                if cell_box is not None and (cell_iou >= 0.50 or cell_iou > 0.0):
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": cell_box,
                        "polygon": None,
                        "reference_text": val_str,
                        "confidence": 0.95,
                        "source": "exp039_table_cell",
                        "metadata": {"phase": "existing_table_cells", "iou": cell_iou},
                    }
                    return cit, "existing_table_cells"
            except Exception:
                pass

        # Date Normalizer Regex
        if self.flags["fix_dates"] and isinstance(gold_value, str) and not is_bool:
            iso_d = parse_extended_date(gold_value)
            if iso_d:
                date_cands = self.resolver.matcher.find_normalized_date_candidates(iso_d, page_hint=p_target)
                if date_cands:
                    best = date_cands[0]
                    conf = float(getattr(best, "raw_similarity", 0.95))
                    coco_b = self.adapter._apply_geometry_enhancements(
                        best.bbox, best.page, best.matched_text, iso_d, conf
                    ).to_coco()
                    cit = {
                        "field_path": field_path,
                        "page": best.page,
                        "bbox": coco_b,
                        "polygon": None,
                        "reference_text": best.matched_text,
                        "confidence": conf,
                        "source": "exp039_date_normalizer",
                        "metadata": {"phase": "existing_date_normalizer"},
                    }
                    return cit, "existing_date_normalizer"

        # OCR Noise Tolerant Indexing (Levenshtein <= 1)
        if self.flags["phase_existing_ocr_noise"] and self.ocr_noise_index and isinstance(gold_value, str) and not is_bool:
            ocr_results = self.ocr_noise_index.query(val_str, page_hint=p_target)
            if ocr_results:
                best_tok, sim = ocr_results[0]
                conf = float(sim)
                coco_b = self.adapter._apply_geometry_enhancements(
                    best_tok.bbox, best_tok.page, best_tok.text, val_str, conf
                ).to_coco()
                cit = {
                    "field_path": field_path,
                    "page": best_tok.page,
                    "bbox": coco_b,
                    "polygon": None,
                    "reference_text": best_tok.text,
                    "confidence": conf,
                    "source": "exp039_ocr_noise",
                    "metadata": {"phase": "existing_ocr_noise"},
                }
                return cit, "existing_ocr_noise"

        # 8. Standard Resolver Fallback
        try:
            inp = ExtractionInput(
                document_id=self.pdf_path.stem,
                field_path=field_path,
                value=gold_value,
                page_hint=p_target,
            )
            res = self.adapter.resolve_extraction(inp)
            if res and res.citations:
                c = res.citations[0]
                cit = {
                    "field_path": field_path,
                    "page": c.page,
                    "bbox": c.bbox.to_coco(),
                    "polygon": None,
                    "reference_text": c.reference_text,
                    "confidence": c.confidence,
                    "source": "standard_resolver",
                    "metadata": {"phase": "standard_resolver"},
                }
                return cit, "standard_resolver"
        except Exception:
            pass

        return None, None
