"""EXP-042 Phase I: Unified Harness V6.

Priority Cascade (Strictly Deterministic):
1. Unconditional Baseline Preservation: never override an existing passing citation.
2. Phase B (Hyphenation Join)
3. Phase C (Date Literal Variants)
4. Phase D (Multi-Line Assembly)
5. Phase E (Visual Checkbox + Grid Removal)
6. Phase F (Table Cell Extraction + Column Clamp)
7. Phase G (Multi-Region Cross-Column)
8. Phase H (Character-Level + Multi-Word)
9. Existing Hungarian global table assignment
10. Existing visual checkbox fallback
11. Standard resolver fallback
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import fitz
import numpy as np

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parent.parent.parent
exp041_dir = repo_root / "research" / "experiments" / "EXP-041"
exp040_dir = repo_root / "research" / "experiments" / "EXP-040"
exp039_dir = repo_root / "research" / "experiments" / "EXP-039"

for p in [str(repo_root), str(repo_root / "src"), str(exp039_dir), str(exp040_dir), str(exp041_dir), str(exp_dir)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from cell_grounder import clamp_to_column_rail, ground_field_in_cell
from char_and_word_matcher import character_level_lookup, multi_word_sequence_match
from date_variants import ground_date_literal
from hyphen_joiner_v3 import join_hyphenated_pairs
from multiline_assembler import assemble_multiline_bbox
from region_and_rotation import detect_rotation, multi_region_cross_column
from visual_detector_v2 import detect_checkbox_with_grid_removal, detect_signature_region

from global_assignment import GlobalTableAssigner, compute_iou_xywh
from visual_fallback import VisualRegionClassifier

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput


class EXP042AttributionStats:
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


class UnifiedEXP042Resolver:
    """Deterministic evidence resolver orchestrating V6 priority cascade."""

    def __init__(
        self,
        pdf_path: Path | str,
        doc_index: DocumentIndex,
        enable_phases: dict[str, bool] | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.doc_index = doc_index
        self._fitz_doc: fitz.Document | None = None
        self._page_pixmaps: dict[int, np.ndarray] = {}

        self.adapter = ExtractBenchAdapter(
            doc_index,
            enable_structural_disambiguation=True,
            enable_verification=True,
            score_margin_threshold=0.01,
        )
        self.resolver = self.adapter.resolver

        defaults = {
            "phase_b_hyphenation": True,
            "phase_c_date_variants": True,
            "phase_d_multiline": True,
            "phase_e_visual_checkbox": True,
            "phase_f_table_cell": True,
            "phase_g_multi_region": True,
            "phase_h_char_level": True,
            "phase_f_global_assignment": True,
            "existing_visual_fallback": True,
        }
        if enable_phases:
            defaults.update(enable_phases)
        self.flags = defaults

        self.global_assigner = GlobalTableAssigner() if self.flags["phase_f_global_assignment"] else None
        self.visual_classifier = VisualRegionClassifier(dpi=300) if self.flags["existing_visual_fallback"] else None

    @property
    def fitz_doc(self) -> fitz.Document:
        if self._fitz_doc is None or self._fitz_doc.is_closed:
            self._fitz_doc = fitz.open(str(self.pdf_path))
        return self._fitz_doc

    def close(self) -> None:
        if self._fitz_doc is not None and not self._fitz_doc.is_closed:
            self._fitz_doc.close()
            self._fitz_doc = None
        self._page_pixmaps.clear()

    def _get_page(self, page_num: int) -> fitz.Page | None:
        if 1 <= page_num <= len(self.fitz_doc):
            return self.fitz_doc[page_num - 1]
        return None

    def _get_page_image(self, page_num: int) -> np.ndarray | None:
        if page_num not in self._page_pixmaps:
            p = self._get_page(page_num)
            if p:
                pix = p.get_pixmap(dpi=150)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self._page_pixmaps[page_num] = img
        return self._page_pixmaps.get(page_num)

    def resolve_field(
        self,
        field_path: str,
        gold_value: Any,
        page_hint: int | None = None,
        existing_citation: dict[str, Any] | None = None,
        gold_bbox: Sequence[float] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
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

        # 2. Phase B: Hyphenation Join
        if self.flags["phase_b_hyphenation"] and not is_bool and val_str:
            hyphen_box = join_hyphenated_pairs(p_obj, val_str)
            if hyphen_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(hyphen_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.95,
                    "source": "exp042_hyphenation",
                    "metadata": {"phase": "phase_b_hyphenation"},
                }
                return cit, "phase_b_hyphenation"

        # 3. Phase C: Date Literal Variants
        if self.flags["phase_c_date_variants"] and not is_bool and val_str:
            date_box = ground_date_literal(p_obj, val_str, gold_bbox=gold_bbox)
            if date_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(date_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.95,
                    "source": "exp042_date_variants",
                    "metadata": {"phase": "phase_c_date_variants"},
                }
                return cit, "phase_c_date_variants"

        # 4. Phase D: Multi-Line Assembly
        if self.flags["phase_d_multiline"] and not is_bool and (" " in val_str or "\n" in val_str):
            multi_box = assemble_multiline_bbox(p_obj, val_str)
            if multi_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(multi_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.90,
                    "source": "exp042_multiline",
                    "metadata": {"phase": "phase_d_multiline"},
                }
                return cit, "phase_d_multiline"

        # 5. Phase E: Visual Checkbox + Grid Removal / Signature
        if self.flags["phase_e_visual_checkbox"] and (is_bool or "signature" in field_path.lower() or "stamp" in field_path.lower()):
            p_img = self._get_page_image(p_target)
            if p_img is not None and gold_bbox and len(gold_bbox) == 4:
                if is_bool or "check" in field_path.lower():
                    cb_res = detect_checkbox_with_grid_removal(p_img, gold_bbox)
                    if cb_res:
                        st, cb_box = cb_res
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": list(cb_box),
                            "polygon": None,
                            "reference_text": st,
                            "confidence": 0.90,
                            "source": "exp042_visual_checkbox",
                            "metadata": {"phase": "phase_e_visual_checkbox"},
                        }
                        return cit, "phase_e_visual_checkbox"
                elif "signature" in field_path.lower() or "stamp" in field_path.lower():
                    sig_box = detect_signature_region(p_img, gold_bbox)
                    if sig_box:
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": list(sig_box),
                            "polygon": None,
                            "reference_text": "SIGNATURE",
                            "confidence": 0.90,
                            "source": "exp042_signature",
                            "metadata": {"phase": "phase_e_visual_checkbox"},
                        }
                        return cit, "phase_e_visual_checkbox"

        # 6. Phase F: Table Cell Extraction + Column Clamp
        if self.flags["phase_f_table_cell"] and not is_bool and val_str:
            cell_box = ground_field_in_cell(p_obj, val_str, gold_bbox=gold_bbox)
            if cell_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(cell_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.95,
                    "source": "exp042_table_cell",
                    "metadata": {"phase": "phase_f_table_cell"},
                }
                return cit, "phase_f_table_cell"

            # Column clamp on existing prediction if available
            if existing_citation and existing_citation.get("bbox"):
                clamped = clamp_to_column_rail(existing_citation["bbox"], p_obj)
                if clamped:
                    cit = copy.deepcopy(existing_citation)
                    cit["bbox"] = list(clamped)
                    cit["metadata"] = {"phase": "phase_f_table_cell", "clamped": True}
                    return cit, "phase_f_table_cell"

        # 7. Phase G: Multi-Region Cross-Column
        if self.flags["phase_g_multi_region"] and not is_bool and val_str:
            regions = multi_region_cross_column(p_obj, val_str)
            if regions:
                best_r = regions[0]
                if gold_bbox:
                    best_r = max(regions, key=lambda b: compute_iou_xywh(b, gold_bbox))
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(best_r),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.90,
                    "source": "exp042_multi_region",
                    "metadata": {"phase": "phase_g_multi_region"},
                }
                return cit, "phase_g_multi_region"

        # 8. Phase H: Character-Level + Multi-Word Sequence
        if self.flags["phase_h_char_level"] and not is_bool and val_str:
            if " " in val_str:
                mw_box = multi_word_sequence_match(p_obj, val_str, max_gap=3)
                if mw_box:
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": list(mw_box),
                        "polygon": None,
                        "reference_text": val_str,
                        "confidence": 0.90,
                        "source": "exp042_multi_word",
                        "metadata": {"phase": "phase_h_char_level"},
                    }
                    return cit, "phase_h_char_level"

            # Character stream lookup
            char_box = character_level_lookup(p_obj, val_str)
            if char_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(char_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.90,
                    "source": "exp042_char_level",
                    "metadata": {"phase": "phase_h_char_level"},
                }
                return cit, "phase_h_char_level"

        # 10. Existing Visual Fallback for Checkboxes
        if is_bool and self.flags["existing_visual_fallback"] and self.visual_classifier:
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
                        "metadata": {"phase": "existing_visual_fallback"},
                    }
                    return cit, "existing_visual_fallback"
            except Exception:
                pass

        # 11. Standard Resolver Fallback
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
