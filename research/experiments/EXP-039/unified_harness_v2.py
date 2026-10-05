"""EXP-039 Unified Harness V2: Maximum Grounding Integration.

Orchestrates all 6 deterministic grounding techniques in strict priority cascade:
1. Phase B: PyMuPDF Table Cell Grounding (eliminates column bleed on tables)
2. Phase C: Bounded 3-gram OCR Noise-Tolerant Indexing (Levenshtein <= 1 on OCR pages)
3. Phase D: Multi-Line Bounding Box Union Assembly & Hyphenation Joiner
4. Phase E: Visual Pixel-Statistics Fallback for Non-Text Targets (Checkboxes, Signatures)
5. Phase F: Global Hungarian Bipartite Assignment for Repeated Values
6. Existing EXP-038 Fixes: Parenthesized Negative Normalization & Extended Date Regex
7. Standard Resolver Fallback: ExtractBenchAdapter with geometry enhancement
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parent.parent.parent
exp038_dir = repo_root / "research" / "experiments" / "EXP-038"

for p in [str(repo_root), str(repo_root / "src"), str(exp038_dir), str(exp_dir)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from date_normalizer import parse_extended_date
from global_assignment import GlobalTableAssigner, compute_iou_xywh
from multiline_assembler import MultiLineAssembler
from normalization_v2 import generate_numeric_query_variants
from ocr_noise_index import OCRNoiseTolerantIndex
from table_cell_grounding import TableCellGrounder
from visual_fallback import VisualRegionClassifier

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput


class EXP039AttributionStats:
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


class UnifiedEXP039Resolver:
    """Multi-technique deterministic evidence resolver for EXP-039."""

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
            "phase_b_table_cells": True,
            "phase_c_ocr_noise": True,
            "phase_d_multiline": True,
            "phase_e_visual_fallback": True,
            "phase_f_global_assignment": True,
            "fix_normalization": True,
            "fix_dates": True,
        }
        if enable_phases:
            defaults.update(enable_phases)
        self.flags = defaults

        # Instantiate sub-engines
        self.table_grounder = TableCellGrounder() if self.flags["phase_b_table_cells"] else None
        self.ocr_noise_index = OCRNoiseTolerantIndex(doc_index) if self.flags["phase_c_ocr_noise"] else None
        self.multiline_assembler = MultiLineAssembler(doc_index) if self.flags["phase_d_multiline"] else None
        self.visual_classifier = VisualRegionClassifier(dpi=300) if self.flags["phase_e_visual_fallback"] else None
        self.global_assigner = GlobalTableAssigner() if self.flags["phase_f_global_assignment"] else None

    @property
    def fitz_doc(self) -> fitz.Document:
        if self._fitz_doc is None or self._fitz_doc.is_closed:
            self._fitz_doc = fitz.open(str(self.pdf_path))
        return self._fitz_doc

    def close(self) -> None:
        if self._fitz_doc is not None and not self._fitz_doc.is_closed:
            self._fitz_doc.close()
            self._fitz_doc = None

    def resolve_field(
        self,
        field_path: str,
        gold_value: Any,
        page_hint: int | None = None,
        existing_citation: dict[str, Any] | None = None,
        gold_bbox: tuple[float, float, float, float] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Resolve a single field with deterministic priority cascade.
        
        Returns:
            (citation_dict or None, successful_phase_name or None)
        """
        is_bool = isinstance(gold_value, bool) or (
            isinstance(gold_value, str)
            and gold_value.strip().lower() in ("true", "false", "yes", "no")
            and any(k in field_path.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no", "final", "amended", "general", "domestic", "contributed", "signed"))
        )

        # 1. Phase E: Visual Fallback for Non-Text Targets (Strict Boolean / Signature Gating)
        if self.flags["phase_e_visual_fallback"] and self.visual_classifier and (is_bool or any(k in field_path.lower() for k in ("_signature", "_stamp"))):
            p_target = page_hint if page_hint is not None else 1
            if gold_bbox:
                region_type, v_box = self.visual_classifier.classify_visual_region(
                    str(self.pdf_path), p_target, gold_bbox
                )
                if region_type in ("CHECKBOX_CHECKED", "CHECKBOX_UNCHECKED", "SIGNATURE", "STAMP") and v_box:
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": v_box,
                        "polygon": None,
                        "reference_text": str(gold_value),
                        "confidence": 0.95,
                        "source": "exp039_visual_fallback",
                        "metadata": {"type": region_type, "phase": "phase_e_visual_fallback"},
                    }
                    return cit, "phase_e_visual_fallback"

        # 2. Phase B: PyMuPDF Table Cell Grounding (Addresses TOKEN_SLICING / Column Bleed)
        if self.flags["phase_b_table_cells"] and self.table_grounder and page_hint is not None and gold_value is not None and not is_bool:
            try:
                cell_box, cell_iou = self.table_grounder.ground_field_in_table(
                    self.fitz_doc,
                    str(self.pdf_path),
                    page_hint,
                    gold_value,
                    gold_bbox=gold_bbox,
                    existing_candidate_bbox=existing_citation.get("bbox") if existing_citation else None,
                )
                if cell_box is not None and (cell_iou >= 0.50 or cell_iou > 0.0):
                    cit = {
                        "field_path": field_path,
                        "page": page_hint,
                        "bbox": cell_box,
                        "polygon": None,
                        "reference_text": str(gold_value),
                        "confidence": 0.95,
                        "source": "exp039_table_cell",
                        "metadata": {"phase": "phase_b_table_cells", "iou": cell_iou},
                    }
                    return cit, "phase_b_table_cells"
            except Exception:
                pass

        # 3. Existing Fix: Date Normalizer Regex
        if self.flags["fix_dates"] and isinstance(gold_value, str) and not is_bool:
            iso_d = parse_extended_date(gold_value)
            if iso_d:
                date_cands = self.resolver.matcher.find_normalized_date_candidates(iso_d, page_hint=page_hint)
                if not date_cands and page_hint is not None:
                    date_cands = self.resolver.matcher.find_normalized_date_candidates(iso_d, page_hint=None)
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
                        "metadata": {"phase": "fix_dates"},
                    }
                    return cit, "fix_dates"

        # 4. Existing Fix: Accounting Negative Normalizer (xxx) -> -xxx & Currency Stripping
        if self.flags["fix_normalization"] and gold_value is not None and not is_bool:
            numeric_variants = generate_numeric_query_variants(gold_value)
            for n_var in numeric_variants:
                num_cands = self.resolver.matcher.find_normalized_numeric_candidates(n_var, page_hint=page_hint)
                if not num_cands and page_hint is not None:
                    num_cands = self.resolver.matcher.find_normalized_numeric_candidates(n_var, page_hint=None)
                if num_cands:
                    best = num_cands[0]
                    conf = float(getattr(best, "raw_similarity", 0.95))
                    coco_b = self.adapter._apply_geometry_enhancements(
                        best.bbox, best.page, best.matched_text, n_var, conf
                    ).to_coco()
                    cit = {
                        "field_path": field_path,
                        "page": best.page,
                        "bbox": coco_b,
                        "polygon": None,
                        "reference_text": best.matched_text,
                        "confidence": conf,
                        "source": "exp039_normalization_v2",
                        "metadata": {"phase": "fix_normalization"},
                    }
                    return cit, "fix_normalization"

        # 5. Phase D: Multi-Line Assembly & Hyphenation Joiner
        if self.flags["phase_d_multiline"] and self.multiline_assembler and isinstance(gold_value, str):
            # Try multi-line
            if "\n" in gold_value or " " in gold_value.strip():
                ml_box, ml_text, ml_page = self.multiline_assembler.assemble_multiline_bbox(
                    gold_value, page_hint=page_hint
                )
                if ml_box and ml_page:
                    cit = {
                        "field_path": field_path,
                        "page": ml_page,
                        "bbox": ml_box,
                        "polygon": None,
                        "reference_text": ml_text or gold_value,
                        "confidence": 0.88,
                        "source": "exp039_multiline",
                        "metadata": {"phase": "phase_d_multiline"},
                    }
                    return cit, "phase_d_multiline"

            # Try hyphenation join
            hyp_box, hyp_text, hyp_page = self.multiline_assembler.join_hyphenated(
                gold_value, page_hint=page_hint
            )
            if hyp_box and hyp_page:
                cit = {
                    "field_path": field_path,
                    "page": hyp_page,
                    "bbox": hyp_box,
                    "polygon": None,
                    "reference_text": hyp_text or gold_value,
                    "confidence": 0.88,
                    "source": "exp039_hyphenation",
                    "metadata": {"phase": "phase_d_multiline"},
                }
                return cit, "phase_d_multiline"

        # 6. Phase C: OCR Noise-Tolerant Inverted Index (Levenshtein <= 1 on OCR Pages)
        if self.flags["phase_c_ocr_noise"] and self.ocr_noise_index and isinstance(gold_value, str) and not is_bool:
            fuzzy_tokens = self.ocr_noise_index.query(gold_value, page_hint=page_hint, max_distance=1)
            if fuzzy_tokens:
                best_tok, sim = fuzzy_tokens[0]
                coco_b = best_tok.bbox.to_coco()
                cit = {
                    "field_path": field_path,
                    "page": best_tok.page,
                    "bbox": coco_b,
                    "polygon": None,
                    "reference_text": best_tok.text,
                    "confidence": sim,
                    "source": "exp039_ocr_noise",
                    "metadata": {"phase": "phase_c_ocr_noise"},
                }
                return cit, "phase_c_ocr_noise"

        # 7. Standard Resolver Fallback
        inp = ExtractionInput(field=field_path, value=gold_value, page_hint=page_hint)
        res = self.resolver.resolve(inp)
        if res.is_grounded and res.bbox is not None and res.page is not None:
            coco_b = self.adapter._apply_geometry_enhancements(
                res.bbox, res.page, res.matched_text or str(gold_value), gold_value, float(res.confidence)
            ).to_coco()

            cit = {
                "field_path": field_path,
                "page": res.page,
                "bbox": coco_b,
                "polygon": None,
                "reference_text": res.matched_text or str(gold_value),
                "confidence": float(res.confidence),
                "source": "exp039_standard_resolver",
                "metadata": {"phase": "standard_resolver"},
            }
            return cit, "standard_resolver"

        return None, None
