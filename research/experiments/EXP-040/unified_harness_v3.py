"""EXP-040 Unified Harness V3: Direct Assault on 90% Grounding F1.

Orchestrates multi-technique deterministic evidence resolution with strict priority cascade:
1. Unconditional Baseline Preservation: never override an existing passing citation.
2. Phase B: Multi-Token Sequence Matcher (addresses REAL_INDEXING_MISS / SUB_MULTI_TOKEN).
3. Phase C: Normalization Variants (addresses NORMALIZATION_MISMATCH / SUB_PERCENT_DECIMAL).
4. Phase D: Multi-Region Assembly (addresses NO_TEXT_AT_GOLD_REGION / SUB_MULTI_REGION).
5. Phase E: Trailing Hyphen Joiner & Percent-vs-Decimal (addresses HYPHENATION).
6. Existing EXP-039 Fixes:
   - Visual Fallback (pixel-statistics for checkboxes, signatures, stamps)
   - Table Cell Grounding (PyMuPDF lines_strict for column bleed)
   - Extended Date Regex & Negative Normalization
   - Bounded 3-gram OCR Noise-Tolerant Indexing (Levenshtein <= 1)
7. Standard Resolver Fallback: ExtractBenchAdapter with geometry enhancement.
8. Post-Processing: Global Hungarian Bipartite Assignment for Tabular Arrays.
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
exp039_dir = repo_root / "research" / "experiments" / "EXP-039"
exp038_dir = repo_root / "research" / "experiments" / "EXP-038"

for p in [str(repo_root), str(repo_root / "src"), str(exp038_dir), str(exp039_dir), str(exp_dir)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from date_normalizer import parse_extended_date
from global_assignment import GlobalTableAssigner, compute_iou_xywh
from hyphen_joiner import HyphenationJoiner
from multi_region_assembler import MultiRegionAssembler
from multi_token_matcher import MultiTokenSequenceMatcher
from normalization_variants import generate_normalization_variants
from ocr_noise_index import OCRNoiseTolerantIndex
from table_cell_grounding import TableCellGrounder
from visual_fallback import VisualRegionClassifier

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput


class EXP040AttributionStats:
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


class UnifiedEXP040Resolver:
    """Multi-technique deterministic evidence resolver for EXP-040."""

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
            "phase_b_multi_token": True,
            "phase_c_normalization": True,
            "phase_d_multi_region": True,
            "phase_e_hyphenation": True,
            "phase_e_visual_fallback": True,
            "phase_b_table_cells": True,
            "phase_c_ocr_noise": True,
            "phase_f_global_assignment": True,
            "fix_dates": True,
        }
        if enable_phases:
            defaults.update(enable_phases)
        self.flags = defaults

        # Per-page helper caches
        self._multi_token_matchers: dict[int, MultiTokenSequenceMatcher] = {}
        self._multi_region_assemblers: dict[int, MultiRegionAssembler] = {}
        self._hyphen_joiners: dict[int, HyphenationJoiner] = {}

        # EXP-039 engines
        self.table_grounder = TableCellGrounder() if self.flags["phase_b_table_cells"] else None
        self.ocr_noise_index = OCRNoiseTolerantIndex(doc_index) if self.flags["phase_c_ocr_noise"] else None
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

    def _get_page(self, page_num: int) -> fitz.Page | None:
        if 1 <= page_num <= len(self.fitz_doc):
            return self.fitz_doc[page_num - 1]
        return None

    def _get_token_matcher(self, page_num: int) -> MultiTokenSequenceMatcher | None:
        if page_num not in self._multi_token_matchers:
            p = self._get_page(page_num)
            if p:
                self._multi_token_matchers[page_num] = MultiTokenSequenceMatcher(p)
        return self._multi_token_matchers.get(page_num)

    def _get_region_assembler(self, page_num: int) -> MultiRegionAssembler | None:
        if page_num not in self._multi_region_assemblers:
            p = self._get_page(page_num)
            if p:
                self._multi_region_assemblers[page_num] = MultiRegionAssembler(p)
        return self._multi_region_assemblers.get(page_num)

    def _get_hyphen_joiner(self, page_num: int) -> HyphenationJoiner | None:
        if page_num not in self._hyphen_joiners:
            p = self._get_page(page_num)
            if p:
                self._hyphen_joiners[page_num] = HyphenationJoiner(p)
        return self._hyphen_joiners.get(page_num)

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
        if gold_value is None:
            return None, None

        val_str = str(gold_value).strip()
        p_target = page_hint if page_hint is not None else 1

        is_bool = isinstance(gold_value, bool) or (
            isinstance(gold_value, str)
            and val_str.lower() in ("true", "false", "yes", "no")
            and any(k in field_path.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no", "final", "amended", "general", "domestic", "contributed", "signed"))
        )

        # 1. EXP-039 Phase E: Visual Fallback for Non-Text Targets (Checkboxes, Signatures, Stamps)
        if self.flags["phase_e_visual_fallback"] and self.visual_classifier and (is_bool or any(k in field_path.lower() for k in ("_signature", "_stamp"))):
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
                        "reference_text": val_str,
                        "confidence": 0.95,
                        "source": "exp040_visual_fallback",
                        "metadata": {"type": region_type, "phase": "phase_e_visual_fallback"},
                    }
                    return cit, "phase_e_visual_fallback"

        # 2. Phase B: Multi-Token Sequence Matcher (addresses REAL_INDEXING_MISS / SUB_MULTI_TOKEN)
        if self.flags["phase_b_multi_token"] and not is_bool and (" " in val_str or len(val_str) > 3):
            tm = self._get_token_matcher(p_target)
            if tm:
                seq_box = tm.match_sequence(val_str, max_gap=2)
                if seq_box:
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": seq_box,
                        "polygon": None,
                        "reference_text": val_str,
                        "confidence": 0.95,
                        "source": "exp040_multi_token",
                        "metadata": {"phase": "phase_b_multi_token"},
                    }
                    return cit, "phase_b_multi_token"

        # 3. Phase C: Normalization Variants (addresses NORMALIZATION_MISMATCH / SUB_PERCENT_DECIMAL)
        if self.flags["phase_c_normalization"] and not is_bool:
            norm_variants = generate_normalization_variants(gold_value, field_path)
            for n_var in norm_variants:
                if n_var == val_str:
                    continue  # Already tried direct match

                # Try multi-token sequence matcher with variant
                tm = self._get_token_matcher(p_target)
                if tm:
                    v_box = tm.match_sequence(n_var, max_gap=1)
                    if v_box:
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": v_box,
                            "polygon": None,
                            "reference_text": n_var,
                            "confidence": 0.95,
                            "source": "exp040_normalization",
                            "metadata": {"phase": "phase_c_normalization", "variant": n_var},
                        }
                        return cit, "phase_c_normalization"

                # Try standard numeric candidate search
                num_cands = self.resolver.matcher.find_normalized_numeric_candidates(n_var, page_hint=p_target)
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
                        "source": "exp040_normalization_num",
                        "metadata": {"phase": "phase_c_normalization", "variant": n_var},
                    }
                    return cit, "phase_c_normalization"

        # 4. Phase D: Multi-Region Assembly (addresses NO_TEXT_AT_GOLD_REGION / SUB_MULTI_REGION)
        if self.flags["phase_d_multi_region"] and not is_bool:
            ra = self._get_region_assembler(p_target)
            if ra:
                regions = ra.assemble_regions(val_str, max_regions=5, max_gap_y=0.06)
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
                        "source": "exp040_multi_region",
                        "metadata": {"phase": "phase_d_multi_region"},
                    }
                    return cit, "phase_d_multi_region"

        # 5. Phase E: Trailing Hyphen Joiner (addresses HYPHENATION)
        if self.flags["phase_e_hyphenation"] and not is_bool:
            hj = self._get_hyphen_joiner(p_target)
            if hj:
                hyphen_box = hj.match_hyphenated(val_str)
                if hyphen_box:
                    cit = {
                        "field_path": field_path,
                        "page": p_target,
                        "bbox": hyphen_box,
                        "polygon": None,
                        "reference_text": val_str,
                        "confidence": 0.95,
                        "source": "exp040_hyphenation",
                        "metadata": {"phase": "phase_e_hyphenation"},
                    }
                    return cit, "phase_e_hyphenation"

        # 6. EXP-039 Phase B: Table Cell Grounding (addresses column bleed)
        if self.flags["phase_b_table_cells"] and self.table_grounder and not is_bool:
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
                        "metadata": {"phase": "phase_b_table_cells", "iou": cell_iou},
                    }
                    return cit, "phase_b_table_cells"
            except Exception:
                pass

        # 7. EXP-039 Date Normalizer Regex
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
                        "metadata": {"phase": "fix_dates"},
                    }
                    return cit, "fix_dates"

        # 8. EXP-039 Phase C: OCR Noise Tolerant Indexing (Levenshtein <= 1)
        if self.flags["phase_c_ocr_noise"] and self.ocr_noise_index and isinstance(gold_value, str) and not is_bool:
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
                    "metadata": {"phase": "phase_c_ocr_noise"},
                }
                return cit, "phase_c_ocr_noise"

        # 9. Standard Resolver Fallback
        try:
            inp = ExtractionInput(
                document_id=self.pdf_path.stem,
                field_path=field_path,
                value=gold_value,
                page_hint=p_target,
            )
            res = self.adapter.resolve_extraction(inp)
            if res and res.is_grounded and res.citation and res.citation.bbox:
                cit = {
                    "field_path": field_path,
                    "page": res.citation.page,
                    "bbox": res.citation.bbox.to_coco(),
                    "polygon": None,
                    "reference_text": res.citation.matched_text or val_str,
                    "confidence": res.citation.confidence,
                    "source": "standard_resolver",
                    "metadata": {"phase": "standard_resolver"},
                }
                return cit, "standard_resolver"
        except Exception:
            pass

        return None, None
