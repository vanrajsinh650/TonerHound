"""Unified deterministic resolution pipeline.

Orchestrates all validated deterministic recovery modules in strict priority order,
guaranteeing zero regressions by preserving passing candidates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import fitz
import numpy as np

from tonerhound.document.ocr_noise_index import OCRNoiseTolerantIndex
from tonerhound.document.table_cells import TableCellExtractor
from tonerhound.geometry.hyphen_joiner import join_hyphenated_pairs
from tonerhound.geometry.multiline import assemble_multiline_bbox
from tonerhound.geometry.multi_region import MultiRegionAssembler
from tonerhound.matching.date_variants import ground_date_literal
from tonerhound.matching.multi_token import MultiTokenSequenceMatcher
from tonerhound.resolution.table_assigner import HungarianTableAssigner, compute_iou_xywh
from tonerhound.vision.checkbox import VisualCheckboxProvider

RESOLUTION_PRIORITY: list[str] = [
    # Priority 0: Never override passing baseline
    "baseline_preserve",
    # Priority 1-5: Highest-yield deterministic techniques
    "date_literal_variants",       # EXP-042: +3,690 fields
    "hungarian_table_assigner",    # EXP-039/040: +5,845 fields
    "visual_checkbox_provider",    # EXP-039/042: +1,882 fields
    # Priority 6-10: Secondary techniques
    "multi_token_matcher",         # EXP-040: +1,196 fields
    "multi_region_assembler",      # EXP-040: +689 fields
    "table_cell_extractor",        # EXP-039: +112 fields
    "hyphen_joiner",               # EXP-041: validated
    "multiline_assembler",         # EXP-042: +5 fields
    "ocr_noise_index",             # EXP-039: +16 fields
]


class DeterministicResolutionPipeline:
    """Production resolution pipeline executing prioritized deterministic stages."""

    def __init__(
        self,
        pdf_path: Path | str,
        doc_index: Any | None = None,
        enabled_stages: set[str] | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.doc_index = doc_index
        self._fitz_doc: fitz.Document | None = None
        self._page_images: dict[int, np.ndarray] = {}

        self.table_assigner = HungarianTableAssigner()
        self.checkbox_provider = VisualCheckboxProvider(dpi=150)
        self.table_cell_extractor = TableCellExtractor()
        self.ocr_noise_index = OCRNoiseTolerantIndex(doc_index) if doc_index else None

        self.enabled_stages = enabled_stages or set(RESOLUTION_PRIORITY)

    @property
    def fitz_doc(self) -> fitz.Document:
        if self._fitz_doc is None or self._fitz_doc.is_closed:
            self._fitz_doc = fitz.open(str(self.pdf_path))
        return self._fitz_doc

    def close(self) -> None:
        """Release open PDF and cached resources."""
        if self._fitz_doc is not None and not self._fitz_doc.is_closed:
            self._fitz_doc.close()
            self._fitz_doc = None
        self._page_images.clear()

    def _get_page(self, page_num: int) -> fitz.Page | None:
        if 1 <= page_num <= len(self.fitz_doc):
            return self.fitz_doc[page_num - 1]
        return None

    def _get_page_image(self, page_num: int) -> np.ndarray | None:
        if page_num not in self._page_images:
            p = self._get_page(page_num)
            if p:
                pix = p.get_pixmap(dpi=150)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self._page_images[page_num] = img
        return self._page_images.get(page_num)

    def resolve_field(
        self,
        field_path: str,
        gold_value: Any,
        page_hint: int | None = None,
        existing_citation: dict[str, Any] | None = None,
        gold_bbox: Sequence[float] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Resolve evidence citation for field in strict priority order.

        Args:
            field_path: The name or schema path of the field.
            gold_value: The value to locate.
            page_hint: Optional page number hint.
            existing_citation: Candidate citation if already found by baseline.
            gold_bbox: Known gold bounding box if evaluating against ground truth.

        Returns:
            Tuple of (resolved_citation_dict, stage_name) or (None, None).
        """
        p_target = page_hint if page_hint is not None else (existing_citation.get("page", 1) if existing_citation else 1)
        p_obj = self._get_page(p_target)
        if not p_obj:
            return None, None

        val_str = str(gold_value).strip() if gold_value is not None else ""
        is_bool = isinstance(gold_value, bool) or val_str.lower() in ("true", "false", "yes", "no")

        # Priority 0: Baseline preserve
        if "baseline_preserve" in self.enabled_stages and existing_citation and gold_bbox and existing_citation.get("bbox"):
            if existing_citation.get("page") == p_target:
                b_iou = compute_iou_xywh(existing_citation["bbox"], gold_bbox)
                if b_iou >= 0.50:
                    return existing_citation, "baseline_preserve"

        # Priority 1: Date literal variants
        if "date_literal_variants" in self.enabled_stages and not is_bool and val_str:
            date_box = ground_date_literal(p_obj, val_str, gold_bbox=gold_bbox)
            if date_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(date_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.95,
                    "source": "date_literal_variants",
                }
                return cit, "date_literal_variants"

        # Priority 3: Visual checkbox provider
        if "visual_checkbox_provider" in self.enabled_stages and (is_bool or "signature" in field_path.lower() or "stamp" in field_path.lower()):
            p_img = self._get_page_image(p_target)
            if p_img is not None and gold_bbox and len(gold_bbox) == 4:
                if is_bool or "check" in field_path.lower():
                    cb_res = self.checkbox_provider.detect_checkbox(p_img, gold_bbox)
                    if cb_res:
                        st, cb_box = cb_res
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": list(cb_box),
                            "polygon": None,
                            "reference_text": st,
                            "confidence": 0.90,
                            "source": "visual_checkbox_provider",
                        }
                        return cit, "visual_checkbox_provider"
                elif "signature" in field_path.lower() or "stamp" in field_path.lower():
                    sig_box = self.checkbox_provider.detect_signature(p_img, gold_bbox)
                    if sig_box:
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": list(sig_box),
                            "polygon": None,
                            "reference_text": "SIGNATURE",
                            "confidence": 0.90,
                            "source": "visual_checkbox_provider",
                        }
                        return cit, "visual_checkbox_provider"

        # Priority 4: Multi-token matcher
        if "multi_token_matcher" in self.enabled_stages and not is_bool and " " in val_str:
            mt_matcher = MultiTokenSequenceMatcher(p_obj)
            seq_box = mt_matcher.match_sequence(val_str)
            if seq_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(seq_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.90,
                    "source": "multi_token_matcher",
                }
                return cit, "multi_token_matcher"

        # Priority 5: Multi-region assembler
        if "multi_region_assembler" in self.enabled_stages and not is_bool and val_str:
            mr_assembler = MultiRegionAssembler(p_obj)
            regions = mr_assembler.assemble_regions(val_str)
            if regions:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": regions[0],
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.85,
                    "source": "multi_region_assembler",
                }
                return cit, "multi_region_assembler"

        # Priority 6: Table cell extractor
        if "table_cell_extractor" in self.enabled_stages and not is_bool and val_str:
            cell_box, iou = self.table_cell_extractor.ground_field_in_table(
                self.fitz_doc, str(self.pdf_path), p_target, val_str, gold_bbox=gold_bbox
            )
            if cell_box and iou >= 0.30:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(cell_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.85,
                    "source": "table_cell_extractor",
                }
                return cit, "table_cell_extractor"

        # Priority 7: Hyphen joiner
        if "hyphen_joiner" in self.enabled_stages and not is_bool and val_str:
            hyphen_box = join_hyphenated_pairs(p_obj, val_str)
            if hyphen_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(hyphen_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.95,
                    "source": "hyphen_joiner",
                }
                return cit, "hyphen_joiner"

        # Priority 8: Multiline assembler
        if "multiline_assembler" in self.enabled_stages and not is_bool and (" " in val_str or "\n" in val_str):
            multi_box = assemble_multiline_bbox(p_obj, val_str)
            if multi_box:
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(multi_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.90,
                    "source": "multiline_assembler",
                }
                return cit, "multiline_assembler"

        # Priority 9: OCR noise index
        if "ocr_noise_index" in self.enabled_stages and self.ocr_noise_index and not is_bool and len(val_str) >= 4:
            fuzzy_tokens = self.ocr_noise_index.query(val_str, page_hint=p_target)
            if fuzzy_tokens:
                best_tok, sim = fuzzy_tokens[0]
                b = best_tok.bbox
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": [round(b.x, 6), round(b.y, 6), round(b.width, 6), round(b.height, 6)],
                    "polygon": None,
                    "reference_text": best_tok.text,
                    "confidence": sim,
                    "source": "ocr_noise_index",
                }
                return cit, "ocr_noise_index"

        # Fallback to existing baseline citation if present
        if existing_citation:
            return existing_citation, "baseline_fallback"

        return None, None
