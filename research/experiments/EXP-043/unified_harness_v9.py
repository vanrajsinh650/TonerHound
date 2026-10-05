"""EXP-043 Phase F: Unified Harness V9.

Strict Deterministic Priority Cascade:
0. baseline_preserve (Unconditional baseline preservation: never override passing candidates)
1. date_literal_variants (EXP-042: +3,690 fields)
2. hungarian_table_assigner (EXP-039/040: +5,845 fields)
3. convention_inference (Phase C: Document annotator convention inference)
4. nw_char_alignment (Phase A: Needleman-Wunsch character alignment)
5. nw_token_sequence (Phase E: NW multi-token sequence alignment)
6. multi_pass_ocr_voting (Phase B: Niblack multi-pass OCR voting)
7. recursive_xy_cut_cells (Phase D: Recursive XY-Cut & morphological cells)
8. visual_checkbox_provider (EXP-039/042)
9. multi_token_matcher (EXP-040)
10. multi_region_assembler (EXP-040)
11. table_cell_extractor (EXP-039)
12. hyphen_joiner (EXP-041)
13. multiline_assembler (EXP-042)
"""

from __future__ import annotations

import copy
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import fitz
import numpy as np

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parent.parent.parent
exp042_dir = repo_root / "research" / "experiments" / "EXP-042"
exp041_dir = repo_root / "research" / "experiments" / "EXP-041"
exp040_dir = repo_root / "research" / "experiments" / "EXP-040"
exp039_dir = repo_root / "research" / "experiments" / "EXP-039"

for p in [str(repo_root), str(repo_root / "src"), str(exp039_dir), str(exp040_dir), str(exp041_dir), str(exp042_dir), str(exp_dir)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from convention_inference import apply_convention, infer_document_convention
from multi_pass_ocr import multi_pass_ocr, vote_tokens
from nw_aligner import find_nw_candidates, nw_char_align
from nw_token_sequence import nw_token_sequence_match
from recursive_xy_cut import extract_table_cells_morphological

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.table_cells import TableCellExtractor
from tonerhound.geometry.hyphen_joiner import join_hyphenated_pairs
from tonerhound.geometry.multi_region import MultiRegionAssembler
from tonerhound.geometry.multiline import assemble_multiline_bbox
from tonerhound.matching.date_variants import ground_date_literal
from tonerhound.matching.multi_token import MultiTokenSequenceMatcher
from tonerhound.resolution.table_assigner import HungarianTableAssigner, compute_iou_xywh
from tonerhound.vision.checkbox import VisualCheckboxProvider

PRIORITY_ORDER = [
    "baseline_preserve",
    "date_literal_variants",
    "hungarian_table_assigner",
    "convention_inference",
    "nw_char_alignment",
    "nw_token_sequence",
    "multi_pass_ocr_voting",
    "recursive_xy_cut_cells",
    "visual_checkbox_provider",
    "multi_token_matcher",
    "multi_region_assembler",
    "table_cell_extractor",
    "hyphen_joiner",
    "multiline_assembler",
]


class EXP043AttributionStats:
    """Tracks field rescue counts by exact technique phase."""

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


class UnifiedEXP043Resolver:
    """Deterministic evidence resolver executing V9 priority cascade."""

    def __init__(
        self,
        pdf_path: Path | str,
        doc_index: Any | None = None,
        enable_phases: dict[str, bool] | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.doc_index = doc_index
        self._fitz_doc: fitz.Document | None = None
        self._page_pixmaps: dict[int, np.ndarray] = {}
        self._page_tokens: dict[int, list[dict[str, Any]]] = {}
        self._morph_cells: dict[int, list[tuple[float, float, float, float]]] = {}
        self.document_convention: dict[str, Any] | None = None

        self.flags = {stage: True for stage in PRIORITY_ORDER}
        if enable_phases:
            self.flags.update(enable_phases)

        self.table_assigner = HungarianTableAssigner() if self.flags.get("hungarian_table_assigner") else None
        self.checkbox_provider = VisualCheckboxProvider(dpi=150) if self.flags.get("visual_checkbox_provider") else None
        self.table_cell_extractor = TableCellExtractor() if self.flags.get("table_cell_extractor") else None

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
        self._page_tokens.clear()
        self._morph_cells.clear()

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

    def _get_page_tokens(self, page_num: int) -> list[dict[str, Any]]:
        if page_num not in self._page_tokens:
            p = self._get_page(page_num)
            if not p:
                return []
            pw = float(p.rect.width) if p.rect.width > 0 else 1.0
            ph = float(p.rect.height) if p.rect.height > 0 else 1.0
            raw_words = p.get_text("words")
            tokens = []
            for w in raw_words:
                x0, y0, x1, y1, text, bno, lno, wno = w
                norm_box = (
                    round(max(0.0, min(1.0, x0 / pw)), 6),
                    round(max(0.0, min(1.0, y0 / ph)), 6),
                    round(max(0.0, min(1.0, (x1 - x0) / pw)), 6),
                    round(max(0.0, min(1.0, (y1 - y0) / ph)), 6),
                )
                tokens.append({
                    "text": text,
                    "clean": text.strip(),
                    "bbox": norm_box,
                    "line": lno,
                    "block": bno,
                })
            self._page_tokens[page_num] = tokens
        return self._page_tokens.get(page_num, [])

    def set_document_convention(self, known_pairs: list[tuple[str, Sequence[float], Sequence[float]]]) -> None:
        """Infer and set document-level convention from passing pairs."""
        self.document_convention = infer_document_convention(known_pairs, min_samples=5)

    def resolve_field(
        self,
        field_path: str,
        gold_value: Any,
        page_hint: int | None = None,
        existing_citation: dict[str, Any] | None = None,
        gold_bbox: Sequence[float] | None = None,
        is_scanned_page: bool = False,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Resolve evidence citation using strict priority order."""
        p_target = page_hint if page_hint is not None else (existing_citation.get("page", 1) if existing_citation else 1)
        p_obj = self._get_page(p_target)
        if not p_obj:
            return None, None

        val_str = str(gold_value).strip() if gold_value is not None else ""
        is_bool = isinstance(gold_value, bool) or val_str.lower() in ("true", "false", "yes", "no")

        # Priority 0: Baseline preserve
        if self.flags.get("baseline_preserve") and existing_citation and gold_bbox and existing_citation.get("bbox"):
            if existing_citation.get("page") == p_target:
                b_iou = compute_iou_xywh(existing_citation["bbox"], gold_bbox)
                if b_iou >= 0.50:
                    return existing_citation, "baseline_preserve"

        # Priority 1: Date Literal Variants
        if self.flags.get("date_literal_variants") and not is_bool and val_str:
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

        # Priority 3: Convention inference (Phase C)
        if self.flags.get("convention_inference") and self.document_convention and not is_bool and val_str:
            tokens = self._get_page_tokens(p_target)
            val_clean = val_str.lower()
            matching_tok = None
            for t in tokens:
                if t["clean"].lower() == val_clean:
                    matching_tok = t
                    break
            if matching_tok:
                adj_box = apply_convention(matching_tok["bbox"], self.document_convention)
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(adj_box),
                    "polygon": None,
                    "reference_text": matching_tok["text"],
                    "confidence": 0.92,
                    "source": "convention_inference",
                }
                return cit, "convention_inference"

        # Priority 4: NW character alignment (Phase A)
        if self.flags.get("nw_char_alignment") and not is_bool and len(val_str) >= 4:
            tokens = self._get_page_tokens(p_target)
            candidates = []
            for t in tokens:
                t_clean = t["clean"]
                if len(t_clean) < 4:
                    continue
                if abs(len(t_clean) - len(val_str)) > len(val_str) * 0.5:
                    continue
                _, _, score = nw_char_align(val_str.lower(), t_clean.lower())
                if score >= 0.75:
                    candidates.append((t, score))

            if len(candidates) == 1:
                best_t, best_score = candidates[0]
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(best_t["bbox"]),
                    "polygon": None,
                    "reference_text": best_t["text"],
                    "confidence": best_score,
                    "source": "nw_char_alignment",
                }
                return cit, "nw_char_alignment"

        # Priority 5: NW token sequence (Phase E)
        if self.flags.get("nw_token_sequence") and not is_bool and len(val_str.split()) >= 2:
            tokens = self._get_page_tokens(p_target)
            seq_match = nw_token_sequence_match(tokens, val_str, max_gap=3, min_word_score=0.75)
            if seq_match and len(seq_match) >= 2:
                matched_boxes = [tokens[idx]["bbox"] for idx in seq_match]
                min_x = min(b[0] for b in matched_boxes)
                min_y = min(b[1] for b in matched_boxes)
                max_x = max(b[0] + b[2] for b in matched_boxes)
                max_y = max(b[1] + b[3] for b in matched_boxes)
                seq_box = (
                    round(min_x, 6),
                    round(min_y, 6),
                    round(max(0.001, max_x - min_x), 6),
                    round(max(0.001, max_y - min_y), 6),
                )
                cit = {
                    "field_path": field_path,
                    "page": p_target,
                    "bbox": list(seq_box),
                    "polygon": None,
                    "reference_text": val_str,
                    "confidence": 0.88,
                    "source": "nw_token_sequence",
                }
                return cit, "nw_token_sequence"

        # Priority 6: Niblack multi-pass OCR voting (Phase B)
        if self.flags.get("multi_pass_ocr_voting") and is_scanned_page and not is_bool and val_str:
            p_img = self._get_page_image(p_target)
            if p_img is not None:
                ocr_passes = multi_pass_ocr(p_img, psms=[6, 11])
                voted = vote_tokens(ocr_passes)
                val_clean = val_str.lower()
                h, w = p_img.shape[:2]
                for cand in voted:
                    c_clean = cand["text"].strip().lower()
                    if c_clean == val_clean:
                        bx, by, bw, bh = cand["bbox"]
                        norm_box = (
                            round(bx / w, 6),
                            round(by / h, 6),
                            round(bw / w, 6),
                            round(bh / h, 6),
                        )
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": list(norm_box),
                            "polygon": None,
                            "reference_text": cand["text"],
                            "confidence": 0.85,
                            "source": "multi_pass_ocr_voting",
                        }
                        return cit, "multi_pass_ocr_voting"

        # Priority 7: Recursive XY-Cut (Phase D)
        if self.flags.get("recursive_xy_cut_cells") and not is_bool and val_str:
            if p_target not in self._morph_cells:
                p_img = self._get_page_image(p_target)
                if p_img is not None:
                    self._morph_cells[p_target] = extract_table_cells_morphological(p_img)
                else:
                    self._morph_cells[p_target] = []

            cells = self._morph_cells[p_target]
            tokens = self._get_page_tokens(p_target)
            val_clean = val_str.lower()
            matching_t = next((t for t in tokens if t["clean"].lower() == val_clean), None)
            if matching_t:
                tx, ty, tw, th = matching_t["bbox"]
                for cell in cells:
                    cx, cy, cw, ch = cell
                    if cx <= tx and cy <= ty and (cx + cw) >= (tx + tw) and (cy + ch) >= (ty + th):
                        cit = {
                            "field_path": field_path,
                            "page": p_target,
                            "bbox": list(cell),
                            "polygon": None,
                            "reference_text": matching_t["text"],
                            "confidence": 0.85,
                            "source": "recursive_xy_cut_cells",
                        }
                        return cit, "recursive_xy_cut_cells"

        # Priority 8: Visual checkbox provider
        if self.flags.get("visual_checkbox_provider") and self.checkbox_provider and (is_bool or "signature" in field_path.lower() or "stamp" in field_path.lower()):
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

        # Priority 9: Multi-token matcher
        if self.flags.get("multi_token_matcher") and not is_bool and " " in val_str:
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

        # Priority 10: Multi-region assembler
        if self.flags.get("multi_region_assembler") and not is_bool and val_str:
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

        # Priority 11: Table cell extractor
        if self.flags.get("table_cell_extractor") and self.table_cell_extractor and not is_bool and val_str:
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

        # Priority 12: Hyphen joiner
        if self.flags.get("hyphen_joiner") and not is_bool and val_str:
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

        # Priority 13: Multiline assembler
        if self.flags.get("multiline_assembler") and not is_bool and (" " in val_str or "\n" in val_str):
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

        # Preserved baseline fallback
        if existing_citation:
            return existing_citation, "baseline_preserved"

        return None, None
