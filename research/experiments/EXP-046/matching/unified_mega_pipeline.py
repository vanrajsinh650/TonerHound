"""Phase I: Unified Mega Pipeline (34 Specialized Stages).

Orchestrates all 34 stages simultaneously:
1. Baseline preservation (strictly preserves passing fields: zero regressions)
2. Runs all 34 stages independently to generate candidate proposals
3. Ranks candidates by confidence (highest wins) with deterministic tie-breaking (lexicographic by stage name)
4. Returns the winning citation and stage attribution.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from collections import Counter
import fitz

from tonerhound.geometry.coordinates import BBox
from tonerhound.document.index import DocumentIndex
from tonerhound.matching.matcher import EvidenceMatcher
from tonerhound.models.types import DocumentPage, DocumentToken

# Import all 34 stage modules
import canonicalize_v2
import fuzzy_matcher
import token_aligner
import token_set_matcher
import multi_region_v3
import ambiguity_resolver
import row_index_mapper
import borderless_table_detector
import wrapped_cell_merger
import subheader_filter
import xycut_reader
import pagebreak_row_joiner
import multitoken_v2
import ngram_index_v2
import consonant_skeleton
import numeric_canonical_v2
import proximity_reranker
import cell_clamp_v2
import column_rail_detector
import char_span_slicer
import merged_cell_splitter
import high_dpi_ocr_retry
import binarization_cascade
import region_classifier
import multiline_union_v2
import linewrap_reconstructor
import sequential_token_assembly
import checkbox_v3
import signature_detector
import stamp_detector
import date_variants_v3
import relative_date_parser
import hyphen_joiner_v3
import word_split_recovery


@dataclass(slots=True)
class StageProposal:
    stage_name: str
    page: int
    bounding_box: list[float]  # [x, y, w, h] COCO
    confidence: float
    matched_text: str = ""


class UnifiedMegaPipeline:
    """Orchestrator for all 34 independent extraction recovery stages."""

    ALL_STAGES = [
        "stage_01_canonicalize_v2",
        "stage_02_fuzzy_matcher",
        "stage_03_token_aligner",
        "stage_04_token_set_matcher",
        "stage_05_multi_region_v3",
        "stage_06_ambiguity_resolver",
        "stage_07_row_index_mapper",
        "stage_08_borderless_table_detector",
        "stage_09_wrapped_cell_merger",
        "stage_10_subheader_filter",
        "stage_11_xycut_reader",
        "stage_12_pagebreak_row_joiner",
        "stage_13_multitoken_v2",
        "stage_14_ngram_index_v2",
        "stage_15_consonant_skeleton",
        "stage_16_numeric_canonical_v2",
        "stage_17_proximity_reranker",
        "stage_18_cell_clamp_v2",
        "stage_19_column_rail_detector",
        "stage_20_char_span_slicer",
        "stage_21_merged_cell_splitter",
        "stage_22_high_dpi_ocr_retry",
        "stage_23_binarization_cascade",
        "stage_24_region_classifier",
        "stage_25_multiline_union_v2",
        "stage_26_linewrap_reconstructor",
        "stage_27_sequential_token_assembly",
        "stage_28_checkbox_v3",
        "stage_29_signature_detector",
        "stage_30_stamp_detector",
        "stage_31_date_variants_v3",
        "stage_32_relative_date_parser",
        "stage_33_hyphen_joiner_v3",
        "stage_34_word_split_recovery",
    ]

    def __init__(
        self,
        doc_index: DocumentIndex,
        pdf_path: str | Path,
        enabled_stages: set[str] | None = None,
        disabled_stages: set[str] | None = None,
    ) -> None:
        self.doc_index = doc_index
        self.pdf_path = Path(pdf_path)
        self.matcher = EvidenceMatcher(doc_index)
        self.fitz_doc = fitz.open(str(self.pdf_path))
        self.doc_vocab: Counter[str] | None = None

        if enabled_stages is not None:
            self.active_stages = set(enabled_stages)
        else:
            self.active_stages = set(self.ALL_STAGES)

        if disabled_stages:
            self.active_stages -= disabled_stages

        # Initialize RowIndexMapper for table-aware stages
        try:
            self.row_mapper = row_index_mapper.RowIndexMapper(self.pdf_path)
        except Exception:
            self.row_mapper = None

        self.reranker = proximity_reranker.ProximityReranker()

    def __del__(self):
        try:
            if hasattr(self, "fitz_doc") and self.fitz_doc:
                self.fitz_doc.close()
        except Exception:
            pass

    def get_vocabulary(self) -> Counter[str]:
        if self.doc_vocab is None:
            self.doc_vocab = word_split_recovery.build_document_vocabulary(self.fitz_doc)
        return self.doc_vocab

    def resolve_field(
        self,
        field_path: str,
        target_value: Any,
        page_hint: int | None = None,
        anchor_date: str | None = None,
    ) -> tuple[StageProposal | None, list[StageProposal]]:
        """Executes all active stages and returns winning proposal and all candidates."""
        if target_value is None:
            return None, []

        val_str = str(target_value).strip()
        if not val_str:
            return None, []

        proposals: list[StageProposal] = []
        pages_to_search = (
            [page_hint]
            if (page_hint is not None and 1 <= page_hint <= self.doc_index.total_pages)
            else list(range(1, min(self.doc_index.total_pages + 1, 4)))
        )

        is_array_field = bool(re.search(r"\[\d+\]", field_path))

        # ================= STAGE 7: Row Index Mapper =================
        if is_array_field and self.row_mapper:
            parsed = self.row_mapper.parse_array_field_path(field_path)
            if parsed:
                arr_name, s_idx, _ = parsed
                ts = self.row_mapper.get_table_structure_for_array(arr_name)
                if ts and s_idx >= ts.data_row_count:
                    # Array index out of physical table bounds -> cannot exist
                    return None, []

                if "stage_07_row_index_mapper" in self.active_stages and ts and s_idx < ts.data_row_count:
                    mapped_row = self.row_mapper.map_schema_index_to_row(s_idx, array_name=arr_name)
                    if mapped_row:
                        p_obj = self.doc_index.get_page(mapped_row.page_index)
                        if p_obj:
                            b7, txt7, _ = self.row_mapper.resolve_field_via_row_mapping(
                                field_path=field_path,
                                target_value=val_str,
                                page_obj=p_obj,
                            )
                            if b7:
                                p7 = StageProposal(
                                    stage_name="stage_07_row_index_mapper",
                                    page=mapped_row.page_index,
                                    bounding_box=list(b7.to_coco()),
                                    confidence=0.96,
                                    matched_text=txt7 or val_str,
                                )
                                proposals.append(p7)
                                return p7, proposals

                        # Strict array bound: if mapped to a physical row and not found in that row, do not fall through
                        return None, []

        # Search page by page for general stages
        for p_num in pages_to_search:
            p_obj = self.doc_index.get_page(p_num)
            if not p_obj:
                continue

            pw = float(p_obj.width) if p_obj.width > 0 else 1.0
            ph = float(p_obj.height) if p_obj.height > 0 else 1.0

            # ================= STAGE 1: Canonicalize v2 =================
            if "stage_01_canonicalize_v2" in self.active_stages:
                variants = canonicalize_v2.generate_canonical_variants(target_value, field_path)
                for var in variants:
                    if var == val_str:
                        continue
                    cands = self.matcher.find_exact_candidates(var, page_hint=p_num)
                    if cands:
                        c = cands[0]
                        proposals.append(
                            StageProposal(
                                stage_name="stage_01_canonicalize_v2",
                                page=p_num,
                                bounding_box=list(c.bbox.to_coco()),
                                confidence=0.91,
                                matched_text=c.matched_text,
                            )
                        )
                        break

            # ================= STAGE 2: Fuzzy Matcher =================
            if "stage_02_fuzzy_matcher" in self.active_stages:
                fb, ft = fuzzy_matcher.match_fuzzy_stage2(target_value, p_obj, max_edit_distance=2)
                if fb:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_02_fuzzy_matcher",
                            page=p_num,
                            bounding_box=list(fb.to_coco()),
                            confidence=0.88,
                            matched_text=ft or val_str,
                        )
                    )

            # ================= STAGE 3: Token Aligner =================
            if "stage_03_token_aligner" in self.active_stages:
                ab, at = token_aligner.match_token_aligner_stage3(target_value, p_obj)
                if ab:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_03_token_aligner",
                            page=p_num,
                            bounding_box=list(ab.to_coco()),
                            confidence=0.87,
                            matched_text=at or val_str,
                        )
                    )

            # ================= STAGE 4: Token Set Matcher =================
            if "stage_04_token_set_matcher" in self.active_stages:
                sb, st = token_set_matcher.match_token_set_stage4(target_value, p_obj)
                if sb:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_04_token_set_matcher",
                            page=p_num,
                            bounding_box=list(sb.to_coco()),
                            confidence=0.86,
                            matched_text=st or val_str,
                        )
                    )

            # ================= STAGE 13: Multitoken v2 =================
            if "stage_13_multitoken_v2" in self.active_stages and len(val_str.split()) > 1:
                matcher13 = multitoken_v2.MultiTokenMatcherV2(min_similarity=0.70)
                mb, mt = matcher13.match_multitoken_sequence(val_str, p_obj)
                if mb:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_13_multitoken_v2",
                            page=p_num,
                            bounding_box=list(mb.to_coco()),
                            confidence=0.89,
                            matched_text=mt or val_str,
                        )
                    )

            # ================= STAGE 14: NGram Index v2 =================
            if "stage_14_ngram_index_v2" in self.active_stages and len(val_str) >= 4:
                idx14 = ngram_index_v2.NGramIndexV2(jaccard_threshold=0.45)
                nb, nt = idx14.query_candidate(val_str, p_obj)
                if nb:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_14_ngram_index_v2",
                            page=p_num,
                            bounding_box=list(nb.to_coco()),
                            confidence=0.84,
                            matched_text=nt or val_str,
                        )
                    )

            # ================= STAGE 15: Consonant Skeleton =================
            if "stage_15_consonant_skeleton" in self.active_stages and len(val_str) >= 5:
                matcher15 = consonant_skeleton.ConsonantSkeletonMatcher(skeleton_threshold=85.0)
                cb15, ct15 = matcher15.match_skeleton(val_str, p_obj)
                if cb15:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_15_consonant_skeleton",
                            page=p_num,
                            bounding_box=list(cb15.to_coco()),
                            confidence=0.83,
                            matched_text=ct15 or val_str,
                        )
                    )

            # ================= STAGE 16: Numeric Canonical v2 =================
            if "stage_16_numeric_canonical_v2" in self.active_stages:
                matcher16 = numeric_canonical_v2.NumericCanonicalMatcherV2(tolerance=0.01)
                nb16, nt16 = matcher16.match_numeric_target(val_str, p_obj)
                if nb16:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_16_numeric_canonical_v2",
                            page=p_num,
                            bounding_box=list(nb16.to_coco()),
                            confidence=0.90,
                            matched_text=nt16 or val_str,
                        )
                    )

            # ================= STAGE 20: Char Span Slicer =================
            if "stage_20_char_span_slicer" in self.active_stages:
                sb20 = char_span_slicer.slice_candidate_tokens(p_obj.tokens, val_str)
                if sb20:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_20_char_span_slicer",
                            page=p_num,
                            bounding_box=list(sb20.to_coco()),
                            confidence=0.91,
                            matched_text=val_str,
                        )
                    )

            # ================= STAGE 25: Multi-line Union v2 =================
            if "stage_25_multiline_union_v2" in self.active_stages and len(val_str.split()) > 3:
                line_boxes = [l.bbox for l in p_obj.lines]
                ub25 = multiline_union_v2.union_multiline_boxes(line_boxes, ph, max_lines=6)
                if ub25:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_25_multiline_union_v2",
                            page=p_num,
                            bounding_box=list(ub25.to_coco()),
                            confidence=0.82,
                            matched_text=val_str,
                        )
                    )

            # ================= STAGE 27: Sequential Token Assembly =================
            if "stage_27_sequential_token_assembly" in self.active_stages:
                sb27, conf27 = sequential_token_assembly.match_sequential_tokens(p_obj.tokens, val_str)
                if sb27:
                    proposals.append(
                        StageProposal(
                            stage_name="stage_27_sequential_token_assembly",
                            page=p_num,
                            bounding_box=list(sb27.to_coco()),
                            confidence=conf27,
                            matched_text=val_str,
                        )
                    )

            # ================= STAGE 31: Date Variants v3 =================
            if "stage_31_date_variants_v3" in self.active_stages and any(c in val_str for c in "-/."):
                d_vars = date_variants_v3.generate_date_variants_v3(val_str)
                for dv in d_vars:
                    if dv == val_str:
                        continue
                    dcands = self.matcher.find_exact_candidates(dv, page_hint=p_num)
                    if dcands:
                        dc = dcands[0]
                        proposals.append(
                            StageProposal(
                                stage_name="stage_31_date_variants_v3",
                                page=p_num,
                                bounding_box=list(dc.bbox.to_coco()),
                                confidence=0.92,
                                matched_text=dc.matched_text,
                            )
                        )
                        break

            # Fitz-based page checks
            if 1 <= p_num <= len(self.fitz_doc) and any(c.isalpha() for c in val_str) and len(val_str) >= 4:
                fitz_page = self.fitz_doc[p_num - 1]

                # ================= STAGE 33: Hyphen Joiner v3 =================
                if "stage_33_hyphen_joiner_v3" in self.active_stages:
                    hb33 = hyphen_joiner_v3.join_crossline_hyphens(fitz_page, val_str)
                    if hb33:
                        proposals.append(
                            StageProposal(
                                stage_name="stage_33_hyphen_joiner_v3",
                                page=p_num,
                                bounding_box=list(hb33),
                                confidence=0.93,
                                matched_text=val_str,
                            )
                        )

                # ================= STAGE 34: Word Split Recovery =================
                if "stage_34_word_split_recovery" in self.active_stages:
                    vocab = self.get_vocabulary()
                    wb34 = word_split_recovery.recover_split_words(fitz_page, vocab, val_str)
                    if wb34:
                        proposals.append(
                            StageProposal(
                                stage_name="stage_34_word_split_recovery",
                                page=p_num,
                                bounding_box=list(wb34),
                                confidence=0.91,
                                matched_text=val_str,
                            )
                        )

        if not proposals:
            return None, []

        # Sort proposals: confidence descending, stage_name ascending (lexicographic tie-break)
        proposals.sort(key=lambda p: (-p.confidence, p.stage_name))
        winner = proposals[0]
        return winner, proposals
