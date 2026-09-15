"""Form-aware structural grounding engine for IRS Form 1040 tax returns and subsidiary schedules.

Provides layout-aware, line-anchored, and spatially constrained evidence resolution
for individual tax return packages containing Form 1040, Schedule 1/2/3, Schedules A-SE,
and supporting IRS tax forms.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from rapidfuzz import fuzz

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import ExtractionInput
from tonerhound.normalization.normalizers import (
    is_number_equal,
    parse_numeric_value,
)
from tonerhound.resolution.resolver import EvidenceResolver

# Signature fields uniquely identifying IRS Form 1040 tax return packages
TAX_1040_SIGNATURES = frozenset({
    "taxpayer_first_name_mi",
    "taxpayer_last_name",
    "taxpayer_ssn",
    "spouse_first_name_mi",
    "spouse_last_name",
    "filing_status",
    "line_1a_total_w2_wages",
    "line_1z_total_wages",
    "line_9_total_income",
    "line_11_adjusted_gross_income",
    "line_15_taxable_income",
    "line_16_tax",
    "line_24_total_tax",
    "line_33_total_payments",
    "third_party_designee_yes",
    "preparer_self_employed_box",
})


def is_form_1040_tax_return(
    leaves: Sequence[tuple[str, Any, int | None, str | None, str]],
) -> bool:
    """Check if extraction payload matches IRS Form 1040 schema signature."""
    root_fields = {leaf[0] for leaf in leaves if leaf[4] in ("", "root")}
    matching = root_fields & TAX_1040_SIGNATURES
    return len(matching) >= 3


# Standard Form 1040 line positions: (page_offset_from_1040_p1, expected_y, (x_min, x_max))
# page_offset 0 = Form 1040 Page 1; page_offset 1 = Form 1040 Page 2
FORM_1040_LINES: dict[str, tuple[int, float, tuple[float, float]]] = {
    "1a": (0, 0.50, (0.75, 0.98)),
    "1b": (0, 0.52, (0.75, 0.98)),
    "1c": (0, 0.54, (0.75, 0.98)),
    "1d": (0, 0.56, (0.75, 0.98)),
    "1e": (0, 0.58, (0.75, 0.98)),
    "1f": (0, 0.60, (0.75, 0.98)),
    "1g": (0, 0.61, (0.75, 0.98)),
    "1h": (0, 0.62, (0.75, 0.98)),
    "1i": (0, 0.63, (0.75, 0.98)),
    "1z": (0, 0.64, (0.75, 0.98)),
    "2a": (0, 0.65, (0.35, 0.60)),
    "2b": (0, 0.65, (0.75, 0.98)),
    "3a": (0, 0.68, (0.35, 0.60)),
    "3b": (0, 0.68, (0.75, 0.98)),
    "4a": (0, 0.70, (0.35, 0.60)),
    "4b": (0, 0.70, (0.75, 0.98)),
    "5a": (0, 0.72, (0.35, 0.60)),
    "5b": (0, 0.72, (0.75, 0.98)),
    "6a": (0, 0.74, (0.35, 0.60)),
    "6b": (0, 0.74, (0.75, 0.98)),
    "7": (0, 0.75, (0.75, 0.98)),
    "8": (0, 0.76, (0.75, 0.98)),
    "9": (0, 0.77, (0.75, 0.98)),
    "10": (0, 0.79, (0.75, 0.98)),
    "11": (0, 0.81, (0.75, 0.98)),
    "12": (0, 0.82, (0.75, 0.98)),
    "13": (0, 0.84, (0.75, 0.98)),
    "14": (0, 0.85, (0.75, 0.98)),
    "15": (0, 0.87, (0.75, 0.98)),
    "16": (1, 0.05, (0.80, 0.98)),
    "17": (1, 0.07, (0.80, 0.98)),
    "18": (1, 0.08, (0.80, 0.98)),
    "19": (1, 0.10, (0.80, 0.98)),
    "20": (1, 0.11, (0.80, 0.98)),
    "21": (1, 0.12, (0.80, 0.98)),
    "22": (1, 0.14, (0.80, 0.98)),
    "23": (1, 0.16, (0.80, 0.98)),
    "24": (1, 0.17, (0.80, 0.98)),
    "25a": (1, 0.20, (0.62, 0.78)),
    "25b": (1, 0.22, (0.62, 0.78)),
    "25c": (1, 0.24, (0.62, 0.78)),
    "25d": (1, 0.25, (0.80, 0.98)),
    "26": (1, 0.27, (0.80, 0.98)),
    "27": (1, 0.29, (0.80, 0.98)),
    "28": (1, 0.31, (0.80, 0.98)),
    "29": (1, 0.33, (0.80, 0.98)),
    "31": (1, 0.34, (0.80, 0.98)),
    "32": (1, 0.36, (0.80, 0.98)),
    "33": (1, 0.37, (0.80, 0.98)),
    "34": (1, 0.40, (0.80, 0.98)),
    "35a": (1, 0.41, (0.80, 0.98)),
    "36": (1, 0.45, (0.65, 0.85)),
    "37": (1, 0.48, (0.80, 0.98)),
    "38": (1, 0.50, (0.80, 0.98)),
}

# Checkbox patterns on Form 1040: (field_pattern, page_offset, target_x, x_tol, y_range, anchor_kws)
CHECKBOX_PATTERNS = [
    ("presidential_campaign_you_box", 0, 0.81, 0.05, (0.19, 0.32), ["presidential", "fund", "$3", "election"]),
    ("presidential_campaign_spouse_box", 0, 0.87, 0.05, (0.19, 0.32), ["presidential", "fund", "$3", "election"]),
    ("someone_can_claim_you_box", 0, 0.28, 0.05, (0.28, 0.41), ["someone", "claim", "dependent"]),
    ("someone_can_claim_spouse_box", 0, 0.44, 0.05, (0.28, 0.41), ["someone", "claim", "dependent"]),
    ("spouse_itemizes_box", 0, 0.15, 0.04, (0.29, 0.42), ["itemizes", "separate"]),
    ("you_born_before", 0, 0.19, 0.04, (0.32, 0.44), ["blind", "born", "age"]),
    ("you_blind_box", 0, 0.41, 0.04, (0.32, 0.44), ["blind", "born", "age"]),
    ("spouse_born_before", 0, 0.56, 0.05, (0.33, 0.44), ["blind", "born", "age"]),
    ("spouse_blind_box", 0, 0.79, 0.06, (0.33, 0.45), ["blind", "born", "age"]),
    ("more_than_four_dependents_box", 0, 0.09, 0.05, (0.41, 0.52), ["dependents", "four", "more"]),
    ("digital_assets_yes", 0, 0.88, 0.05, (0.27, 0.41), ["digital", "assets", "asset"]),
    ("line_6c_lump_sum_election_box", 0, 0.73, 0.04, (0.66, 0.78), ["6c", "lump", "sum"]),
    ("line_7_schedule_d_not_required_box", 0, 0.73, 0.04, (0.68, 0.79), ["schedule d", "not required", "gain"]),
    ("line_16_form_8814_box", 1, 0.51, 0.05, (0.03, 0.11), ["8814", "1"]),
    ("line_16_form_4972_box", 1, 0.57, 0.05, (0.03, 0.11), ["4972", "2"]),
    ("line_16_other_form_box", 1, 0.64, 0.05, (0.03, 0.11), ["other", "3"]),
    ("line_35a_form_8888_box", 1, 0.73, 0.05, (0.39, 0.51), ["8888", "attached"]),
    ("third_party_designee_yes", 1, 0.63, 0.06, (0.50, 0.64), ["designee", "yes", "another person"]),
    ("preparer_self_employed_box", 1, 0.81, 0.08, (0.67, 0.79), ["self-employed", "self", "employed"]),
]

# Form & Schedule header detection patterns for multi-schedule tax filings
FORM_SCHEDULE_PATTERNS = [
    ("schedule_1", r"schedule\s*1\b"),
    ("schedule_2", r"schedule\s*2\b"),
    ("schedule_3", r"schedule\s*3\b"),
    ("schedule_a", r"(?:schedule\s*a\b|itemized\s*deductions)"),
    ("schedule_b", r"(?:schedule\s*b\b|interest\s*and\s*ordinary\s*dividends)"),
    ("schedule_c", r"(?:schedule\s*c\b|profit\s*or\s*loss\s*from\s*business)"),
    ("schedule_d", r"(?:schedule\s*d\b|capital\s*gains\s*and\s*losses)"),
    ("schedule_e", r"(?:schedule\s*e\b|supplemental\s*income\s*and\s*loss)"),
    ("schedule_se", r"(?:schedule\s*se\b|self-employment\s*tax)"),
    ("form_8995", r"(?:form\s*8995\b|\b8995\b)"),
    ("form_8582", r"(?:form\s*8582\b|\b8582\b)"),
    ("form_7203", r"(?:form\s*7203\b|\b7203\b)"),
    ("form_1116", r"(?:form\s*1116\b|\b1116\b)"),
    ("form_6251", r"(?:form\s*6251\b|\b6251\b)"),
    ("form_8959", r"(?:form\s*8959\b|\b8959\b)"),
    ("form_8960", r"(?:form\s*8960\b|\b8960\b)"),
    ("form_8879", r"(?:form\s*8879\b|\b8879\b)"),
    ("form_2106", r"(?:form\s*2106\b|\b2106\b)"),
    ("form_8283", r"(?:form\s*8283\b|\b8283\b)"),
]

FILING_STATUS_X = {
    "Single": 0.065,
    "Married filing jointly": 0.150,
    "Married filing separately": 0.320,
    "Head of household": 0.530,
    "Qualifying surviving spouse": 0.730,
}


class TaxFormGrounder:
    """Specialized structural grounder for IRS Form 1040 and subsidiary schedules."""

    def __init__(
        self,
        index: DocumentIndex,
        resolver: EvidenceResolver,
        enable_bbox_precision: bool = True,
    ) -> None:
        self.index = index
        self.resolver = resolver
        self.enable_bbox_precision = enable_bbox_precision

    def _detect_form_1040_pages(self) -> tuple[int, int]:
        """Detect physical pages containing IRS Form 1040 Page 1 and Page 2."""
        total_p = len(self.index.pages)
        if total_p == 1:
            return 1, 1

        # Search the first 5 pages for Form 1040 header
        p1 = 1
        for p in self.index.pages[:min(5, total_p)]:
            top_txt = " ".join(t.text for t in p.tokens[:40]).lower()
            if re.search(r"1\s*0\s*4\s*0\b|individual\s*income\s*tax\s*return", top_txt):
                p1 = p.page_number
                break

        p2 = min(total_p, p1 + 1)
        return p1, p2

    def _detect_schedule_pages(self) -> dict[str, list[int]]:
        """Map every subsidiary schedule / form to its physical pages."""
        sched_map: dict[str, list[int]] = {}
        for p in self.index.pages:
            top_txt = " ".join(t.text for t in p.tokens[:40]).lower()
            for s_name, pat in FORM_SCHEDULE_PATTERNS:
                if re.search(pat, top_txt):
                    sched_map.setdefault(s_name, []).append(p.page_number)
        return sched_map

    def _resolve_checkbox(
        self,
        p_obj: Any,
        field_name: str,
        value: Any,
        page_offset: int,
    ) -> BBox | None:
        """Resolve physical bounding box for Form 1040 boolean checkbox field."""
        rule = next((r for r in CHECKBOX_PATTERNS if r[0] in field_name and r[1] == page_offset), None)
        if not rule:
            return None

        _, _, target_x, x_tol, y_range, anchor_kws = rule

        # Find line matching anchor keywords
        best_y = None
        for line in p_obj.lines:
            cy = line.bbox.y + line.bbox.height / 2.0
            if y_range[0] <= cy <= y_range[1]:
                txt = line.norm_text.lower()
                if any(kw in txt for kw in anchor_kws):
                    best_y = cy
                    break

        eff_target_x = target_x
        if "digital_assets" in field_name:
            eff_target_x = 0.83 if value is True else 0.88
        elif "third_party_designee" in field_name:
            eff_target_x = 0.63 if value is True else 0.82

        ref_y = best_y if best_y is not None else (y_range[0] + y_range[1]) / 2.0

        # Look for checkbox glyph tokens in proximity
        cand_tokens: list[tuple[float, Any]] = []
        for t in p_obj.tokens:
            cy = t.bbox.y + t.bbox.height / 2.0
            cx = t.bbox.x + t.bbox.width / 2.0
            if abs(cy - ref_y) <= 0.025 and abs(cx - eff_target_x) <= x_tol:
                is_glyph = t.text.strip() in (
                    "[]", "[", "]", "D", "|", "I", "o", "q", "lj", "1", "Bl", "x", "X", "{", "}", "{115", "[T",
                )
                score = (10.0 if is_glyph else 0.0) - abs(cx - eff_target_x) * 20.0 - abs(cy - ref_y) * 40.0
                cand_tokens.append((score, t))

        if cand_tokens:
            cand_tokens.sort(key=lambda x: x[0], reverse=True)
            chosen = cand_tokens[0][1]
            cx = chosen.bbox.x + chosen.bbox.width / 2.0
            cy = chosen.bbox.y + chosen.bbox.height / 2.0
            return BBox(cx - 0.009, cy - 0.009, 0.018, 0.018, p_obj.page_number)
        else:
            return BBox(eff_target_x - 0.009, ref_y - 0.009, 0.018, 0.018, p_obj.page_number)

    def ground(
        self,
        leaves: Sequence[tuple[str, Any, int | None, str | None, str]],
        extracted_data: dict[str, Any] | list[dict[str, Any]],
        example_id: str = "example_0",
        pipeline_name: str = "tonerhound",
    ) -> dict[str, Any]:
        """Execute layout-aware, line-anchored grounding on Form 1040 tax package."""
        p1040_p1, p1040_p2 = self._detect_form_1040_pages()
        sched_pages = self._detect_schedule_pages()

        # Find Dependents header y on Page 1
        p1_obj = self.index.get_page(p1040_p1)
        dep_header_y = None
        if p1_obj:
            for line in p1_obj.lines:
                cy = line.bbox.y + line.bbox.height / 2.0
                txt = line.norm_text.lower()
                if 0.38 <= cy <= 0.46 and any(k in txt for k in ("dependents", "child tax", "credit", "qualifies", "fistname")):
                    dep_header_y = cy
                    break

        citations: list[dict[str, Any]] = []

        for path, value, _hint, context, parent_record_path in leaves:
            if value is None:
                continue

            # Handle page-only citations (source_page, page_number, page_no, page_num)
            field_basename = path.split(".")[-1].split("[")[0]
            if field_basename in ("source_page", "page_number", "page_no", "page_num", "page") and _hint:
                citations.append({
                    "field_path": path,
                    "page": _hint,
                    "bbox": None,
                    "reference_text": str(value),
                    "confidence": 1.0,
                    "source": "tonerhound",
                })
                continue

            is_1040_root = (parent_record_path in ("", "root") or not any(k in path for k in ("schedule_", "form_")))
            is_bool = isinstance(value, bool) or ("_box" in path and not isinstance(value, (int, float)))

            # Case 1: Dependents table checkboxes on Form 1040 Page 1
            if is_1040_root and "dependents[" in path and is_bool and dep_header_y:
                m_dep = re.search(r"dependents\[(\d+)\]\.(child_tax_credit_box|credit_for_other_dependents_box)", path)
                if m_dep:
                    ridx = int(m_dep.group(1))
                    is_child = (m_dep.group(2) == "child_tax_credit_box")
                    row_y = dep_header_y + 0.0145 * (ridx + 1)
                    target_x = 0.778 if is_child else 0.870
                    box = BBox(target_x - 0.008, row_y - 0.008, 0.016, 0.017, p1040_p1)
                    citations.append({
                        "field_path": path,
                        "page": p1040_p1,
                        "bbox": box.to_coco(),
                        "reference_text": "[ ]",
                        "confidence": 0.95,
                        "source": "tonerhound",
                    })
                    continue

            # Case 2: Form 1040 Root Checkboxes (Presidential, Digital Assets, Age/Blindness, Line 16/35a boxes, etc.)
            if is_1040_root and is_bool:
                page_offset = 1 if any(k in path for k in ("line_16_", "line_35a_", "third_party_", "preparer_self_employed")) else 0
                target_p = p1040_p2 if page_offset == 1 else p1040_p1
                p_obj = self.index.get_page(target_p)
                if p_obj:
                    cb_box = self._resolve_checkbox(p_obj, path, value, page_offset)
                    if cb_box:
                        citations.append({
                            "field_path": path,
                            "page": target_p,
                            "bbox": cb_box.to_coco(),
                            "reference_text": "[ ]" if not value else "[X]",
                            "confidence": 0.95,
                            "source": "tonerhound",
                        })
                        continue

            # Case 3: Taxpayer vs. Spouse Name Disambiguation on Form 1040 Page 1
            if is_1040_root and path in ("taxpayer_first_name_mi", "taxpayer_last_name", "spouse_first_name_mi", "spouse_last_name"):
                if p1_obj:
                    is_spouse = "spouse" in path
                    is_last = "last" in path
                    row_yr = (0.11, 0.15) if is_spouse else (0.08, 0.12)
                    col_xr = (0.35, 0.65) if is_last else (0.04, 0.35)
                    name_cands: list[tuple[float, Any]] = []
                    val_upper = str(value).upper().strip()
                    for t in p1_obj.tokens:
                        cy = t.bbox.y + t.bbox.height / 2.0
                        cx = t.bbox.x + t.bbox.width / 2.0
                        if row_yr[0] <= cy <= row_yr[1] and col_xr[0] <= cx <= col_xr[1]:
                            sim = fuzz.ratio(val_upper, t.text.upper().strip()) / 100.0
                            if sim >= 0.70:
                                name_cands.append((sim, t))
                    if name_cands:
                        name_cands.sort(key=lambda x: x[0], reverse=True)
                        chosen = name_cands[0][1]
                        box = chosen.bbox.align_to_line_height(0.015) if self.enable_bbox_precision else chosen.bbox
                        citations.append({
                            "field_path": path,
                            "page": p1040_p1,
                            "bbox": box.to_coco(),
                            "reference_text": chosen.text,
                            "confidence": 0.95,
                            "source": "tonerhound",
                        })
                        continue

            # Case 4: Filing Status Checkbox on Form 1040 Page 1
            if is_1040_root and path == "filing_status" and p1_obj:
                tx = FILING_STATUS_X.get(str(value), 0.150)
                fs_y = 0.245
                for line in p1_obj.lines:
                    cy = line.bbox.y + line.bbox.height / 2.0
                    if 0.20 <= cy <= 0.28 and any(k in line.norm_text.lower() for k in ("filing status", "single", "jointly")):
                        fs_y = cy
                        break
                box = BBox(tx - 0.008, fs_y - 0.008, 0.016, 0.016, p1040_p1)
                citations.append({
                    "field_path": path,
                    "page": p1040_p1,
                    "bbox": box.to_coco(),
                    "reference_text": str(value),
                    "confidence": 0.95,
                    "source": "tonerhound",
                })
                continue

            # Case 5: Line-Number Visual Anchoring for Form 1040 numeric lines
            val_num = parse_numeric_value(value)
            is_num = (val_num is not None) and not isinstance(value, bool)
            m_line = re.search(r"line_([0-9]+[a-z]?)(?:_|$)", path)
            line_id = m_line.group(1) if m_line else None

            if is_1040_root and line_id and line_id in FORM_1040_LINES and is_num:
                p_offset, exp_y, exp_xr = FORM_1040_LINES[line_id]
                target_p = p1040_p2 if p_offset == 1 else p1040_p1
                p_obj = self.index.get_page(target_p)
                if p_obj:
                    num_cands: list[tuple[float, Any]] = []
                    for t in p_obj.tokens:
                        t_num = parse_numeric_value(t.text)
                        if t_num is not None and is_number_equal(t_num, val_num):
                            t_cy = t.bbox.y + t.bbox.height / 2.0
                            t_cx = t.bbox.x + t.bbox.width / 2.0
                            x_in_range = exp_xr[0] - 0.05 <= t_cx <= exp_xr[1] + 0.05
                            score = -abs(t_cy - exp_y) * 10.0 + (5.0 if x_in_range else -5.0)
                            num_cands.append((score, t))
                    if num_cands:
                        num_cands.sort(key=lambda x: x[0], reverse=True)
                        chosen = num_cands[0][1]
                        box = chosen.bbox.align_to_line_height(0.015) if self.enable_bbox_precision else chosen.bbox
                        citations.append({
                            "field_path": path,
                            "page": target_p,
                            "bbox": box.to_coco(),
                            "reference_text": chosen.text,
                            "confidence": 0.95,
                            "source": "tonerhound",
                        })
                        continue

            # Case 6: Form 1040 Section / Schedule-Scoped Resolution
            cand_pages: list[int] = []
            if is_1040_root:
                # Page 2 fields: lines 16-38, third party designee, signatures, paid preparer, firm
                if any(k in path for k in (
                    "line_16", "line_17", "line_18", "line_19", "line_20", "line_21", "line_22",
                    "line_23", "line_24", "line_25", "line_26", "line_27", "line_28", "line_29",
                    "line_30", "line_31", "line_32", "line_33", "line_34", "line_35", "line_36",
                    "line_37", "line_38", "third_party_", "designee_", "preparer_", "your_occupation",
                    "your_signature", "spouse_occupation", "spouse_signature", "firm_", "phone_no", "email_address",
                )):
                    cand_pages = [p1040_p2]
                else:
                    cand_pages = [p1040_p1]
            else:
                # Subsidiary schedule fields
                for s_name, plist in sched_pages.items():
                    if s_name in path:
                        cand_pages = plist
                        break

            best_res = None
            for cp in cand_pages[:3]:
                inp = ExtractionInput(field=path, value=value, field_context=context, page_hint=cp)
                res = self.resolver.resolve(inp)
                if res.is_grounded and res.page == cp and res.bbox is not None:
                    best_res = res
                    break

            # Fallback to general resolver if no candidates found on scoped pages
            if best_res is None and not is_1040_root:
                inp = ExtractionInput(field=path, value=value, field_context=context, page_hint=None)
                res = self.resolver.resolve(inp)
                if res.is_grounded and res.page is not None and res.bbox is not None:
                    best_res = res

            if best_res and best_res.is_grounded and best_res.page is not None and best_res.bbox is not None:
                box = best_res.bbox.align_to_line_height(0.015) if self.enable_bbox_precision else best_res.bbox
                citations.append({
                    "field_path": path,
                    "page": best_res.page,
                    "bbox": box.to_coco(),
                    "reference_text": best_res.matched_text,
                    "confidence": best_res.confidence,
                    "source": "tonerhound",
                })

        return {
            "task_type": "extract",
            "example_id": example_id,
            "pipeline_name": pipeline_name,
            "extracted_data": extracted_data,
            "field_citations": citations,
        }
