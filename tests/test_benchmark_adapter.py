"""End-to-End benchmark integration test: TonerHound + Official ExtractBench Evaluator."""

from __future__ import annotations

from extract_bench.test_cases.schema import ExtractFieldTestRule, FieldEvidence

from tonerhound import DocumentIndex, DocumentPage, DocumentToken, VisualLine
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.geometry.coordinates import BBox


def _build_benchmark_document() -> DocumentPage:
    """Document with invoice header, date, currency, and line items."""
    # Line 0: "ACME Industrial Supplies"
    t_vname = DocumentToken("ACME", BBox(0.1, 0.05, 0.10, 0.02, page=1), page=1, char_index_in_page=0, line_index=0)
    t_vname2 = DocumentToken("Industrial", BBox(0.21, 0.05, 0.15, 0.02, page=1), page=1, char_index_in_page=5, line_index=0)
    line0 = VisualLine([t_vname, t_vname2], page=1, line_index=0, bbox=BBox(0.1, 0.05, 0.26, 0.02, page=1))

    # Line 1: "Invoice Number: INV-98765"
    t_inlbl = DocumentToken("Invoice", BBox(0.1, 0.10, 0.08, 0.02, page=1), page=1, char_index_in_page=16, line_index=1)
    t_inlbl2 = DocumentToken("Number:", BBox(0.19, 0.10, 0.08, 0.02, page=1), page=1, char_index_in_page=24, line_index=1)
    t_inval = DocumentToken("INV-98765", BBox(0.30, 0.10, 0.14, 0.02, page=1), page=1, char_index_in_page=32, line_index=1)
    line1 = VisualLine([t_inlbl, t_inlbl2, t_inval], page=1, line_index=1, bbox=BBox(0.1, 0.10, 0.34, 0.02, page=1))

    # Line 2: "Invoice Date: 2026-03-15"
    t_dtlbl = DocumentToken("Invoice", BBox(0.1, 0.14, 0.08, 0.02, page=1), page=1, char_index_in_page=42, line_index=2)
    t_dtlbl2 = DocumentToken("Date:", BBox(0.19, 0.14, 0.06, 0.02, page=1), page=1, char_index_in_page=50, line_index=2)
    t_dtval = DocumentToken("2026-03-15", BBox(0.30, 0.14, 0.12, 0.02, page=1), page=1, char_index_in_page=56, line_index=2)
    line2 = VisualLine([t_dtlbl, t_dtlbl2, t_dtval], page=1, line_index=2, bbox=BBox(0.1, 0.14, 0.32, 0.02, page=1))

    # Line 3: Table Header: "Item Description Qty Unit Price Total"
    t_h1 = DocumentToken("Item", BBox(0.1, 0.22, 0.06, 0.02, page=1), page=1, char_index_in_page=67, line_index=3)
    t_h2 = DocumentToken("Description", BBox(0.2, 0.22, 0.15, 0.02, page=1), page=1, char_index_in_page=72, line_index=3)
    t_h3 = DocumentToken("Qty", BBox(0.45, 0.22, 0.05, 0.02, page=1), page=1, char_index_in_page=84, line_index=3)
    t_h4 = DocumentToken("Price", BBox(0.55, 0.22, 0.07, 0.02, page=1), page=1, char_index_in_page=88, line_index=3)
    line3 = VisualLine([t_h1, t_h2, t_h3, t_h4], page=1, line_index=3, bbox=BBox(0.1, 0.22, 0.52, 0.02, page=1))

    # Line 4: Row 1: "Steel Bolts 10 $25.00"
    r1_d = DocumentToken("Steel Bolts", BBox(0.1, 0.26, 0.16, 0.02, page=1), page=1, char_index_in_page=94, line_index=4)
    r1_q = DocumentToken("10", BBox(0.45, 0.26, 0.04, 0.02, page=1), page=1, char_index_in_page=106, line_index=4)
    r1_p = DocumentToken("$25.00", BBox(0.55, 0.26, 0.08, 0.02, page=1), page=1, char_index_in_page=109, line_index=4)
    line4 = VisualLine([r1_d, r1_q, r1_p], page=1, line_index=4, bbox=BBox(0.1, 0.26, 0.53, 0.02, page=1))

    # Line 5: Row 2: "Copper Washers 5 $50.00"
    r2_d = DocumentToken("Copper Washers", BBox(0.1, 0.30, 0.20, 0.02, page=1), page=1, char_index_in_page=117, line_index=5)
    r2_q = DocumentToken("5", BBox(0.45, 0.30, 0.03, 0.02, page=1), page=1, char_index_in_page=132, line_index=5)
    r2_p = DocumentToken("$50.00", BBox(0.55, 0.30, 0.08, 0.02, page=1), page=1, char_index_in_page=134, line_index=5)
    line5 = VisualLine([r2_d, r2_q, r2_p], page=1, line_index=5, bbox=BBox(0.1, 0.30, 0.53, 0.02, page=1))

    # Line 6: "Subtotal: $300.00"
    t_sublbl = DocumentToken("Subtotal:", BBox(0.35, 0.36, 0.10, 0.02, page=1), page=1, char_index_in_page=142, line_index=6)
    t_subval = DocumentToken("$300.00", BBox(0.55, 0.36, 0.09, 0.02, page=1), page=1, char_index_in_page=153, line_index=6)
    line6 = VisualLine([t_sublbl, t_subval], page=1, line_index=6, bbox=BBox(0.35, 0.36, 0.29, 0.02, page=1))

    # Line 7: "Tax: $24.00"
    t_txlbl = DocumentToken("Tax:", BBox(0.35, 0.40, 0.08, 0.02, page=1), page=1, char_index_in_page=161, line_index=7)
    t_txval = DocumentToken("$24.00", BBox(0.55, 0.40, 0.08, 0.02, page=1), page=1, char_index_in_page=166, line_index=7)
    line7 = VisualLine([t_txlbl, t_txval], page=1, line_index=7, bbox=BBox(0.35, 0.40, 0.28, 0.02, page=1))

    # Line 8: "Total Due: USD 324.00"
    t_totlbl = DocumentToken("Total Due:", BBox(0.35, 0.44, 0.12, 0.02, page=1), page=1, char_index_in_page=173, line_index=8)
    t_totval = DocumentToken("USD 324.00", BBox(0.55, 0.44, 0.14, 0.02, page=1), page=1, char_index_in_page=184, line_index=8)
    line8 = VisualLine([t_totlbl, t_totval], page=1, line_index=8, bbox=BBox(0.35, 0.44, 0.34, 0.02, page=1))

    tokens = [
        t_vname, t_vname2,
        t_inlbl, t_inlbl2, t_inval,
        t_dtlbl, t_dtlbl2, t_dtval,
        t_h1, t_h2, t_h3, t_h4,
        r1_d, r1_q, r1_p,
        r2_d, r2_q, r2_p,
        t_sublbl, t_subval,
        t_txlbl, t_txval,
        t_totlbl, t_totval,
    ]
    lines = [line0, line1, line2, line3, line4, line5, line6, line7, line8]
    return DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "vendor_name": {"type": "string"},
            "invoice_number": {"type": "string"},
            "invoice_date": {"type": "string"},
            "subtotal": {"type": "number"},
            "tax": {"type": "number"},
            "total_due": {"type": "number"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "price": {"type": "number"},
                    },
                },
            },
        },
    }


def _ground_truth_rules() -> tuple[dict, list[ExtractFieldTestRule]]:
    """Ground truth expected output and rules with bounding boxes."""
    expected_output = {
        "vendor_name": "ACME Industrial Supplies",
        "invoice_number": "INV-98765",
        "invoice_date": "2026-03-15",
        "subtotal": 300.0,
        "tax": 24.0,
        "total_due": 324.0,
        "items": [
            {"description": "Steel Bolts", "quantity": 10.0, "price": 25.0},
            {"description": "Copper Washers", "quantity": 5.0, "price": 50.0},
        ],
    }

    rules = [
        ExtractFieldTestRule(
            field_path="vendor_name",
            expected_value="ACME Industrial Supplies",
            evidence=[FieldEvidence(value="ACME Industrial", page=1, bbox=[0.1, 0.05, 0.26, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="invoice_number",
            expected_value="INV-98765",
            evidence=[FieldEvidence(value="INV-98765", page=1, bbox=[0.30, 0.10, 0.14, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="invoice_date",
            expected_value="2026-03-15",
            evidence=[FieldEvidence(value="2026-03-15", page=1, bbox=[0.30, 0.14, 0.12, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="subtotal",
            expected_value=300.0,
            evidence=[FieldEvidence(value="$300.00", page=1, bbox=[0.55, 0.36, 0.09, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="tax",
            expected_value=24.0,
            evidence=[FieldEvidence(value="$24.00", page=1, bbox=[0.55, 0.40, 0.08, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="total_due",
            expected_value=324.0,
            evidence=[FieldEvidence(value="USD 324.00", page=1, bbox=[0.55, 0.44, 0.14, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="items[0].description",
            expected_value="Steel Bolts",
            evidence=[FieldEvidence(value="Steel Bolts", page=1, bbox=[0.1, 0.26, 0.16, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="items[0].quantity",
            expected_value=10.0,
            evidence=[FieldEvidence(value="10", page=1, bbox=[0.45, 0.26, 0.04, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="items[0].price",
            expected_value=25.0,
            evidence=[FieldEvidence(value="$25.00", page=1, bbox=[0.55, 0.26, 0.08, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="items[1].description",
            expected_value="Copper Washers",
            evidence=[FieldEvidence(value="Copper Washers", page=1, bbox=[0.1, 0.30, 0.20, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="items[1].quantity",
            expected_value=5.0,
            evidence=[FieldEvidence(value="5", page=1, bbox=[0.45, 0.30, 0.03, 0.02])],
        ),
        ExtractFieldTestRule(
            field_path="items[1].price",
            expected_value=50.0,
            evidence=[FieldEvidence(value="$50.00", page=1, bbox=[0.55, 0.30, 0.08, 0.02])],
        ),
    ]
    return expected_output, rules


def test_baseline_vlm_has_zero_grounding():
    """Verify that a native VLM extraction with no citations scores 0.00% Word Grounding."""
    expected_output, rules = _ground_truth_rules()
    extracted_data = dict(expected_output)  # 100% accurate extracted values

    # Evaluated WITHOUT citations (e.g. OpenAI GPT-6 Astra, Gemini 3.8 Flash, Codex)
    unshaded_eval = evaluate_prediction(
        expected_output=expected_output,
        extracted_data=extracted_data,
        field_rules=rules,
        field_citations=[],
        data_schema=_schema(),
    )

    # Value F1 is 100% (1.0), but word grounding F1 is exactly 0.00% (0.0)!
    assert unshaded_eval["value_f1"] == 1.0
    assert unshaded_eval["word_grounding_f1"] == 0.0
    assert unshaded_eval["page_grounding_f1"] == 0.0


def test_tonerhound_grounding_evaluates_with_official_extractbench_metric():
    """Verify that TonerHound grounds the extraction and achieves > 90% Word Grounding F1."""
    doc_page = _build_benchmark_document()
    index = DocumentIndex.from_pages([doc_page])
    adapter = ExtractBenchAdapter(index)

    expected_output, rules = _ground_truth_rules()
    extracted_data = dict(expected_output)

    # Run TonerHound resolver adapter to ground the extracted fields
    grounded_output = adapter.ground_extracted_data(
        extracted_data=extracted_data,
        example_id="invoice_acme_001",
    )

    citations = grounded_output["field_citations"]
    assert len(citations) > 0

    # Evaluate with official ExtractBench evaluator
    results = evaluate_prediction(
        expected_output=expected_output,
        extracted_data=extracted_data,
        field_rules=rules,
        field_citations=citations,
        data_schema=_schema(),
        bbox_iou_threshold=0.50,
    )

    assert results["value_f1"] == 1.0
    assert results["page_grounding_f1"] == 1.0
    # TonerHound word grounding F1 must be > 0.90, shattering the 46.43% baseline!
    assert results["word_grounding_f1"] is not None
    assert results["word_grounding_f1"] >= 0.90
    assert results["word_grounding_precision"] == 1.0
    assert results["word_grounding_recall"] >= 0.90
