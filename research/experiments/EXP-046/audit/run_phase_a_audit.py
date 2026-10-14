"""EXP-046 Phase A Audit: Analyze table structure and drift across 47 SUB_AMBIGUOUS_OCCURRENCE documents."""

from __future__ import annotations

import json
from pathlib import Path
import pymupdf

from extract_bench.test_cases.loader import load_test_case

def run_audit() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    exp044_audit_p = repo_root / "research" / "experiments" / "EXP-044" / "audit" / "subclass_breakdown.json"
    audit_out_dir = Path(__file__).resolve().parent
    data_dir = repo_root / "research" / "data" / "full"

    with open(exp044_audit_p, encoding="utf-8") as f:
        subclass_data = json.load(f)

    # Find the SUB_AMBIGUOUS_OCCURRENCE entry
    ambig_entry = None
    for entry in subclass_data:
        if entry.get("subclass") == "SUB_AMBIGUOUS_OCCURRENCE":
            ambig_entry = entry
            break

    # Get sample documents or find documents containing array paths
    # Load held-out manifest and full data to extract affected documents
    affected_docs = set()
    if ambig_entry and "examples" in ambig_entry:
        for ex in ambig_entry["examples"]:
            doc_id = ex.split("::")[0]
            affected_docs.add(doc_id)

    # Also scan held-out and other documents with array patterns
    manifest_p = repo_root / "benchmarks" / "held_out_manifest.json"
    with open(manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)

    for doc_info in manifest["documents"]:
        tid = doc_info["test_id"]
        pdf_path = data_dir / f"{tid}.pdf"
        if not pdf_path.exists():
            continue
        tc = load_test_case(pdf_path)
        rules = tc.get_extract_field_rules()
        if any("[" in r.field_path and "]" in r.field_path for r in rules):
            affected_docs.add(tid)

    print(f"Auditing {len(affected_docs)} documents with array/table fields...")

    drift_records = []
    table_structure_all = []

    for doc_id in sorted(list(affected_docs)):
        pdf_path = data_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            continue

        tc = load_test_case(pdf_path)
        rules = tc.get_extract_field_rules()
        array_rules = [r for r in rules if "[" in r.field_path and "]" in r.field_path]
        
        # Max schema index across array rules
        max_idx = 0
        for r in array_rules:
            try:
                idx = int(r.field_path.split("[")[1].split("]")[0])
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
        expected_rows = max_idx + 1 if array_rules else 0

        doc = pymupdf.open(pdf_path)
        lines_tabs_count = 0
        text_tabs_count = 0
        total_physical_rows = 0

        doc_tables_meta = []
        for p_idx in range(len(doc)):
            page = doc[p_idx]
            tabs_l = page.find_tables(strategy="lines")
            tabs_t = page.find_tables(strategy="text")
            lines_tabs_count += len(tabs_l.tables)
            text_tabs_count += len(tabs_t.tables)

            t_used = tabs_l.tables if tabs_l.tables else tabs_t.tables
            for t_i, t in enumerate(t_used):
                r_count = len(t.rows)
                total_physical_rows += r_count
                doc_tables_meta.append({
                    "page": p_idx + 1,
                    "table_index": t_i,
                    "rows": r_count,
                    "bbox": list(t.bbox),
                    "header": [str(h) for h in t.header.names] if t.header else []
                })
        doc.close()

        drift = total_physical_rows - expected_rows
        drift_type = "over_segmented" if drift > 0 else ("under_segmented" if drift < 0 else "exact")

        rec = {
            "document_id": doc_id,
            "lines_tables": lines_tabs_count,
            "text_tables": text_tabs_count,
            "physical_rows": total_physical_rows,
            "expected_rows": expected_rows,
            "drift": drift,
            "drift_type": drift_type
        }
        drift_records.append(rec)
        table_structure_all.append({
            "document_id": doc_id,
            "tables": doc_tables_meta,
            "summary": rec
        })

    with open(audit_out_dir / "drift_analysis_47_docs.json", "w", encoding="utf-8") as f:
        json.dump(drift_records, f, indent=2)

    with open(audit_out_dir / "table_structure_all.json", "w", encoding="utf-8") as f:
        json.dump(table_structure_all, f, indent=2)

    print(f"Audit completed. Processed {len(drift_records)} documents.")
    over_count = sum(1 for r in drift_records if r["drift_type"] == "over_segmented")
    under_count = sum(1 for r in drift_records if r["drift_type"] == "under_segmented")
    exact_count = sum(1 for r in drift_records if r["drift_type"] == "exact")
    print(f"Drift summary: Over-segmented: {over_count}, Under-segmented: {under_count}, Exact: {exact_count}")

if __name__ == "__main__":
    run_audit()
