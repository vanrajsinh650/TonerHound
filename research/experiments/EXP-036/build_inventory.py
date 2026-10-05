"""EXP-036 Phase 0: Build and verify the 286-field checkbox inventory."""

from __future__ import annotations

import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
data_dir = repo_root / "research" / "data" / "full"

audit_path = repo_root / "research" / "experiments" / "EXP-035" / "failure_audit.json"
with open(audit_path, encoding="utf-8") as f:
    audit_data = json.load(f)

cases = audit_data["cases"]
bool_cases = [c for c in cases if c["failure_mechanism"] == "WRONG_CLASSIFICATION"]

inventory = []
for idx, c in enumerate(bool_cases, 1):
    doc_id = c["document_id"]
    pdf_path = data_dir / f"{doc_id}.pdf"
    assert pdf_path.exists(), f"PDF missing: {pdf_path}"

    record = {
        "index": idx,
        "document_id": doc_id,
        "field_path": c["field_path"],
        "value": c["value"],
        "gold_page": c["gold_page"],
        "gold_bbox": c["gold_bbox"],
        "pdf_path": str(pdf_path.relative_to(repo_root)),
    }
    inventory.append(record)

# VERIFY EXACT COUNT OF 286
assert len(inventory) == 286, f"Expected exactly 286 records, found {len(inventory)}"

out_path = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "experiment": "EXP-036_PHASE_0",
        "total_records": len(inventory),
        "unique_documents": len(set(r["document_id"] for r in inventory)),
        "inventory": inventory,
    }, f, indent=2)

print(f"Verified inventory count: {len(inventory)} records across {len(set(r['document_id'] for r in inventory))} documents.")
print(f"Saved to {out_path}")
