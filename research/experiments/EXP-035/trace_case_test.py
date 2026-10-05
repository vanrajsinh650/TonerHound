import json
import sys
from pathlib import Path

repo_root = Path(".")
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput
from tonerhound.resolution.resolver import EvidenceResolver

with open("research/experiments/EXP-035/alleged_indexing_miss_fields.json") as f:
    alleged = json.load(f)["cases"]

for sample in alleged[:5]:
    print("=" * 60)
    print("Sample case:", sample["document_id"], sample["field_path"], sample["value"])
    pdf_path = repo_root / "research" / "data" / "full" / f"{sample['document_id']}.pdf"
    idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True)
    resolver = EvidenceResolver(idx)

    inp = ExtractionInput(field=sample["field_path"], value=sample["value"], page_hint=sample["gold_page"])
    cands = resolver.collect_candidates(inp)
    print("Collected candidates:", len(cands))
    for c in cands[:3]:
        print("  cand:", c.page, c.bbox.to_coco(), c.matched_text, c.match_type, "IoU:", iou_xywh(c.bbox.to_coco(), sample["gold_bbox"]))
    res = resolver.resolve(inp)
    iou = iou_xywh(res.bbox.to_coco(), sample["gold_bbox"]) if res.bbox else 0.0
    print("Resolved:", res.is_grounded, res.status, res.explanation, res.page, res.bbox.to_coco() if res.bbox else None, "IoU:", iou)


