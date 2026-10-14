import sys
import json
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "research" / "reference" / "ExtractBench" / "src"))

from extract_bench.test_cases.loader import load_test_case

exp047_res_path = repo_root / "research" / "experiments" / "EXP-047" / "results" / "full_benchmark_results.json"
with open(exp047_res_path) as f:
    exp047_data = json.load(f)

eval_docs = [d for d in exp047_data["per_document"] if d["word_f1"] is not None]
print(f"Total evaluated documents: {len(eval_docs)}")

data_dir = repo_root / "research" / "data" / "full"

t0 = time.time()
field_counts = {}
for i, d in enumerate(eval_docs[:10]):
    tid = d["test_id"]
    pdf_path = data_dir / f"{tid}.pdf"
    if pdf_path.exists():
        tc = load_test_case(pdf_path)
        rules = tc.get_extract_field_rules()
        field_counts[tid] = len(rules)
    else:
        field_counts[tid] = None
print(f"Processed 10 docs in {time.time() - t0:.2f}s: {field_counts}")
