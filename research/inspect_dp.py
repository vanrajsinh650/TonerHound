import sys
sys.path.insert(0, ".")
from pathlib import Path
from tonerhound.document.index import DocumentIndex
from tonerhound.benchmark.adapter import _matches_table_line, parse_numeric_value
from research.analyze_true_geometry import page_stats, tc
import numpy as np

print("Loading index...")
idx = DocumentIndex.from_pdf("research/data/full/long/real_ftx_full_corrupted.pdf", enable_ocr=True)

# Replicate the DP matching in _align_table_arrays to see what it finds
rows_map = {}
for r_idx, c in enumerate(tc.expected_output["creditors"]):
    leaves = []
    for k, v in c.items():
        leaves.append((f"creditors[{r_idx}].{k}", v, c.get("source_page"), None, f"creditors[{r_idx}]"))
    rows_map[r_idx] = leaves

# Group by page
page_to_rows = {}
for r_idx, c in enumerate(tc.expected_output["creditors"]):
    p = c.get("source_page", 2)
    page_to_rows.setdefault(p, []).append(r_idx)

dp_results = []
for p_num in sorted(page_to_rows.keys()):
    page_rows = page_to_rows[p_num]
    page_obj = idx.get_page(p_num)
    if not page_obj or not page_obj.lines:
        continue
        
    cand_lines = [
        l for l in page_obj.lines
        if 0.03 <= l.bbox.y <= 0.97
        and not any(hp in l.norm_text.lower() for hp in (
            "page ", "form 13f", "omb no", "case ", "creditor matrix",
            "creditor name", "address 1", "attention address",
            "item description", "title of class", "name of issuer",
        ))
    ]
    cand_lines.sort(key=lambda l: l.bbox.y)
    
    boilerplate_salient = {
        "NAME ON FILE", "ADDRESS ON FILE", "NONE", "N/A", "CA", "USA",
        "TRUE", "FALSE", "YES", "NO", "NULL", "UNKNOWN",
    }
    row_salient_strings = []
    for r_idx in page_rows:
        salient = []
        for _p, v, _ph, _ctx, _rp in rows_map[r_idx]:
            if v is not None and not isinstance(v, bool):
                vs = str(v).strip().upper()
                if len(vs) >= 2 and vs not in boilerplate_salient:
                    salient.append(vs)
                pnum = parse_numeric_value(v)
                if pnum is not None and pnum.is_integer() and abs(pnum) >= 1000:
                    formatted_commas = f"{int(pnum):,}"
                    if formatted_commas != vs:
                        salient.append(formatted_commas)
        row_salient_strings.append(salient)

    col0_x_votes = []
    for row_strs in row_salient_strings:
        if row_strs:
            first_s = row_strs[0]
            for l in cand_lines:
                if first_s in l.norm_text.upper() and l.tokens:
                    col0_x_votes.append(l.tokens[0].bbox.x)
                    break

    filtered_lines = cand_lines
    if len(col0_x_votes) >= 5:
        col0_x_votes.sort()
        med_col0 = col0_x_votes[len(col0_x_votes) // 2]
        col_matching_lines = [
            l for l in cand_lines
            if l.tokens and abs(l.tokens[0].bbox.x - med_col0) <= 0.08
        ]
        if len(col_matching_lines) >= len(page_rows) * 0.8:
            filtered_lines = col_matching_lines

    M = len(page_rows)
    N = len(filtered_lines)

    dp = [[0.0] * (N + 1) for _ in range(M + 1)]
    parent_dp = [[(-1, -1)] * (N + 1) for _ in range(M + 1)]

    for i in range(1, M + 1): parent_dp[i][0] = (i - 1, 0)
    for j in range(1, N + 1): parent_dp[0][j] = (0, j - 1)

    for i in range(1, M + 1):
        r_strs = row_salient_strings[i - 1]
        for j in range(1, N + 1):
            b_val = dp[i][j - 1]
            b_p = (i, j - 1)
            if dp[i - 1][j] > b_val:
                b_val = dp[i - 1][j]
                b_p = (i - 1, j)
            line_txt = filtered_lines[j - 1].norm_text.upper()
            match_count = sum(1 for s in r_strs if _matches_table_line(s, line_txt))
            if match_count > 0:
                match_score = match_count * 4.0
                if dp[i - 1][j - 1] + match_score > b_val:
                    b_val = dp[i - 1][j - 1] + match_score
                    b_p = (i - 1, j - 1)
            dp[i][j] = b_val
            parent_dp[i][j] = b_p

    aligned_indices = []
    curr_i, curr_j = M, N
    while curr_i > 0 and curr_j > 0:
        pi, pj = parent_dp[curr_i][curr_j]
        if pi == curr_i - 1 and pj == curr_j - 1:
            aligned_row_idx = page_rows[curr_i - 1]
            aligned_line = filtered_lines[curr_j - 1]
            r_strs = row_salient_strings[curr_i - 1]
            line_txt = aligned_line.norm_text.upper()
            if any(_matches_table_line(s, line_txt) for s in r_strs):
                aligned_indices.append((aligned_row_idx, aligned_line))
        curr_i, curr_j = pi, pj
    aligned_indices.reverse()
    
    dp_results.append({
        "page": p_num,
        "n_aligned": len(aligned_indices),
        "aligned": aligned_indices,
    })

print(f"DP matching summary across all {len(dp_results)} pages:")
n_aligned_list = [d["n_aligned"] for d in dp_results]
print(f"  Mean aligned rows per page: {np.mean(n_aligned_list):.1f}")
print(f"  Min aligned rows          : {np.min(n_aligned_list)}")
print(f"  Max aligned rows          : {np.max(n_aligned_list)}")
print(f"  Pages with >= 5 aligned   : {sum(1 for n in n_aligned_list if n >= 5)} / {len(dp_results)}")
print(f"  Pages with >= 10 aligned  : {sum(1 for n in n_aligned_list if n >= 10)} / {len(dp_results)}")
