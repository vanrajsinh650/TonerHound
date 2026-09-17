#!/usr/bin/env python3
"""Detailed 32-Document Pass 4 Analysis Generator."""

import json
from pathlib import Path
from collections import defaultdict

repo_root = Path(__file__).resolve().parent.parent

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    manifest = load_json(repo_root / "benchmarks/exp005_local_manifest.json")
    p4 = load_json(repo_root / "experiments/EXP-005-pass4-eval.json")
    v2 = load_json(repo_root / "experiments/EXP-005-table-dp-v2.json")
    p4_map = {d["test_id"]: d for d in p4["document_results"]}
    v2_map = {d["test_id"]: d for d in v2["document_results"]}

    print("==================================================================================")
    print("EXP-005 PASS 4 vs TABLE-DP-V2 COMPREHENSIVE BENCHMARK AUDIT")
    print("==================================================================================")

    # 1. Overall Aggregates
    print("\n### 1. OVERALL AGGREGATES")
    p4_ov = p4["overall"]
    v2_ov = v2["overall"]
    print(f"Overall Word F1:      {v2_ov['mean_word_grounding_f1']*100:.2f}% -> {p4_ov['mean_word_grounding_f1']*100:.2f}% (Delta: {(p4_ov['mean_word_grounding_f1']-v2_ov['mean_word_grounding_f1'])*100:+.2f} pp)")
    print(f"Overall Word Prec:    {v2_ov['mean_word_precision']*100:.2f}% -> {p4_ov['mean_word_precision']*100:.2f}% (Delta: {(p4_ov['mean_word_precision']-v2_ov['mean_word_precision'])*100:+.2f} pp)")
    print(f"Overall Word Recall:  {v2_ov['mean_word_recall']*100:.2f}% -> {p4_ov['mean_word_recall']*100:.2f}% (Delta: {(p4_ov['mean_word_recall']-v2_ov['mean_word_recall'])*100:+.2f} pp)")
    print(f"Overall Page F1:      {v2_ov['mean_page_grounding_f1']*100:.2f}% -> {p4_ov['mean_page_grounding_f1']*100:.2f}% (Delta: {(p4_ov['mean_page_grounding_f1']-v2_ov['mean_page_grounding_f1'])*100:+.2f} pp)")
    print(f"Overall Page Prec:    {v2_ov['mean_page_precision']*100:.2f}% -> {p4_ov['mean_page_precision']*100:.2f}% (Delta: {(p4_ov['mean_page_precision']-v2_ov['mean_page_precision'])*100:+.2f} pp)")
    print(f"Overall Page Recall:  {v2_ov['mean_page_recall']*100:.2f}% -> {p4_ov['mean_page_recall']*100:.2f}% (Delta: {(p4_ov['mean_page_recall']-v2_ov['mean_page_recall'])*100:+.2f} pp)")
    print(f"False Grounding Rate: {v2_ov['mean_false_grounding_rate']*100:.2f}% -> {p4_ov['mean_false_grounding_rate']*100:.2f}% (Delta: {(p4_ov['mean_false_grounding_rate']-v2_ov['mean_false_grounding_rate'])*100:+.2f} pp)")
    print(f"Total Citations:      {v2_ov['total_citations']:,} -> {p4_ov['total_citations']:,} (+{p4_ov['total_citations']-v2_ov['total_citations']:,})")
    print(f"Suite Runtime:        {v2['suite_total_time_sec']:.2f}s ({v2['suite_total_time_sec']/60:.2f}m) -> {p4['suite_total_time_sec']:.2f}s ({p4['suite_total_time_sec']/60:.2f}m)")

    # 2. Slices
    print("\n### 2. SLICE SPLITS")
    for s_name in ["short", "medium", "long"]:
        v2_s = v2["slices"][s_name]
        p4_s = p4["slices"][s_name]
        print(f"[{s_name.upper()}] (n={v2_s['count']}, {v2_s['total_pages']} pgs)")
        print(f"  Word F1:      {v2_s['mean_word_grounding_f1']*100:.2f}% -> {p4_s['mean_word_grounding_f1']*100:.2f}% ({(p4_s['mean_word_grounding_f1']-v2_s['mean_word_grounding_f1'])*100:+.2f} pp)")
        print(f"  Word Prec:    {v2_s['mean_word_precision']*100:.2f}% -> {p4_s['mean_word_precision']*100:.2f}% ({(p4_s['mean_word_precision']-v2_s['mean_word_precision'])*100:+.2f} pp)")
        print(f"  Word Recall:  {v2_s['mean_word_recall']*100:.2f}% -> {p4_s['mean_word_recall']*100:.2f}% ({(p4_s['mean_word_recall']-v2_s['mean_word_recall'])*100:+.2f} pp)")
        print(f"  Page F1:      {v2_s['mean_page_grounding_f1']*100:.2f}% -> {p4_s['mean_page_grounding_f1']*100:.2f}% ({(p4_s['mean_page_grounding_f1']-v2_s['mean_page_grounding_f1'])*100:+.2f} pp)")
        print(f"  Latency:      {v2_s['total_latency_sec']:.2f}s -> {p4_s['total_latency_sec']:.2f}s")

    for s_name in ["train_dev", "local_validation"]:
        v2_s = v2[s_name]
        p4_s = p4[s_name]
        print(f"[{s_name.upper()}] (n={v2_s['count']}, {v2_s['total_pages']} pgs)")
        print(f"  Word F1:      {v2_s['mean_word_grounding_f1']*100:.2f}% -> {p4_s['mean_word_grounding_f1']*100:.2f}% ({(p4_s['mean_word_grounding_f1']-v2_s['mean_word_grounding_f1'])*100:+.2f} pp)")
        print(f"  Page F1:      {v2_s['mean_page_grounding_f1']*100:.2f}% -> {p4_s['mean_page_grounding_f1']*100:.2f}% ({(p4_s['mean_page_grounding_f1']-v2_s['mean_page_grounding_f1'])*100:+.2f} pp)")

    # 3. Document details
    print("\n### 3. ALL 32 DOCUMENTS COMPARISON TABLE")
    rows = []
    for tid, p_doc in p4_map.items():
        v_doc = v2_map[tid]
        d_wf1 = (p_doc["word_grounding_f1"] - v_doc["word_grounding_f1"]) * 100
        d_pf1 = (p_doc["page_grounding_f1"] - v_doc["page_grounding_f1"]) * 100
        rows.append({
            "tid": tid,
            "split": p_doc["split"],
            "len": p_doc["length_class"],
            "domain": p_doc["domain"],
            "pages": p_doc["num_pages"],
            "cites": p_doc["num_citations"],
            "v2_wf1": v_doc["word_grounding_f1"] * 100,
            "p4_wf1": p_doc["word_grounding_f1"] * 100,
            "delta_wf1": d_wf1,
            "v2_pf1": v_doc["page_grounding_f1"] * 100,
            "p4_pf1": p_doc["page_grounding_f1"] * 100,
            "delta_pf1": d_pf1,
            "fg": p_doc["false_grounding_rate"] * 100,
            "time": p_doc["total_time_sec"],
        })

    rows.sort(key=lambda x: x["delta_wf1"], reverse=True)

    print(f"| {'Test ID':<46} | {'Split':<10} | {'Len':<6} | {'Domain':<22} | {'Pgs':<4} | {'Cites':<6} | {'v2 F1':<7} | {'p4 F1':<7} | {'Delta F1':<9} | {'v2 PF1':<7} | {'p4 PF1':<7} | {'Delta PF1':<9} | {'Time (s)':<8} |")
    print("|" + "-"*48 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*24 + "|" + "-"*6 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*11 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*11 + "|" + "-"*10 + "|")

    for r in rows:
        print(f"| {r['tid']:<46} | {r['split']:<10} | {r['len']:<6} | {r['domain']:<22} | {r['pages']:<4} | {r['cites']:<6} | {r['v2_wf1']:6.2f}% | {r['p4_wf1']:6.2f}% | {r['delta_wf1']:+7.2f}pp | {r['v2_pf1']:6.2f}% | {r['p4_pf1']:6.2f}% | {r['delta_pf1']:+7.2f}pp | {r['time']:8.2f} |")

    # Gains, Losses, Neutral
    gains = [r for r in rows if r["delta_wf1"] > 0.05]
    neutrals = [r for r in rows if abs(r["delta_wf1"]) <= 0.05]
    regressions = [r for r in rows if r["delta_wf1"] < -0.05]

    print(f"\nTotal Gains: {len(gains)} documents")
    print(f"Total Neutral: {len(neutrals)} documents")
    print(f"Total Regressions: {len(regressions)} documents")

    print("\n--- TOP GAINERS ---")
    for r in gains[:5]:
        print(f"  + {r['tid']}: {r['v2_wf1']:.2f}% -> {r['p4_wf1']:.2f}% ({r['delta_wf1']:+.2f} pp) [Domain: {r['domain']}]")

    print("\n--- REGRESSIONS ---")
    for r in regressions:
        print(f"  - {r['tid']}: {r['v2_wf1']:.2f}% -> {r['p4_wf1']:.2f}% ({r['delta_wf1']:+.2f} pp) [Domain: {r['domain']}]")

if __name__ == "__main__":
    main()
