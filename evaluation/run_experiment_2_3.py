import json
import math
import os
from typing import Dict, List, Any
from app.retrieval import ProductRetriever
from app.filtered_retrieval import FilteredProductRetriever
from app.hybrid_retrieval import HybridProductRetriever
from app.rerank_retrieval import RerankedProductRetriever
from evaluation.evaluate_recall import (
    GOLDEN_DATASET_PATH,
    compute_relevance_grade,
    calculate_ndcg
)


def evaluate_retriever_model(retriever, questions: List[Dict[str, Any]], model_type: str = "dense", top_k: int = 5):
    """Run standardized evaluation across all 25 golden questions."""
    metrics = {
        "hr_at_1": 0,
        "hr_at_3": 0,
        "hr_at_5": 0,
        "ideal_hr_at_1": 0,
        "ideal_hr_at_3": 0,
        "mrr_list": [],
        "ndcg_1_list": [],
        "ndcg_3_list": [],
        "ndcg_5_list": [],
        "p_at_3_list": [],
        "p_at_5_list": [],
        "per_query": []
    }

    for item in questions:
        qid = item["id"]
        question = item["question"]
        known_grades = item.get("relevance_grades", {})

        if model_type == "metadata":
            results = retriever.search(question, top_k=top_k, use_filters=True)
        else:
            results = retriever.search(question, top_k=top_k)

        retrieved_metas = results["metadatas"][0] if results and "metadatas" in results else []
        grades = [compute_relevance_grade(m, item) for m in retrieved_metas]
        all_possible_grades = list(known_grades.values())
        if not all_possible_grades:
            all_possible_grades = [3, 3, 2, 2, 1]

        rel_ranks = [i + 1 for i, g in enumerate(grades) if g >= 2]
        ideal_ranks = [i + 1 for i, g in enumerate(grades) if g == 3]

        hit_1 = len(rel_ranks) > 0 and rel_ranks[0] <= 1
        hit_3 = len(rel_ranks) > 0 and rel_ranks[0] <= 3
        hit_5 = len(rel_ranks) > 0 and rel_ranks[0] <= 5

        if hit_1:
            metrics["hr_at_1"] += 1
        if hit_3:
            metrics["hr_at_3"] += 1
        if hit_5:
            metrics["hr_at_5"] += 1

        if ideal_ranks and ideal_ranks[0] <= 1:
            metrics["ideal_hr_at_1"] += 1
        if ideal_ranks and ideal_ranks[0] <= 3:
            metrics["ideal_hr_at_3"] += 1

        mrr = 1.0 / rel_ranks[0] if rel_ranks else 0.0
        metrics["mrr_list"].append(mrr)

        ndcg_1 = calculate_ndcg(grades, all_possible_grades, 1)
        ndcg_3 = calculate_ndcg(grades, all_possible_grades, 3)
        ndcg_5 = calculate_ndcg(grades, all_possible_grades, 5)

        metrics["ndcg_1_list"].append(ndcg_1)
        metrics["ndcg_3_list"].append(ndcg_3)
        metrics["ndcg_5_list"].append(ndcg_5)

        p_at_3 = sum(1 for g in grades[:3] if g >= 2) / 3.0
        p_at_5 = sum(1 for g in grades[:5] if g >= 2) / 5.0
        metrics["p_at_3_list"].append(p_at_3)
        metrics["p_at_5_list"].append(p_at_5)

        metrics["per_query"].append({
            "id": qid,
            "question": question,
            "category": item["category"],
            "query_type": item["query_type"],
            "grades": grades,
            "ndcg_3": ndcg_3,
            "ndcg_5": ndcg_5,
            "first_rel": rel_ranks[0] if rel_ranks else None,
            "retrieved": [
                {
                    "rank": i + 1,
                    "grade": g,
                    "id": m.get("product_id"),
                    "name": m.get("product_name", "")[:55],
                    "price": m.get("price", ""),
                    "subcat": m.get("category", "").split("|")[-1],
                    "cross_encoder_score": m.get("cross_encoder_score"),
                    "rrf_rank": m.get("rrf_rank")
                }
                for i, (g, m) in enumerate(zip(grades, retrieved_metas))
            ]
        })

    n = len(questions)
    return {
        "hr_at_1": (metrics["hr_at_1"] / n) * 100,
        "hr_at_3": (metrics["hr_at_3"] / n) * 100,
        "hr_at_5": (metrics["hr_at_5"] / n) * 100,
        "ideal_hr_at_1": (metrics["ideal_hr_at_1"] / n) * 100,
        "ideal_hr_at_3": (metrics["ideal_hr_at_3"] / n) * 100,
        "mrr": sum(metrics["mrr_list"]) / n,
        "precision_at_3": (sum(metrics["p_at_3_list"]) / n) * 100,
        "precision_at_5": (sum(metrics["p_at_5_list"]) / n) * 100,
        "ndcg_at_1": sum(metrics["ndcg_1_list"]) / n,
        "ndcg_at_3": sum(metrics["ndcg_3_list"]) / n,
        "ndcg_at_5": sum(metrics["ndcg_5_list"]) / n,
        "per_query": metrics["per_query"]
    }


def run_experiment_2_3():
    print("=" * 115)
    print("  PHASE 2.3 EXPERIMENT: DENSE vs. METADATA vs. HYBRID vs. CROSS-ENCODER RERANKER")
    print("=" * 115)

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} evaluation questions from golden dataset.\n")

    print("[1/4] Benchmarking Model A (Dense Baseline)...")
    dense_retriever = ProductRetriever()
    res_a = evaluate_retriever_model(dense_retriever, questions, model_type="dense", top_k=5)

    print("[2/4] Benchmarking Model B (Dense + Metadata-Filtered)...")
    filtered_retriever = FilteredProductRetriever()
    res_b = evaluate_retriever_model(filtered_retriever, questions, model_type="metadata", top_k=5)

    print("[3/4] Benchmarking Model C (Hybrid: Dense + BM25 + RRF)...")
    hybrid_retriever = HybridProductRetriever(rrf_k=60, dense_candidates=20, bm25_candidates=20)
    res_c = evaluate_retriever_model(hybrid_retriever, questions, model_type="hybrid", top_k=5)

    print("[4/4] Benchmarking Model D (Hybrid + Cross-Encoder Reranker)...")
    rerank_retriever = RerankedProductRetriever(candidate_pool_k=20, rrf_k=60)
    res_d = evaluate_retriever_model(rerank_retriever, questions, model_type="rerank", top_k=5)

    # 4-way Comparison Table
    print("\n" + "=" * 115)
    print("  PHASE 2.3 FOUR-WAY COMPARATIVE BENCHMARK TABLE")
    print("=" * 115)
    header = f"  {'Metric':<25} | {'Model A (Dense)':<15} | {'Model B (Meta)':<15} | {'Model C (Hybrid)':<16} | {'Model D (Rerank)':<16} | {'Δ (D vs C)':<10} | {'Δ (D vs A)':<10}"
    print(header)
    print("-" * 115)

    rows = [
        ("Hit Rate @ 1 (Rel >= 2)", res_a["hr_at_1"], res_b["hr_at_1"], res_c["hr_at_1"], res_d["hr_at_1"], "%"),
        ("Hit Rate @ 3 (Rel >= 2)", res_a["hr_at_3"], res_b["hr_at_3"], res_c["hr_at_3"], res_d["hr_at_3"], "%"),
        ("Hit Rate @ 5 (Rel >= 2)", res_a["hr_at_5"], res_b["hr_at_5"], res_c["hr_at_5"], res_d["hr_at_5"], "%"),
        ("Ideal Match @ 1 (Rel=3)", res_a["ideal_hr_at_1"], res_b["ideal_hr_at_1"], res_c["ideal_hr_at_1"], res_d["ideal_hr_at_1"], "%"),
        ("Ideal Match @ 3 (Rel=3)", res_a["ideal_hr_at_3"], res_b["ideal_hr_at_3"], res_c["ideal_hr_at_3"], res_d["ideal_hr_at_3"], "%"),
        ("MRR", res_a["mrr"], res_b["mrr"], res_c["mrr"], res_d["mrr"], "raw"),
        ("Precision @ 3", res_a["precision_at_3"], res_b["precision_at_3"], res_c["precision_at_3"], res_d["precision_at_3"], "%"),
        ("Precision @ 5", res_a["precision_at_5"], res_b["precision_at_5"], res_c["precision_at_5"], res_d["precision_at_5"], "%"),
        ("NDCG @ 1", res_a["ndcg_at_1"], res_b["ndcg_at_1"], res_c["ndcg_at_1"], res_d["ndcg_at_1"], "raw"),
        ("NDCG @ 3", res_a["ndcg_at_3"], res_b["ndcg_at_3"], res_c["ndcg_at_3"], res_d["ndcg_at_3"], "raw"),
        ("NDCG @ 5", res_a["ndcg_at_5"], res_b["ndcg_at_5"], res_c["ndcg_at_5"], res_d["ndcg_at_5"], "raw"),
    ]

    table_data = []
    for label, a, b, c, d, mode in rows:
        if mode == "%":
            str_a = f"{a:5.1f}%"
            str_b = f"{b:5.1f}%"
            str_c = f"{c:5.1f}%"
            str_d = f"{d:5.1f}%"
            delta_dc = f"{d - c:+5.1f}%"
            delta_da = f"{d - a:+5.1f}%"
        else:
            str_a = f"{a:6.3f}"
            str_b = f"{b:6.3f}"
            str_c = f"{c:6.3f}"
            str_d = f"{d:6.3f}"
            delta_dc = f"{d - c:+6.3f}"
            delta_da = f"{d - a:+6.3f}"
        print(f"  {label:<25} | {str_a:<15} | {str_b:<15} | {str_c:<16} | {str_d:<16} | {delta_dc:<10} | {delta_da:<10}")
        table_data.append((label, str_a, str_b, str_c, str_d, delta_dc, delta_da))

    print("=" * 115)

    # Deep-dive on Key Focus Queries
    focus_ids = ["Q02", "Q05", "Q11", "Q16", "Q22"]
    print("\n" + "=" * 115)
    print("  DEEP-DIVE ON THE 5 FOCUS QUERIES (Model A vs Model C vs Model D)")
    print("=" * 115)

    focus_reports = []
    query_lookup_a = {q["id"]: q for q in res_a["per_query"]}
    query_lookup_c = {q["id"]: q for q in res_c["per_query"]}
    query_lookup_d = {q["id"]: q for q in res_d["per_query"]}

    for qid in focus_ids:
        qa = query_lookup_a[qid]
        qc = query_lookup_c[qid]
        qd = query_lookup_d[qid]
        q_text = qa["question"]

        print(f"\n--- [{qid}] '{q_text}' ---")
        print(f"  NDCG@3 Progression: Dense={qa['ndcg_3']:.3f} -> Hybrid={qc['ndcg_3']:.3f} -> Rerank={qd['ndcg_3']:.3f}")
        print(f"  Model A (Dense)   Grades: {qa['grades']} | Top-1: {qa['retrieved'][0]['name']}")
        print(f"  Model C (Hybrid)  Grades: {qc['grades']} | Top-1: {qc['retrieved'][0]['name']}")
        print(f"  Model D (Rerank)  Grades: {qd['grades']} | Top-1: {qd['retrieved'][0]['name']}")
        print("  Top 3 Reranked Items:")
        for r in qd["retrieved"][:3]:
            score_str = f"Score={r['cross_encoder_score']:.4f}" if r["cross_encoder_score"] is not None else ""
            print(f"    Rank {r['rank']}: [Grade {r['grade']}] {r['name']} ({r['price']}) {score_str}")

        focus_reports.append((qid, q_text, qa, qc, qd))

    # Generate Markdown Report
    report_path = "evaluation/experiment_2_3_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2.3 Experiment Report: Cross-Encoder Reranking\n\n")
        f.write("**Date:** October 2026  \n")
        f.write("**Corpus:** `data/amazon.csv` (1,465 products) in ChromaDB  \n")
        f.write("**Evaluation Set:** 25 Baseline v1 queries with Graded Relevance (0–3 scale)  \n\n")
        f.write("## 1. Experimental Overview\n\n")
        f.write("In this experiment, we evaluated the full two-stage retrieval architecture:\n")
        f.write("$$\\text{Query} \\to \\text{Dense (20)} + \\text{BM25 (20)} \\to \\text{RRF Fusion} \\to \\text{Candidate Pool (20)} \\to \\text{Cross-Encoder (MiniLM-L12)} \\to \\text{Top 5}$$\n\n")
        f.write("## 2. Four-Way Comparative Benchmark Table\n\n")
        f.write("| Metric | Model A (Dense) | Model B (Metadata) | Model C (Hybrid RRF) | Model D (Cross-Encoder) | Δ (D vs C) | Δ (D vs A) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for label, a_str, b_str, c_str, d_str, d_c, d_a in table_data:
            f.write(f"| **{label}** | {a_str} | {b_str} | {c_str} | **{d_str}** | **{d_c}** | **{d_a}** |\n")

        f.write("\n## 3. Deep-Dive on the 5 Focus Queries\n\n")
        for qid, q_text, qa, qc, qd in focus_reports:
            f.write(f"### `[{qid}]` *\"{q_text}\"*\n")
            f.write(f"- **NDCG@3 Progression:** Model A = `{qa['ndcg_3']:.3f}` $\\to$ Model C = `{qc['ndcg_3']:.3f}` $\\to$ **Model D = `{qd['ndcg_3']:.3f}`**\n")
            f.write(f"- **Model A Grades:** `{qa['grades']}`\n")
            f.write(f"- **Model C Grades:** `{qc['grades']}`\n")
            f.write(f"- **Model D Grades:** `{qd['grades']}`\n\n")
            f.write("| Rank | Grade | Product Name | Price | Cross-Encoder Score |\n")
            f.write("| :---: | :---: | :--- | :---: | :---: |\n")
            for r in qd["retrieved"][:3]:
                score_str = f"{r['cross_encoder_score']:.4f}" if r["cross_encoder_score"] is not None else "N/A"
                f.write(f"| {r['rank']} | **{r['grade']}** | {r['name']} | {r['price']} | {score_str} |\n")
            f.write("\n")

        f.write("## 4. Key Engineering Insights\n\n")
        f.write("1. **Candidate Pool Quality Dictates Reranking Upper Bound:** The cross-encoder can only reorder items present in the 20-candidate pool generated by Stage 1.\n")
        f.write("2. **Query-Document Cross-Attention Corrects Lexical Polysemy:** The cross-encoder evaluates query tokens simultaneously with document tokens, resolving the keyword trap that misled unweighted RRF.\n")
        f.write("3. **Specialized Roles in Production Architecture:**\n")
        f.write("   - **Stage 1 (High Recall):** Dense + BM25 retrieves broad candidate coverage.\n")
        f.write("   - **Stage 1.5 (Hard Constraints):** Metadata guardrails enforce strict budget and catalog boundaries.\n")
        f.write("   - **Stage 2 (High Precision):** Cross-Encoder computes fine-grained semantic relevance.\n")

    print(f"\nReport successfully saved to: {report_path}")


if __name__ == "__main__":
    run_experiment_2_3()
