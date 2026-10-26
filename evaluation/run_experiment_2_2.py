import json
import math
import os
from typing import Dict, List, Any
from app.retrieval import ProductRetriever
from app.filtered_retrieval import FilteredProductRetriever
from app.hybrid_retrieval import HybridProductRetriever
from evaluation.evaluate_recall import (
    GOLDEN_DATASET_PATH,
    compute_relevance_grade,
    calculate_ndcg
)


def evaluate_model(retriever, questions: List[Dict[str, Any]], model_type: str = "dense", top_k: int = 5):
    """Evaluate a specific retrieval model across all questions."""
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
                    "name": m.get("product_name", "")[:50],
                    "price": m.get("price", ""),
                    "subcat": m.get("category", "").split("|")[-1],
                    "dense_rank": m.get("dense_rank"),
                    "bm25_rank": m.get("bm25_rank"),
                    "rrf_score": m.get("rrf_score")
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


def run_experiment_2_2():
    print("=" * 95)
    print("  PHASE 2.2 EXPERIMENT: DENSE vs. METADATA-FILTERED vs. HYBRID (DENSE+BM25+RRF)")
    print("=" * 95)

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} evaluation questions from golden dataset.\n")

    print("[1/3] Benchmarking Model A (Dense Baseline)...")
    dense_retriever = ProductRetriever()
    res_a = evaluate_model(dense_retriever, questions, model_type="dense", top_k=5)

    print("[2/3] Benchmarking Model B (Dense + Metadata-Filtered)...")
    filtered_retriever = FilteredProductRetriever()
    res_b = evaluate_model(filtered_retriever, questions, model_type="metadata", top_k=5)

    print("[3/3] Benchmarking Model C (Hybrid: Dense + BM25 + RRF)...")
    hybrid_retriever = HybridProductRetriever(rrf_k=60, dense_candidates=20, bm25_candidates=20)
    res_c = evaluate_model(hybrid_retriever, questions, model_type="hybrid", top_k=5)

    # 3-way Comparison Table
    print("\n" + "=" * 95)
    print("  PHASE 2.2 THREE-WAY COMPARATIVE BENCHMARK")
    print("=" * 95)
    header = f"  {'Metric':<25} | {'Model A (Dense)':<16} | {'Model B (Meta)':<16} | {'Model C (Hybrid)':<16} | {'Δ (C vs A)':<10}"
    print(header)
    print("-" * 95)

    rows = [
        ("Hit Rate @ 1 (Rel >= 2)", res_a["hr_at_1"], res_b["hr_at_1"], res_c["hr_at_1"], "%"),
        ("Hit Rate @ 3 (Rel >= 2)", res_a["hr_at_3"], res_b["hr_at_3"], res_c["hr_at_3"], "%"),
        ("Hit Rate @ 5 (Rel >= 2)", res_a["hr_at_5"], res_b["hr_at_5"], res_c["hr_at_5"], "%"),
        ("Ideal Match @ 1 (Rel=3)", res_a["ideal_hr_at_1"], res_b["ideal_hr_at_1"], res_c["ideal_hr_at_1"], "%"),
        ("Ideal Match @ 3 (Rel=3)", res_a["ideal_hr_at_3"], res_b["ideal_hr_at_3"], res_c["ideal_hr_at_3"], "%"),
        ("MRR", res_a["mrr"], res_b["mrr"], res_c["mrr"], "raw"),
        ("Precision @ 3", res_a["precision_at_3"], res_b["precision_at_3"], res_c["precision_at_3"], "%"),
        ("Precision @ 5", res_a["precision_at_5"], res_b["precision_at_5"], res_c["precision_at_5"], "%"),
        ("NDCG @ 1", res_a["ndcg_at_1"], res_b["ndcg_at_1"], res_c["ndcg_at_1"], "raw"),
        ("NDCG @ 3", res_a["ndcg_at_3"], res_b["ndcg_at_3"], res_c["ndcg_at_3"], "raw"),
        ("NDCG @ 5", res_a["ndcg_at_5"], res_b["ndcg_at_5"], res_c["ndcg_at_5"], "raw"),
    ]

    table_data = []
    for label, a, b, c, mode in rows:
        if mode == "%":
            str_a = f"{a:5.1f}%"
            str_b = f"{b:5.1f}%"
            str_c = f"{c:5.1f}%"
            delta = f"{c - a:+5.1f}%"
        else:
            str_a = f"{a:6.3f}"
            str_b = f"{b:6.3f}"
            str_c = f"{c:6.3f}"
            delta = f"{c - a:+6.3f}"
        print(f"  {label:<25} | {str_a:<16} | {str_b:<16} | {str_c:<16} | {delta:<10}")
        table_data.append((label, str_a, str_b, str_c, delta))

    print("=" * 95)

    # Per-Query Delta NDCG Analysis
    print("\n" + "=" * 95)
    print("  QUERY-LEVEL ΔNDCG ANALYSIS (Model C vs Model A)")
    print("=" * 95)

    improved = []
    regressed = []
    neutral = []

    for qa, qc in zip(res_a["per_query"], res_c["per_query"]):
        qid = qa["id"]
        q_text = qa["question"]
        delta_ndcg3 = qc["ndcg_3"] - qa["ndcg_3"]

        if delta_ndcg3 > 0.005:
            improved.append((qa, qc, delta_ndcg3))
        elif delta_ndcg3 < -0.005:
            regressed.append((qa, qc, delta_ndcg3))
        else:
            neutral.append((qa, qc))

    print(f"  Queries with Positive ΔNDCG : {len(improved)}")
    print(f"  Queries with Negative ΔNDCG : {len(regressed)}")
    print(f"  Queries with Neutral ΔNDCG  : {len(neutral)}\n")

    if improved:
        print("  --- QUERIES IMPROVED BY HYBRID RETRIEVAL ---")
        for qa, qc, diff in improved:
            print(f"  [{qa['id']}] '{qa['question']}'")
            print(f"     NDCG@3: {qa['ndcg_3']:.3f} -> {qc['ndcg_3']:.3f} (Δ = +{diff:.3f})")
            print(f"     Model A Grades: {qa['grades']} -> {[r['name'] for r in qa['retrieved'][:3]]}")
            print(f"     Model C Grades: {qc['grades']} -> {[r['name'] for r in qc['retrieved'][:3]]}")
            print()

    if regressed:
        print("  --- QUERIES REGRESSED BY HYBRID RETRIEVAL ---")
        for qa, qc, diff in regressed:
            print(f"  [{qa['id']}] '{qa['question']}'")
            print(f"     NDCG@3: {qa['ndcg_3']:.3f} -> {qc['ndcg_3']:.3f} (Δ = {diff:.3f})")
            print(f"     Model A Grades: {qa['grades']} -> {[r['name'] for r in qa['retrieved'][:3]]}")
            print(f"     Model C Grades: {qc['grades']} -> {[r['name'] for r in qc['retrieved'][:3]]}")
            print()

    # Generate Markdown Report
    report_path = "evaluation/experiment_2_2_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2.2 Experiment Report: Hybrid Retrieval (Dense + BM25 + RRF)\n\n")
        f.write("**Date:** October 2026  \n")
        f.write("**Corpus:** `data/amazon.csv` (1,465 products) in ChromaDB  \n")
        f.write("**Evaluation Set:** 25 Baseline v1 queries with Graded Relevance (0–3 scale)  \n\n")
        f.write("## 1. Hypothesis\n\n")
        f.write("> **H1:** Hybrid Dense + BM25 retrieval will improve ranking quality for queries containing exact product attributes, terminology, and constraints that dense semantic retrieval may blur (e.g. \"over ear\", specific model names, specifications).\n\n")
        f.write("## 2. Experimental Setup\n\n")
        f.write("- **Model A (Baseline):** Pure Dense Retrieval (`text-embedding-004`).\n")
        f.write("- **Model B (Metadata-Filtered):** Dense + Structured constraint filtering with dense fallback.\n")
        f.write("- **Model C (Hybrid):** Dense (top 20) + BM25Okapi (top 20) fused via Reciprocal Rank Fusion ($k=60$) -> Top 5.\n\n")
        f.write("## 3. Results Comparison Table\n\n")
        f.write("| Metric | Model A (Dense) | Model B (Metadata) | Model C (Hybrid RRF) | Delta Δ (C vs A) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for label, a_str, b_str, c_str, delta_str in table_data:
            f.write(f"| **{label}** | {a_str} | {b_str} | {c_str} | **{delta_str}** |\n")

        f.write("\n## 4. Query-Level ΔNDCG Breakdown\n\n")
        f.write(f"- **Queries with Positive ΔNDCG:** {len(improved)}\n")
        f.write(f"- **Queries with Negative ΔNDCG:** {len(regressed)}\n")
        f.write(f"- **Queries with Neutral ΔNDCG:** {len(neutral)}\n\n")

        if improved:
            f.write("### Queries Improved by Lexical BM25 Signal\n")
            for qa, qc, diff in improved:
                f.write(f"- **[{qa['id']}] `{qa['question']}`** (ΔNDCG@3 = +{diff:.3f}):\n")
                f.write(f"  - Model A Grades: `{qa['grades']}`\n")
                f.write(f"  - Model C Grades: `{qc['grades']}`\n")

        if regressed:
            f.write("### Queries Where Dense Was Superior\n")
            for qa, qc, diff in regressed:
                f.write(f"- **[{qa['id']}] `{qa['question']}`** (ΔNDCG@3 = {diff:.3f}):\n")
                f.write(f"  - Model A Grades: `{qa['grades']}`\n")
                f.write(f"  - Model C Grades: `{qc['grades']}`\n")

        f.write("\n## 5. Architectural Synthesis: The 3 Retrieval Pillars\n\n")
        f.write("| Retrieval Signal | Superpower | Blind Spot | Ideal Use Case |\n")
        f.write("| :--- | :--- | :--- | :--- | \n")
        f.write("| **Dense Semantic** | High-level intent, conceptual similarity, synonyms | Exact lexical terms (\"over-ear\" vs \"on-ear\"), numeric thresholds | Broad & exploratory queries |\n")
        f.write("| **Lexical BM25** | Exact keywords, technical codes, exact model numbers | Numeric inequalities (< 1500), semantic synonyms | Attribute & terminology queries |\n")
        f.write("| **Structured Metadata** | Hard numerical bounds (price <= 2000), strict categories | Noisy/sparse catalog tags, taxonomy rigidity | Hard filterable constraints |\n\n")

        f.write("## 6. Next Step: Phase 2.3 — Cross-Encoder Reranking\n")
        f.write("Take the top candidates from the hybrid candidate pool and pass them through a cross-encoder to compute full query-document attention scoring.\n")

    print(f"Report successfully saved to: {report_path}")


if __name__ == "__main__":
    run_experiment_2_2()
