import json
import math
import os
import re
from collections import defaultdict
from app.retrieval import ProductRetriever

GOLDEN_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "golden_dataset", "questions.json"
)


def parse_price_inr(price_str):
    """Parse price string like '₹1,299' to numeric float."""
    if not price_str:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(price_str))
    try:
        return float(cleaned)
    except ValueError:
        return None


def compute_relevance_grade(product_meta, question_item):
    """
    Compute graded relevance score (0 to 3) for a product against a query item:
      3 = Highly Relevant / Ideal (satisfies category, intent, brand & constraints)
      2 = Relevant (good alternative / close substitute)
      1 = Partially Relevant (adjacent accessory or misses secondary constraint)
      0 = Irrelevant (wrong category or violates core constraint)
    """
    prod_id = str(product_meta.get("product_id", "")).strip()
    prod_name = str(product_meta.get("product_name", "")).lower()
    prod_cat = str(product_meta.get("category", "")).lower()
    prod_price = parse_price_inr(product_meta.get("price"))

    # 1. Exact ground-truth judgment check
    relevance_grades = question_item.get("relevance_grades", {})
    if prod_id in relevance_grades:
        return relevance_grades[prod_id]

    # 2. Rule-based criteria assessment
    criteria = question_item.get("intent_criteria", {})
    target_subcat = criteria.get("target_subcategory", "").lower()
    req_keywords = [k.lower() for k in criteria.get("required_keywords", [])]
    max_price = criteria.get("max_price")

    # Category compatibility check
    if target_subcat and target_subcat not in prod_cat:
        # Check if adjacent category
        if any(kw in prod_name for kw in req_keywords[:2]):
            return 1
        return 0

    # Keyword match count
    matched_kws = sum(1 for kw in req_keywords if kw in prod_name or kw in prod_cat)
    ratio = matched_kws / max(len(req_keywords), 1)

    # Price constraint check
    if max_price and prod_price:
        if prod_price > max_price * 1.3:
            # Over budget by >30%
            return 1 if ratio >= 0.5 else 0

    if ratio >= 0.6:
        return 3
    elif ratio >= 0.3:
        return 2
    elif ratio > 0:
        return 1
    return 0


def calculate_dcg(grades, k):
    """Calculate Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i, rel in enumerate(grades[:k]):
        # Standard IR formulation: (2^rel - 1) / log2(rank + 1)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 2)  # i=0 -> rank 1 -> log2(2) = 1.0
        dcg += gain / discount
    return dcg


def calculate_ndcg(grades, all_available_grades, k):
    """Calculate Normalized Discounted Cumulative Gain at rank K."""
    dcg = calculate_dcg(grades, k)
    # Ideal ranking sorts all possible positive grades descending
    sorted_ideal = sorted(all_available_grades, reverse=True)
    idcg = calculate_dcg(sorted_ideal, k)
    if idcg <= 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def run_baseline_evaluation(top_k=5):
    print("=" * 80)
    print("  MULTIMODAL RAG: RETRIEVER BASELINE v1 EVALUATION (GRADED RELEVANCE)")
    print("=" * 80)

    if not os.path.exists(GOLDEN_DATASET_PATH):
        raise FileNotFoundError(f"Golden dataset not found at: {GOLDEN_DATASET_PATH}")

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"Loaded {len(golden_data)} golden evaluation questions.")
    print("Relevance Scale: 3=Ideal | 2=Relevant | 1=Partial | 0=Irrelevant")
    print(f"Evaluating top_k={top_k} retrieval performance...\n")

    retriever = ProductRetriever()

    metrics = {
        "hr_at_1": 0,
        "hr_at_3": 0,
        "hr_at_5": 0,
        "ideal_hr_at_1": 0,  # HR with grade == 3
        "ideal_hr_at_3": 0,
        "mrr_list": [],
        "ndcg_1_list": [],
        "ndcg_3_list": [],
        "ndcg_5_list": [],
        "p_at_3_list": [],
        "p_at_5_list": [],
    }

    cat_stats = defaultdict(lambda: {"total": 0, "hits_3": 0, "ndcg_3": []})
    type_stats = defaultdict(lambda: {"total": 0, "hits_3": 0, "ndcg_3": []})
    suboptimal_queries = []

    for item in golden_data:
        qid = item["id"]
        question = item["question"]
        category = item["category"]
        query_type = item["query_type"]
        known_grades = item.get("relevance_grades", {})

        results = retriever.search(question, top_k=top_k)
        retrieved_metas = results["metadatas"][0] if results and "metadatas" in results else []

        grades = [compute_relevance_grade(m, item) for m in retrieved_metas]
        all_possible_grades = list(known_grades.values())
        if not all_possible_grades:
            all_possible_grades = [3, 3, 2, 2, 1]

        # Hit Rate (rel >= 2)
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

        # MRR (first rel >= 2)
        mrr = 1.0 / rel_ranks[0] if rel_ranks else 0.0
        metrics["mrr_list"].append(mrr)

        # NDCG @ 1, 3, 5
        ndcg_1 = calculate_ndcg(grades, all_possible_grades, 1)
        ndcg_3 = calculate_ndcg(grades, all_possible_grades, 3)
        ndcg_5 = calculate_ndcg(grades, all_possible_grades, 5)

        metrics["ndcg_1_list"].append(ndcg_1)
        metrics["ndcg_3_list"].append(ndcg_3)
        metrics["ndcg_5_list"].append(ndcg_5)

        # Precision @ K (rel >= 2)
        p_at_3 = sum(1 for g in grades[:3] if g >= 2) / 3.0
        p_at_5 = sum(1 for g in grades[:5] if g >= 2) / 5.0
        metrics["p_at_3_list"].append(p_at_3)
        metrics["p_at_5_list"].append(p_at_5)

        # Category & Type aggregations
        cat_stats[category]["total"] += 1
        if hit_3:
            cat_stats[category]["hits_3"] += 1
        cat_stats[category]["ndcg_3"].append(ndcg_3)

        type_stats[query_type]["total"] += 1
        if hit_3:
            type_stats[query_type]["hits_3"] += 1
        type_stats[query_type]["ndcg_3"].append(ndcg_3)

        grades_str = str(grades)
        first_rel = f"Rank {rel_ranks[0]}" if rel_ranks else "MISS"
        print(f"[{qid}] '{question[:30]:<30}' | Grades: {grades_str:<15} | First Rel: {first_rel:<7} | NDCG@3: {ndcg_3:.2f}")

        if ndcg_3 < 0.95 or (rel_ranks and rel_ranks[0] > 1) or not rel_ranks:
            suboptimal_queries.append({
                "id": qid,
                "question": question,
                "category": category,
                "grades": grades,
                "ndcg_3": ndcg_3,
                "first_hit": rel_ranks[0] if rel_ranks else None,
                "retrieved": [
                    {
                        "rank": i + 1,
                        "grade": g,
                        "name": m.get("product_name", "")[:55],
                        "price": m.get("price", ""),
                        "subcat": m.get("category", "").split("|")[-1]
                    }
                    for i, (g, m) in enumerate(zip(grades, retrieved_metas))
                ]
            })

    n = len(golden_data)
    mean_ndcg_1 = sum(metrics["ndcg_1_list"]) / n
    mean_ndcg_3 = sum(metrics["ndcg_3_list"]) / n
    mean_ndcg_5 = sum(metrics["ndcg_5_list"]) / n
    mean_mrr = sum(metrics["mrr_list"]) / n
    mean_p_3 = sum(metrics["p_at_3_list"]) / n * 100
    mean_p_5 = sum(metrics["p_at_5_list"]) / n * 100

    print("\n" + "=" * 80)
    print("  BASELINE v1 OVERALL RETRIEVAL BENCHMARK")
    print("=" * 80)
    print(f"  Total Queries Evaluated     : {n}")
    print(f"  Hit Rate @ 1 (Rel >= 2)    : {(metrics['hr_at_1']/n)*100:5.1f}%  ({metrics['hr_at_1']}/{n})")
    print(f"  Hit Rate @ 3 (Rel >= 2)    : {(metrics['hr_at_3']/n)*100:5.1f}%  ({metrics['hr_at_3']}/{n})")
    print(f"  Hit Rate @ 5 (Rel >= 2)    : {(metrics['hr_at_5']/n)*100:5.1f}%  ({metrics['hr_at_5']}/{n})")
    print(f"  Ideal Match @ 1 (Rel == 3) : {(metrics['ideal_hr_at_1']/n)*100:5.1f}%  ({metrics['ideal_hr_at_1']}/{n})")
    print(f"  Ideal Match @ 3 (Rel == 3) : {(metrics['ideal_hr_at_3']/n)*100:5.1f}%  ({metrics['ideal_hr_at_3']}/{n})")
    print(f"  Mean Reciprocal Rank (MRR) : {mean_mrr:6.3f}")
    print(f"  Precision @ 3              : {mean_p_3:5.1f}%")
    print(f"  Precision @ 5              : {mean_p_5:5.1f}%")
    print(f"  NDCG @ 1                   : {mean_ndcg_1:6.3f}")
    print(f"  NDCG @ 3                   : {mean_ndcg_3:6.3f}")
    print(f"  NDCG @ 5                   : {mean_ndcg_5:6.3f}")

    print("\n" + "-" * 80)
    print("  CATEGORY BREAKDOWN (NDCG @ 3 & Hit Rate @ 3)")
    print("-" * 80)
    for cat, data in sorted(cat_stats.items()):
        hr3 = (data["hits_3"] / data["total"]) * 100
        avg_ndcg3 = sum(data["ndcg_3"]) / data["total"]
        print(f"  {cat:<25}: NDCG@3 = {avg_ndcg3:.3f} | HR@3 = {hr3:5.1f}% ({data['hits_3']}/{data['total']})")

    print("\n" + "-" * 80)
    print("  QUERY TYPE BREAKDOWN (NDCG @ 3 & Hit Rate @ 3)")
    print("-" * 80)
    for qtype, data in sorted(type_stats.items()):
        hr3 = (data["hits_3"] / data["total"]) * 100
        avg_ndcg3 = sum(data["ndcg_3"]) / data["total"]
        print(f"  {qtype:<25}: NDCG@3 = {avg_ndcg3:.3f} | HR@3 = {hr3:5.1f}% ({data['hits_3']}/{data['total']})")

    print("\n" + "=" * 80)
    print("  FAILURE & SUBOPTIMALITY ANALYSIS")
    print("=" * 80)
    if not suboptimal_queries:
        print("  All queries achieved optimal rankings (NDCG@3 >= 0.95 and Rank 1 hits).")
    else:
        print(f"  Identified {len(suboptimal_queries)} queries with sub-optimal ranking or lower density:\n")
        for sub in suboptimal_queries:
            print(f"  [{sub['id']}] '{sub['question']}' ({sub['category']}) - NDCG@3: {sub['ndcg_3']:.2f}")
            for r in sub["retrieved"]:
                grade_label = {3: "IDEAL (3)", 2: "RELEVANT (2)", 1: "PARTIAL (1)", 0: "IRRELEVANT (0)"}[r["grade"]]
                print(f"     Rank {r['rank']}: [{grade_label}] {r['name']} ({r['price']})")
            print()
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_baseline_evaluation(top_k=5)
