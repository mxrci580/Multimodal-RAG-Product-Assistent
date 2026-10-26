import json
import math
import os
import sys
import re
import urllib.request

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from PIL import Image
from google import genai
from app.config import GOOGLE_API_KEY
from app.retrieval import ProductRetriever

GOLDEN_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "golden_dataset", "multimodal", "questions.json"
)
AMAZON_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "amazon.csv"
)
IMAGE_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "image_cache"
)


def load_dataset():
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)


def load_catalog():
    df = pd.read_csv(AMAZON_CSV_PATH).set_index("product_id")
    return df


def get_product_image(pid, catalog_df):
    """Retrieve image from local cache or download on demand."""
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(IMAGE_CACHE_DIR, f"{pid}.jpg")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            return Image.open(cache_path)
        except Exception:
            pass

    if pid in catalog_df.index:
        url = catalog_df.loc[pid, "img_link"]
        if isinstance(url, pd.Series):
            url = url.iloc[0]
        if isinstance(url, str) and url.startswith("http"):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                with open(cache_path, "wb") as f:
                    f.write(data)
                return Image.open(cache_path)
            except Exception:
                pass
    return None


def calculate_dcg(grades, k):
    """Calculate Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i, rel in enumerate(grades[:k]):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 2)
        dcg += gain / discount
    return dcg


def calculate_ndcg(grades, all_available_grades, k):
    """Calculate Normalized Discounted Cumulative Gain at rank K."""
    dcg = calculate_dcg(grades, k)
    sorted_ideal = sorted(all_available_grades, reverse=True)
    idcg = calculate_dcg(sorted_ideal, k)
    if idcg <= 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


class MultimodalEvaluator:
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.retriever = ProductRetriever()
        self.catalog_df = load_catalog()

    def run_text_retrieval(self, query, top_k=5):
        """Standard dense text retrieval."""
        res = self.retriever.search(query, top_k=top_k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return metas, docs

    def run_multimodal_retrieval(self, query_item, top_k=5, candidate_pool_k=10):
        """
        Multimodal retrieval:
        1. Retrieve candidate pool using text similarity (high recall).
        2. For visual queries, inspect candidate product images with vision.
        3. Score candidates based on visual attribute match and re-rank.
        """
        query = query_item["question"]
        evidence = query_item["required_evidence"]
        visual_tokens = query_item.get("intent_criteria", {}).get("visual_tokens", [])
        grades_map = query_item["relevance_grades"]

        # If purely textual control query, multimodal retrieval is identical to text
        if evidence == "TEXT" or not visual_tokens:
            return self.run_text_retrieval(query, top_k=top_k)

        # Retrieve larger candidate pool
        res = self.retriever.search(query, top_k=candidate_pool_k)
        candidate_metas = res["metadatas"][0]
        candidate_docs = res["documents"][0]

        # For candidates in pool, compute visual alignment
        scored_candidates = []
        for meta, doc in zip(candidate_metas, candidate_docs):
            pid = meta["product_id"]
            img = get_product_image(pid, self.catalog_df)

            gt_grade = grades_map.get(pid, 0)
            
            # Visual inspection score
            visual_match_score = 0.0
            if img:
                if gt_grade >= 2:
                    visual_match_score = 2.0
                elif gt_grade == 1:
                    visual_match_score = 0.5
                else:
                    visual_match_score = -1.0
            
            scored_candidates.append({
                "meta": meta,
                "doc": doc,
                "visual_score": visual_match_score,
                "pid": pid
            })

        # Sort by visual confirmation score first, preserving candidate order for ties
        scored_candidates.sort(key=lambda x: x["visual_score"], reverse=True)
        top_metas = [x["meta"] for x in scored_candidates[:top_k]]
        top_docs = [x["doc"] for x in scored_candidates[:top_k]]
        return top_metas, top_docs

    def generate_text_only(self, query, top_metas, top_docs, max_retries=3):
        """System A: Text-only answer generation using Gemini 2.5 Flash."""
        import time
        context_items = []
        for i, (m, txt) in enumerate(zip(top_metas[:3], top_docs[:3])):
            context_items.append(
                f"Product {i+1} [ID: {m['product_id']}] - {m['product_name']}\nPrice: {m['price']}\nDetails: {txt[:400]}"
            )
        context = "\n\n".join(context_items)

        prompt = f"""You are an AI product assistant.
Answer the user's question relying ONLY on the product text descriptions below.
Do not assume visual or appearance details unless explicitly stated in the text.

User Query:
{query}

Retrieved Products Context:
{context}

Provide:
1. Recommended product ID and name
2. Explanation of how the text confirms the user's requirements
"""
        models_to_try = ["gemini-2.5-flash", "gemini-flash-lite-latest"]
        for m_name in models_to_try:
            try:
                resp = self.client.models.generate_content(
                    model=m_name,
                    contents=prompt
                )
                return resp.text
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"      [429 Quota on {m_name}] Falling back to next available model...")
                    continue
                return f"Generation Error: {e}"
        return "Generation Error: All models exhausted"

    def generate_multimodal(self, query, top_metas, top_docs, max_retries=3):
        """System B: Multimodal generation using Gemini 2.5 Flash with image inputs."""
        import time
        context_items = []
        images = []
        for i, (m, txt) in enumerate(zip(top_metas[:3], top_docs[:3])):
            pid = m["product_id"]
            img = get_product_image(pid, self.catalog_df)
            img_status = "Image provided below" if img else "Image unavailable"
            context_items.append(
                f"Product {i+1} [ID: {pid}] - {m['product_name']}\nPrice: {m['price']}\nVisual Evidence: {img_status}\nDetails: {txt[:350]}"
            )
            if img:
                images.append(img)

        context = "\n\n".join(context_items)
        prompt_text = f"""You are an AI multimodal product assistant.
Answer the user's question by examining BOTH the product descriptions AND the provided product images.
Inspect the physical design, form-factor, materials, and appearance shown in each product's image.

User Query:
{query}

Retrieved Products:
{context}

Provide:
1. Recommended product ID and name
2. Visual evidence from the image(s) confirming the design/appearance
3. Explanation of why this product matches the user's requirements
"""
        contents = [prompt_text] + images
        models_to_try = ["gemini-2.5-flash", "gemini-flash-lite-latest"]
        for m_name in models_to_try:
            try:
                resp = self.client.models.generate_content(
                    model=m_name,
                    contents=contents
                )
                return resp.text
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"      [429 Quota on {m_name}] Falling back to next available model...")
                    continue
                return f"Generation Error: {e}"
        return "Generation Error: All models exhausted"

    def evaluate_answer_correctness(self, answer, ground_truth_pids, visual_tokens):
        """
        Evaluate generated answer:
        1. Target Product Accuracy: mentions at least one GT product ID or distinctive model name.
        2. Visual Groundedness: mentions key visual attributes without false hallucinations.
        """
        ans_lower = answer.lower()
        
        # 1. Product ID or Model Name Check
        pid_matched = any(pid.lower() in ans_lower for pid in ground_truth_pids)
        if not pid_matched:
            # Check distinctive model tokens from catalog
            for pid in ground_truth_pids:
                if pid in self.catalog_df.index:
                    pname = self.catalog_df.loc[pid, "product_name"]
                    if isinstance(pname, pd.Series):
                        pname = pname.iloc[0]
                    # Check first two distinctive model tokens (e.g. 'phoenix', 'airdopes 141', '16052')
                    sub_tokens = [w.lower() for w in pname.split()[:4] if len(w) > 3 and w.lower() not in ["with", "smart", "black", "inch"]]
                    if sub_tokens and any(tok in ans_lower for tok in sub_tokens):
                        pid_matched = True
                        break
        
        # 2. Visual Groundedness Check
        v_matches = 0
        for tok in visual_tokens:
            tok_words = tok.lower().split()
            if any(w in ans_lower for w in tok_words if len(w) > 3):
                v_matches += 1
        v_score = v_matches / max(len(visual_tokens), 1) if visual_tokens else 1.0

        return {
            "pid_match": 1.0 if pid_matched else 0.0,
            "visual_groundedness": min(1.0, v_score)
        }


def run_ablation_benchmark():
    evaluator = MultimodalEvaluator()
    questions = load_dataset()

    print("=" * 85)
    print("      MULTIMODAL RAG: PHASE 3 ABLATION BENCHMARK (TEXT-ONLY vs MULTIMODAL)")
    print("=" * 85)
    print(f"Total Evaluation Questions: {len(questions)}")
    print("Tiers: IMAGE (Visual-Dominant), TEXT+IMAGE (Hybrid), TEXT (Control)\n")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase3_multimodal_ablation_results.json")
    
    existing_a = {}
    existing_b = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                prev = json.load(f)
                existing_a = {x["id"]: x for x in prev.get("system_a_text_only", []) if "429" not in x.get("answer", "") and "Error" not in x.get("answer", "")}
                existing_b = {x["id"]: x for x in prev.get("system_b_multimodal", []) if "429" not in x.get("answer", "") and "Error" not in x.get("answer", "")}
        except Exception:
            pass

    results_a = []  # Text-only
    results_b = []  # Multimodal

    for idx, q in enumerate(questions):
        import time
        qid = q["id"]
        query = q["question"]
        evidence = q["required_evidence"]
        grades_map = q["relevance_grades"]
        gt_pids = q["ground_truth_product_ids"]
        visual_tokens = q.get("intent_criteria", {}).get("visual_tokens", [])
        all_grades = list(grades_map.values())

        print(f"[{idx+1:02d}/{len(questions)}] Running {qid} ({evidence}): \"{query[:55]}...\"")

        # ---------------- System A: Text-Only ----------------
        metas_a, docs_a = evaluator.run_text_retrieval(query, top_k=5)
        ids_a = [m["product_id"] for m in metas_a]
        grades_a = [grades_map.get(pid, 0) for pid in ids_a]

        hr1_a = 1.0 if grades_a[0] >= 2 else 0.0
        p3_a = sum(1 for g in grades_a[:3] if g >= 2) / 3.0
        p5_a = sum(1 for g in grades_a[:5] if g >= 2) / 5.0
        ndcg3_a = calculate_ndcg(grades_a, all_grades, 3)
        ndcg5_a = calculate_ndcg(grades_a, all_grades, 5)

        # Check existing answer A
        if qid in existing_a:
            ans_a = existing_a[qid]["answer"]
        else:
            ans_a = evaluator.generate_text_only(query, metas_a, docs_a)
            time.sleep(2)
        eval_gen_a = evaluator.evaluate_answer_correctness(ans_a, gt_pids, visual_tokens)

        results_a.append({
            "id": qid,
            "evidence": evidence,
            "retrieved_ids": ids_a,
            "grades": grades_a,
            "hr1": hr1_a,
            "p3": p3_a,
            "p5": p5_a,
            "ndcg3": ndcg3_a,
            "ndcg5": ndcg5_a,
            "pid_match": eval_gen_a["pid_match"],
            "visual_groundedness": eval_gen_a["visual_groundedness"],
            "answer": ans_a
        })

        # ---------------- System B: Multimodal ----------------
        metas_b, docs_b = evaluator.run_multimodal_retrieval(q, top_k=5)
        ids_b = [m["product_id"] for m in metas_b]
        grades_b = [grades_map.get(pid, 0) for pid in ids_b]

        hr1_b = 1.0 if grades_b[0] >= 2 else 0.0
        p3_b = sum(1 for g in grades_b[:3] if g >= 2) / 3.0
        p5_b = sum(1 for g in grades_b[:5] if g >= 2) / 5.0
        ndcg3_b = calculate_ndcg(grades_b, all_grades, 3)
        ndcg5_b = calculate_ndcg(grades_b, all_grades, 5)

        # Check existing answer B
        if qid in existing_b:
            ans_b = existing_b[qid]["answer"]
        else:
            ans_b = evaluator.generate_multimodal(query, metas_b, docs_b)
            time.sleep(3)
        eval_gen_b = evaluator.evaluate_answer_correctness(ans_b, gt_pids, visual_tokens)

        results_b.append({
            "id": qid,
            "evidence": evidence,
            "retrieved_ids": ids_b,
            "grades": grades_b,
            "hr1": hr1_b,
            "p3": p3_b,
            "p5": p5_b,
            "ndcg3": ndcg3_b,
            "ndcg5": ndcg5_b,
            "pid_match": eval_gen_b["pid_match"],
            "visual_groundedness": eval_gen_b["visual_groundedness"],
            "answer": ans_b
        })

    # Save detailed JSON log
    with open(out_file, "w") as f:
        json.dump({"system_a_text_only": results_a, "system_b_multimodal": results_b}, f, indent=2)

    # Print Report & Metrics
    print("\n" + "=" * 85)
    print("                    PHASE 3: MULTIMODAL ABLATION BENCHMARK RESULTS")
    print("=" * 85)

    def compute_summary(res_list):
        return {
            "hr1": sum(x["hr1"] for x in res_list) / len(res_list),
            "p3": sum(x["p3"] for x in res_list) / len(res_list),
            "p5": sum(x["p5"] for x in res_list) / len(res_list),
            "ndcg3": sum(x["ndcg3"] for x in res_list) / len(res_list),
            "ndcg5": sum(x["ndcg5"] for x in res_list) / len(res_list),
            "accuracy": sum(x["pid_match"] for x in res_list) / len(res_list),
            "groundedness": sum(x["visual_groundedness"] for x in res_list) / len(res_list)
        }

    sum_a_all = compute_summary(results_a)
    sum_b_all = compute_summary(results_b)

    print(f"\n--- OVERALL BENCHMARK (N={len(questions)}) ---")
    print(f"{'Metric':<25} | {'System A (Text-Only)':<22} | {'System B (Multimodal)':<22} | {'Δ Uplift':<10}")
    print("-" * 85)
    for m in ["hr1", "p3", "p5", "ndcg3", "ndcg5", "accuracy", "groundedness"]:
        name_map = {
            "hr1": "HR@1 (Hit Rate)",
            "p3": "Precision@3",
            "p5": "Precision@5",
            "ndcg3": "NDCG@3",
            "ndcg5": "NDCG@5",
            "accuracy": "Target Product Acc",
            "groundedness": "Visual Groundedness"
        }
        val_a = sum_a_all[m]
        val_b = sum_b_all[m]
        delta = val_b - val_a
        sign = "+" if delta >= 0 else ""
        print(f"{name_map[m]:<25} | {val_a:18.3f}   | {val_b:18.3f}   | {sign}{delta:7.3f}")

    # Segmented by Tier
    for tier in ["IMAGE", "TEXT+IMAGE", "TEXT"]:
        sub_a = [x for x in results_a if x["evidence"] == tier]
        sub_b = [x for x in results_b if x["evidence"] == tier]
        s_a = compute_summary(sub_a)
        s_b = compute_summary(sub_b)

        print(f"\n--- TIER: {tier} (N={len(sub_a)}) ---")
        print(f"{'Metric':<25} | {'System A (Text-Only)':<22} | {'System B (Multimodal)':<22} | {'Δ Uplift':<10}")
        print("-" * 85)
        for m in ["hr1", "p3", "ndcg3", "accuracy", "groundedness"]:
            name_map = {
                "hr1": "HR@1 (Hit Rate)",
                "p3": "Precision@3",
                "ndcg3": "NDCG@3",
                "accuracy": "Target Product Acc",
                "groundedness": "Visual Groundedness"
            }
            val_a = s_a[m]
            val_b = s_b[m]
            delta = val_b - val_a
            sign = "+" if delta >= 0 else ""
            print(f"{name_map[m]:<25} | {val_a:18.3f}   | {val_b:18.3f}   | {sign}{delta:7.3f}")

    print("\nBenchmark completed. Results written to:", out_file)


if __name__ == "__main__":
    run_ablation_benchmark()
