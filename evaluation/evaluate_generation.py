"""
Phase 4C: Generation Evaluation Framework
Evaluates answers produced from Phase 4B retrieval outputs:
  1. System A: Dense Baseline + Text-Only LLM
  2. System B: Cross-Encoder + Text-Only LLM
  3. System C: Multimodal Retrieval + Gemini Vision

Metrics:
  1. Faithfulness / Groundedness (Claim-level verification)
  2. Answer Correctness (Target product match against golden graded relevance)
  3. Multi-Constraint Completeness (Fulfillment of budget, brand, spec, visual constraints)
  4. Hallucination Rate (Fabricated specs, prices, or visual features)
  5. Evidence Attribution Accuracy (Associating claims with correct evidence source)

Segmented Breakdowns:
  - Evidence Tier: IMAGE (15), TEXT+IMAGE (25), TEXT (10)
  - Difficulty Tier: Easy (15), Medium (15), Hard (12), Adversarial (8)
"""

import csv
import json
import os
import re
import sys
import time
from typing import Dict, List, Any, Optional
from PIL import Image
from dotenv import load_dotenv
from google import genai

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset", "questions_v2_50.json")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "amazon.csv")
IMAGE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "image_cache")
RETRIEVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "phase4b_retrieval_benchmark_results.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "results", "phase4c_generation_checkpoint.json")
FINAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "phase4c_generation_evaluation_results.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "experiment_4c_generation_report.md")

API_KEY = os.getenv("GOOGLE_API_KEY")


def parse_price(p_str) -> float:
    if not p_str:
        return 0.0
    val = re.sub(r"[^\d.]", "", str(p_str).replace(",", ""))
    return float(val) if val else 0.0


def load_catalog() -> Dict[str, Dict[str, Any]]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return {r["product_id"]: r for r in csv.DictReader(f)}


class GenerationHarness:
    def __init__(self):
        if not API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment!")
        self.client = genai.Client(api_key=API_KEY)
        self.catalog = load_catalog()
        self.primary_model = "gemini-2.5-flash"
        self.fallback_model = "gemini-flash-lite-latest"

    def generate_with_fallback(self, contents, is_json=True):
        models = [self.primary_model, self.fallback_model]
        for m in models:
            for attempt in range(2):
                try:
                    resp = self.client.models.generate_content(
                        model=m,
                        contents=contents
                    )
                    txt = resp.text.strip()
                    if is_json:
                        # Clean JSON code fence if present
                        if "```json" in txt:
                            txt = txt.split("```json")[1].split("```")[0].strip()
                        elif "```" in txt:
                            txt = txt.split("```")[1].split("```")[0].strip()
                        return json.loads(txt), m
                    return txt, m
                except Exception as e:
                    time.sleep(1.5)
        # Fallback default empty structure
        return {
            "recommended_product_id": "",
            "recommended_product_name": "",
            "stated_price": 0.0,
            "rationale": "Generation failed due to quota/connection.",
            "factual_claims": []
        }, "failed"

    def build_text_context(self, pids: List[str]) -> str:
        items = []
        for i, pid in enumerate(pids[:3], 1):
            prod = self.catalog.get(pid, {})
            name = prod.get("product_name", "Unknown Product")
            price = prod.get("discounted_price", "N/A")
            rating = prod.get("rating", "N/A")
            about = prod.get("about_product", "")[:400]
            items.append(
                f"Candidate {i} [ID: {pid}]:\n"
                f"Title: {name}\n"
                f"Price: {price} | Rating: {rating}\n"
                f"Description: {about}"
            )
        return "\n\n".join(items)

    def run_text_generation(self, query: str, pids: List[str]) -> Dict[str, Any]:
        context = self.build_text_context(pids)
        prompt = f"""You are an expert e-commerce product recommendation assistant.
User Query: "{query}"

Retrieved Candidate Products (from Catalog Text):
{context}

Instructions:
1. Select the single best product from the candidates that satisfies the user's requirements.
2. In your response, provide your recommendation and break down your answer into atomic factual claims.
3. For each claim, attribute whether your evidence came from "catalog_text" or "general_knowledge".

You MUST reply ONLY with valid JSON in this exact structure:
{{
  "recommended_product_id": "Exact Product ID (e.g. B0...)",
  "recommended_product_name": "Title of product",
  "stated_price": 1299.0,
  "rationale": "Clear explanation of why this product fits the criteria.",
  "factual_claims": [
    {{"claim": "Claim 1 stating a feature or constraint", "evidence_source": "catalog_text"}},
    {{"claim": "Claim 2 stating price or spec", "evidence_source": "catalog_text"}}
  ]
}}"""
        res, model_used = self.generate_with_fallback(prompt, is_json=True)
        res["model_used"] = model_used
        return res

    def run_multimodal_generation(self, query: str, pids: List[str]) -> Dict[str, Any]:
        context = self.build_text_context(pids)
        images = []
        for pid in pids[:3]:
            img_path = os.path.join(IMAGE_CACHE_DIR, f"{pid}.jpg")
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                try:
                    images.append(Image.open(img_path))
                except Exception:
                    pass

        prompt = f"""You are an expert multimodal e-commerce product recommendation assistant.
User Query: "{query}"

Retrieved Candidate Products (Text Metadata):
{context}

You are also provided with the product images of the candidate items. Inspect both the text descriptions AND the physical image pixels (dial shapes, hardware buttons, wire braiding, stem vs pebble form factors, transparent walls, ports, etc.).

Instructions:
1. Select the single best product from the candidates that satisfies all constraints (budget, specs, and physical visual appearance).
2. Break down your answer into atomic factual claims.
3. For each claim, attribute whether your evidence came from "product_image" (observable in photo) or "catalog_text" (stated in text specs).

You MUST reply ONLY with valid JSON in this exact structure:
{{
  "recommended_product_id": "Exact Product ID (e.g. B0...)",
  "recommended_product_name": "Title of product",
  "stated_price": 1299.0,
  "rationale": "Clear explanation citing observable physical features from the image and specifications from the text.",
  "factual_claims": [
    {{"claim": "Physical appearance claim (e.g. circular dial / nylon braiding)", "evidence_source": "product_image"}},
    {{"claim": "Technical spec or price claim (e.g. Bluetooth calling / price under 2000)", "evidence_source": "catalog_text"}}
  ]
}}"""
        contents = images + [prompt]
        res, model_used = self.generate_with_fallback(contents, is_json=True)
        res["model_used"] = model_used
        return res


def evaluate_response(
    query_item: Dict[str, Any],
    response: Dict[str, Any],
    catalog: Dict[str, Any],
    is_multimodal_system: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a generated response using hybrid programmatic, visual, and constraint checks:
      1. Answer Correctness (Target match against golden relevance)
      2. Multi-Constraint Completeness (Checking all constraint facets)
      3. Claim-Level Faithfulness & Hallucination Rate
      4. Evidence Attribution Accuracy
    """
    recommended_pid = response.get("recommended_product_id", "").strip()
    stated_price = float(response.get("stated_price", 0.0) or 0.0)
    claims = response.get("factual_claims", [])
    if not isinstance(claims, list):
        claims = []

    rel_map = {rp["product_id"]: rp["relevance"] for rp in query_item.get("relevant_products", [])}
    constraints = query_item.get("constraints", {})
    required_evidence = query_item.get("required_evidence", {})

    # 1. Answer Correctness
    target_grade = rel_map.get(recommended_pid, 0)
    if target_grade == 3:
        correctness = 1.0
    elif target_grade == 2:
        correctness = 0.67
    elif target_grade == 1:
        correctness = 0.33
    else:
        correctness = 0.0

    # 2. Multi-Constraint Completeness
    satisfied_facets = 0
    total_facets = 0

    prod_info = catalog.get(recommended_pid, {})
    prod_price = parse_price(prod_info.get("discounted_price", 0))
    prod_name = prod_info.get("product_name", "").lower()
    prod_about = prod_info.get("about_product", "").lower()

    # Budget constraint
    price_max = constraints.get("price_max")
    if price_max:
        total_facets += 1
        if prod_price > 0 and prod_price <= price_max:
            satisfied_facets += 1

    # Brand constraint
    brand = constraints.get("brand")
    if brand:
        total_facets += 1
        if brand.lower() in prod_name or brand.lower() in prod_about:
            satisfied_facets += 1

    # Category / Form-factor constraint
    cat = constraints.get("category")
    if cat:
        total_facets += 1
        if target_grade >= 2:
            satisfied_facets += 1
        elif cat.lower() in prod_info.get("category", "").lower() or cat.lower() in prod_name:
            satisfied_facets += 1

    # Specs constraint
    specs = constraints.get("specs", [])
    for sp in specs:
        total_facets += 1
        # Check if spec is satisfied in catalog text or high grade
        if target_grade >= 2 or any(token in prod_name or token in prod_about for token in sp.lower().split()):
            satisfied_facets += 1

    # Visual constraint
    visual_reqs = constraints.get("visual", [])
    for vr in visual_reqs:
        total_facets += 1
        # Visual constraint is truly satisfied if target product is Grade >= 2
        if target_grade >= 2:
            satisfied_facets += 1

    completeness = (satisfied_facets / max(total_facets, 1))

    # 3. Claim-Level Faithfulness & Evidence Attribution
    supported_claims = 0
    correctly_attributed_claims = 0
    total_claims = len(claims)

    visual_keywords = [
        "dial", "screen", "shape", "round", "circular", "square", "rectangular",
        "stem", "pebble", "bezel", "braided", "nylon", "jacket", "glass", "transparent",
        "led", "light", "blue", "ring", "nozzle", "slot", "lever", "grill", "ridge",
        "knob", "rotary", "touch", "panel", "shutter", "clip", "clamp", "thumb", "ergonomic",
        "textured", "leather", "color", "black", "white", "metallic", "metal"
    ]

    for c in claims:
        claim_text = c.get("claim", "").lower()
        source = c.get("evidence_source", "").lower()

        # Is this claim about visual appearance?
        is_visual_claim = any(vk in claim_text for vk in visual_keywords)

        # Evidence attribution check:
        # Visual claims should come from product_image (or catalog text only if multimodal)
        # Spec/price claims should come from catalog_text
        if is_visual_claim:
            if is_multimodal_system and "image" in source:
                correctly_attributed_claims += 1
            elif not is_multimodal_system and "text" in source:
                # Text model attributing visual claims to text is considered flawed if text lacks the detail
                if target_grade >= 2 and any(vk in prod_about for vk in visual_keywords if vk in claim_text):
                    correctly_attributed_claims += 1
            elif "image" in source:
                correctly_attributed_claims += 1
        else:
            if "text" in source or "catalog" in source or "metadata" in source:
                correctly_attributed_claims += 1

        # Grounding / Faithfulness check:
        # A claim is supported if:
        # a) It's a visual claim and the recommended product actually has that visual trait (target_grade >= 2 and is_multimodal_system or verified in text)
        # b) It's a spec/price claim and matches catalog data
        is_supported = True
        if is_visual_claim:
            # If target product is Grade 0 (trap), claiming it has the desired visual trait is a hallucination!
            if target_grade == 0 and any(vr.lower() in claim_text for vr in visual_reqs):
                is_supported = False
            elif not is_multimodal_system and query_item.get("evidence_type") == "IMAGE":
                # Text-only model hallucinating visual confirmation when text didn't contain it
                is_supported = False
        else:
            # Check for price hallucination
            price_mentions = re.findall(r'₹?\s*(\d[\d,]+)', claim_text)
            if price_mentions and prod_price > 0:
                p_val = float(price_mentions[0].replace(",", ""))
                if abs(p_val - prod_price) > 50:
                    is_supported = False

        if is_supported:
            supported_claims += 1

    if total_claims > 0:
        faithfulness = supported_claims / total_claims
        evidence_attribution = correctly_attributed_claims / total_claims
    else:
        faithfulness = 0.5
        evidence_attribution = 0.5

    hallucination_rate = max(0.0, 1.0 - faithfulness)

    return {
        "recommended_product_id": recommended_pid,
        "stated_price": stated_price,
        "actual_catalog_price": prod_price,
        "target_grade": target_grade,
        "correctness": round(correctness, 4),
        "completeness": round(completeness, 4),
        "faithfulness": round(faithfulness, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "evidence_attribution": round(evidence_attribution, 4),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "rationale": response.get("rationale", "")
    }


def main():
    print("=" * 80)
    print("PHASE 4C: GENERATION EVALUATION HARNESS")
    print("=" * 80)

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[✓] Loaded {len(questions)} frozen queries from {BENCHMARK_PATH}")

    with open(RETRIEVAL_RESULTS_PATH, encoding="utf-8") as f:
        retrieval_data = json.load(f)
    print(f"[✓] Loaded Phase 4B retrieval candidates from {RETRIEVAL_RESULTS_PATH}")

    catalog = load_catalog()
    harness = GenerationHarness()

    # Load checkpoint if exists
    checkpoint = {}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"[✓] Resumed from checkpoint: {len(checkpoint.get('Dense-Gen', {}))} Dense, {len(checkpoint.get('Cross-Encoder-Gen', {}))} CE, {len(checkpoint.get('Multimodal-Gen', {}))} MM")
        except Exception:
            checkpoint = {}

    systems_to_run = [
        ("Dense-Gen", "Dense Baseline", False),
        ("Cross-Encoder-Gen", "Cross-Encoder", False),
        ("Multimodal-Gen", "Multimodal", True)
    ]

    for sys_key, retr_key, is_mm in systems_to_run:
        if sys_key not in checkpoint:
            checkpoint[sys_key] = {}

        print(f"\n--- Running Generation & Evaluation for: {sys_key} ---")
        recs = retrieval_data[retr_key]["records"]
        retrieved_map = {r["id"]: r["retrieved_pids"] for r in recs}

        for idx, q in enumerate(questions, 1):
            qid = q["id"]
            if qid in checkpoint[sys_key]:
                continue

            query = q["question"]
            pids = retrieved_map.get(qid, [])

            if is_mm:
                raw_resp = harness.run_multimodal_generation(query, pids)
            else:
                raw_resp = harness.run_text_generation(query, pids)

            eval_res = evaluate_response(q, raw_resp, catalog, is_multimodal_system=is_mm)

            checkpoint[sys_key][qid] = {
                "id": qid,
                "evidence_type": q["evidence_type"],
                "difficulty": q["difficulty"],
                "intent_type": q["intent_type"],
                "raw_response": raw_resp,
                "evaluation": eval_res
            }

            # Periodic save
            if idx % 5 == 0 or idx == len(questions):
                with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, indent=2)
                print(f"  [{idx}/{len(questions)}] Saved checkpoint for {sys_key} (QID: {qid}, Correctness: {eval_res['correctness']:.2f}, Faithfulness: {eval_res['faithfulness']:.2f})")

            time.sleep(0.5)

    # Compute Aggregate Metrics
    print("\n" + "=" * 80)
    print("COMPUTING FINAL GENERATION EVALUATION METRICS")
    print("=" * 80)

    final_results = {}

    def agg_metrics(eval_list):
        if not eval_list:
            return {"correctness": 0.0, "faithfulness": 0.0, "completeness": 0.0, "hallucination_rate": 0.0, "evidence_attribution": 0.0, "count": 0}
        n = len(eval_list)
        return {
            "correctness": round(sum(x["correctness"] for x in eval_list) / n, 4),
            "faithfulness": round(sum(x["faithfulness"] for x in eval_list) / n, 4),
            "completeness": round(sum(x["completeness"] for x in eval_list) / n, 4),
            "hallucination_rate": round(sum(x["hallucination_rate"] for x in eval_list) / n, 4),
            "evidence_attribution": round(sum(x["evidence_attribution"] for x in eval_list) / n, 4),
            "count": n
        }

    for sys_key, _, _ in systems_to_run:
        q_records = list(checkpoint[sys_key].values())
        eval_items = [r["evaluation"] for r in q_records]

        overall = agg_metrics(eval_items)

        by_evidence = {}
        for ev in ["IMAGE", "TEXT+IMAGE", "TEXT"]:
            sub_evals = [r["evaluation"] for r in q_records if r["evidence_type"] == ev]
            by_evidence[ev] = agg_metrics(sub_evals)

        by_difficulty = {}
        for diff in ["easy", "medium", "hard", "adversarial"]:
            sub_evals = [r["evaluation"] for r in q_records if r["difficulty"] == diff]
            by_difficulty[diff] = agg_metrics(sub_evals)

        final_results[sys_key] = {
            "overall": overall,
            "by_evidence": by_evidence,
            "by_difficulty": by_difficulty,
            "records": q_records
        }

    with open(FINAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"[✓] Final results saved to: {FINAL_RESULTS_PATH}")

    # Print Summary Table
    print("\n" + "=" * 90)
    print("PHASE 4C GENERATION BENCHMARK RESULTS (N = 50)")
    print("=" * 90)
    print(f"{'System':<20} | {'Correctness':<11} | {'Faithfulness':<12} | {'Completeness':<12} | {'Hallucination':<13} | {'Attribution':<11}")
    print("-" * 90)
    for sys_key in ["Dense-Gen", "Cross-Encoder-Gen", "Multimodal-Gen"]:
        ov = final_results[sys_key]["overall"]
        print(f"{sys_key:<20} | {ov['correctness']*100:10.1f}% | {ov['faithfulness']*100:11.1f}% | {ov['completeness']*100:11.1f}% | {ov['hallucination_rate']*100:12.1f}% | {ov['evidence_attribution']*100:10.1f}%")

    print("\n" + "=" * 90)
    print("EVIDENCE BREAKDOWN: Answer Correctness / Faithfulness")
    print("=" * 90)
    print(f"{'System':<20} | {'IMAGE (15)':<20} | {'TEXT+IMAGE (25)':<20} | {'TEXT (10)':<20}")
    print("-" * 90)
    for sys_key in ["Dense-Gen", "Cross-Encoder-Gen", "Multimodal-Gen"]:
        be = final_results[sys_key]["by_evidence"]
        print(f"{sys_key:<20} | {be['IMAGE']['correctness']*100:4.1f}% / {be['IMAGE']['faithfulness']*100:4.1f}%   | {be['TEXT+IMAGE']['correctness']*100:4.1f}% / {be['TEXT+IMAGE']['faithfulness']*100:4.1f}%   | {be['TEXT']['correctness']*100:4.1f}% / {be['TEXT']['faithfulness']*100:4.1f}%")

    print("\n" + "=" * 90)
    print("DIFFICULTY BREAKDOWN: Answer Correctness")
    print("=" * 90)
    print(f"{'System':<20} | {'Easy (15)':<12} | {'Medium (15)':<12} | {'Hard (12)':<12} | {'Adversarial (8)':<16}")
    print("-" * 90)
    for sys_key in ["Dense-Gen", "Cross-Encoder-Gen", "Multimodal-Gen"]:
        bd = final_results[sys_key]["by_difficulty"]
        print(f"{sys_key:<20} | {bd['easy']['correctness']*100:11.1f}% | {bd['medium']['correctness']*100:11.1f}% | {bd['hard']['correctness']*100:11.1f}% | {bd['adversarial']['correctness']*100:15.1f}%")


if __name__ == "__main__":
    main()
