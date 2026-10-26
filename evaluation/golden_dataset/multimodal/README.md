# Multimodal Golden Evaluation Dataset (Phase 3 Benchmark)

This directory contains the **Multimodal Golden Evaluation Benchmark** designed specifically to measure **Multimodal Uplift**:

$$\text{Multimodal Uplift} = \text{Score}_{\text{multimodal}} - \text{Score}_{\text{text-only}}$$

The dataset isolates two key dimensions:
1. **Multimodal Retrieval Uplift**: Does visual feature conditioning / image analysis enhance candidate retrieval precision over text-only semantic search?
2. **Multimodal Generation Uplift**: Does visual context enable the Vision-Language Model (VLM) to answer appearance-critical, aesthetic, and physical form-factor questions accurately without hallucination?

---

## 1. Dataset Breakdown & Evidence Tiers

The benchmark contains **25 curated, catalog-grounded queries** divided into three experimental tiers:

| Tier | Required Evidence | Count | Purpose / Hypothesis |
| :--- | :--- | :---: | :--- |
| **Tier 1: Visual-Dominant** | `IMAGE` | **10** | **High Uplift Expected**: The distinguishing factor is primarily or exclusively visual (e.g. transparent glass kettle body with blue LED, circular round dial vs rectangular dial, stem-style vs pebble earbuds, interior digital LED battery readout, circumaural over-ear cups, braided dual-tone cord, ergonomic thumb rest, OTT Netflix/Prime remote buttons, ventilated open-frame aluminum laptop riser). |
| **Tier 2: Hybrid Multimodal** | `TEXT+IMAGE` | **10** | **Moderate-to-High Uplift Expected**: Combines a structured textual constraint (price ceiling $\le \text{₹2,000}$, brand, interface standard like Type-C) with a physical appearance requirement (e.g., black over-ear headphones under ₹2,000, round dial smartwatch under ₹2,000 with calling, frameless 3-side bezel-less 32-inch TV under ₹15,000, faux-leather back Redmi A1 under ₹7,000). |
| **Tier 3: Control Group** | `TEXT` | **5** | **Zero/Parity Uplift Expected**: Purely textual/technical catalog queries (e.g., 500W mixer grinder with 3 jars, 10000mAh dual output power bank, 1.2m lightning cable, 32-inch Android Dolby TV, 1000W dry iron). Verifies whether multimodal pipelines maintain high factual precision without introducing visual noise or hallucinations. |

---

## 2. Graded Relevance Methodology (0 to 3 Scale)

Relevance is evaluated on an explicit graded scale consistent with TREC and e-commerce information retrieval best practices:

- **Grade 3 (Ideal / Highly Relevant)**: Fulfills both textual constraints and the target visual form-factor/attribute completely.
- **Grade 2 (Relevant / Close Substitute)**: Satisfies primary category and visual criteria, with minor secondary variance (e.g. price slightly over bound).
- **Grade 1 (Partially Relevant)**: Category matches or adjacent form-factor, but misses key visual differentiators (e.g. on-ear instead of over-ear, smooth cord instead of braided).
- **Grade 0 (Irrelevant)**: Violates core category, form-factor, or visual criteria (e.g. opaque steel kettle when transparent glass was requested, rectangular dial when circular dial was requested).

---

## 3. Schema Definition (`questions.json`)

Each entry in `questions.json` adheres to the following schema:

```json
{
  "id": "MM01",
  "question": "Which electric kettle has a transparent glass body with blue LED illumination during boiling?",
  "category": "Kitchen & Home Appliances",
  "query_type": "visual_attribute",
  "required_evidence": "IMAGE",
  "evidence_sources": ["IMAGE"],
  "visual_attribute": "transparent_borosilicate_glass_blue_led",
  "description": "Tests identification of transparent borosilicate glass body with glowing blue LED heating ring versus opaque stainless steel bodies.",
  "ground_truth_product_ids": ["B07YR26BJ3", "B08HLC7Z3G"],
  "ground_truth_answer": "The KENT 16052 Elegant Electric Glass Kettle (B07YR26BJ3) and Inalsa Prism Inox (B08HLC7Z3G) feature clear borosilicate glass bodies with internal blue LED illumination glowing during operation, unlike opaque stainless steel kettles.",
  "target_image_url": "https://m.media-amazon.com/images/I/4150hW2kHJL._SX300_SY300_QL70_FMwebp_.jpg",
  "intent_criteria": {
    "target_subcategory": "Kettles",
    "required_keywords": ["kettle", "glass", "led"],
    "visual_tokens": ["transparent", "glass body", "blue led ring", "water level visibility"],
    "max_price": null
  },
  "relevance_grades": {
    "B07YR26BJ3": 3,
    "B08HLC7Z3G": 3,
    "B07WMS7TWB": 0,
    "B01MQZ7J8K": 0,
    "B00NW4UWN6": 0
  }
}
```

---

## 4. Benchmark Execution Workflow

1. **System A (Text-Only Pipeline)**:
   $$\text{User Query} \xrightarrow{\text{Dense/BM25/Rerank}} \text{Top-K Products} \xrightarrow{\text{Text Prompt}} \text{LLM Generation}$$
2. **System B (Multimodal Pipeline)**:
   $$\text{User Query} + \text{Visual Conditioning} \xrightarrow{\text{Multimodal Retrieval}} \text{Top-K Products} \xrightarrow{\text{Image + Text Prompt}} \text{VLM Generation}$$
3. **Metrics Computed**:
   - Retrieval: Precision@3, Precision@5, NDCG@3, NDCG@5, Hit Rate (HR@1).
   - Generation: Visual Attribute Accuracy (0–100%), Visual Groundedness (0–100%), Answer Correctness (0–100%).
   - Multimodal Uplift ($\Delta$): $\text{Metric}_B - \text{Metric}_A$ segmented across the 3 evidence tiers.
