# Multimodal RAG: Retriever Baseline v1 Benchmark Report

**Date:** October 2026  
**Dataset:** [`data/amazon.csv`](../data/amazon.csv) (1,465 products)  
**Vector Store:** ChromaDB (`product_recommendations`)  
**Embedding Model:** `text-embedding-004` (Google GenAI)  
**Evaluation Set:** 25 Curated Queries with Graded Relevance (0–3 scale)

---

## 1. Graded Relevance Definition

| Grade | Label | Description | Example |
| :---: | :--- | :--- | :--- |
| **3** | **Ideal / Highly Relevant** | Satisfies category, specific features, brand, and constraints completely. | boAt Airdopes 141 for *"best bluetooth earbuds under 1500"* |
| **2** | **Relevant** | Strong alternative/substitute; satisfies core category and major requirements. | Boult AirBass PowerBuds for *"best bluetooth earbuds under 1500"* |
| **1** | **Partially Relevant** | Adjacent accessory, different form-factor (e.g., on-ear vs over-ear), or missing minor constraint. | boAt Rockerz 400 On-Ear for *"over ear bluetooth headphones"* |
| **0** | **Irrelevant** | Wrong category entirely, incompatible device, or severe price violation. | USB Cable for *"wireless earbuds"* |

---

## 2. Baseline v1 Benchmark Results ($K=5$)

| Metric | Baseline v1 Score | Description |
| :--- | :---: | :--- |
| **Hit Rate @ 1 (Rel $\ge$ 2)** | **100.0%** (25/25) | First retrieved item is at least relevant for every query. |
| **Hit Rate @ 3 (Rel $\ge$ 2)** | **100.0%** (25/25) | Relevant item retrieved within Top 3 for every query. |
| **Ideal Match @ 1 (Rel == 3)** | **100.0%** (25/25) | First retrieved item is an Ideal (Grade 3) match. |
| **Mean Reciprocal Rank (MRR)** | **1.000** | First relevant hit consistently lands at Rank 1. |
| **Precision @ 3** | **97.3%** | 97.3% of all Top 3 retrieved slots across queries are relevant ($\ge 2$). |
| **Precision @ 5** | **95.2%** | 95.2% of all Top 5 retrieved slots across queries are relevant ($\ge 2$). |
| **NDCG @ 1** | **1.000** | Normalized Discounted Cumulative Gain at Rank 1. |
| **NDCG @ 3** | **0.988** | Ranking quality taking into account graded gain at Top 3. |
| **NDCG @ 5** | **0.986** | Ranking quality taking into account graded gain at Top 5. |

---

## 3. Category Breakdown ($K=3$)

| Category | Queries | NDCG @ 3 | Hit Rate @ 3 (Rel $\ge$ 2) | Ideal Match @ 3 (Rel == 3) |
| :--- | :---: | :---: | :---: | :---: |
| **Audio & Earbuds** | 5 | **0.939** | 100.0% | 100.0% |
| **Cables & Accessories** | 4 | **1.000** | 100.0% | 100.0% |
| **Computers & Peripherals** | 3 | **1.000** | 100.0% | 100.0% |
| **Home & Kitchen** | 4 | **1.000** | 100.0% | 100.0% |
| **Smart TVs** | 3 | **1.000** | 100.0% | 100.0% |
| **Smartphones** | 3 | **1.000** | 100.0% | 100.0% |
| **Smartwatches** | 3 | **1.000** | 100.0% | 100.0% |

---

## 4. Query Intent Breakdown ($K=3$)

| Query Type | Queries | NDCG @ 3 | Hit Rate @ 3 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Exact Product** | 1 | **1.000** | 100.0% | Direct model lookup |
| **Brand + Feature** | 4 | **1.000** | 100.0% | Brand combined with specific specs |
| **Feature Search** | 6 | **1.000** | 100.0% | Unbranded technical specifications |
| **Intent / Use Case** | 3 | **1.000** | 100.0% | Functional purpose queries |
| **Broad Category** | 1 | **1.000** | 100.0% | General category navigation |
| **Brand + Category** | 7 | **0.980** | 100.0% | Slight form-factor variance in audio |
| **Price Constraint** | 3 | **0.944** | 100.0% | Filtered price boundaries |

---

## 5. Failure & Suboptimality Analysis

Across the 25 benchmark queries, 2 queries exhibited sub-optimal ordering:

### 1. `[Q02]` *"best bluetooth earbuds under 1500"* (NDCG@3 = 0.83)
- **Retrieved:**
  - Rank 1: `boAt Airdopes 141` [Grade 3 — Ideal, ₹1,299]
  - Rank 2: `Boult Audio AirBass PowerBuds` [Grade 2 — Relevant, ₹1,499]
  - Rank 3: `ZEBRONICS Zeb-Sound Bomb N1` [Grade 3 — Ideal, ₹999]
- **Diagnostic:** A Grade 2 product was ranked ahead of a Grade 3 product. Pure dense semantic similarity favored Boult slightly over Zebronics due to the keyword overlap with *"PowerBuds"*.
- **Engineering Opportunity:** Metadata reranking / re-scoring based on customer ratings or explicit price-tier optimization will promote ideal lower-priced options to Rank 2.

### 2. `[Q05]` *"boAt Rockerz over ear bluetooth headphones"* (NDCG@3 = 0.86)
- **Retrieved:**
  - Rank 1: `boAt Rockerz 550 Over Ear` [Grade 3 — Ideal, ₹1,799]
  - Rank 2: `boAt Rockerz 400 On-Ear` [Grade 1 — Partial match, ₹1,399]
  - Rank 3: `boAt Rockerz 450 On-Ear / Over-Ear hybrid` [Grade 3 — Ideal, ₹1,220]
- **Diagnostic:** Dense embedding struggled to distinguish *"over-ear"* from *"on-ear"* within the boAt Rockerz series because descriptions share ~90% vocabulary overlap.
- **Engineering Opportunity:** Hybrid retrieval (Dense + BM25 keyword matching on `"over ear"`) or a cross-encoder reranker will demote `"on ear"` headphones.

---

## 6. How to Reproduce

Run the evaluation suite:
```bash
PYTHONPATH=. ./venv/bin/python evaluation/evaluate_recall.py
```
