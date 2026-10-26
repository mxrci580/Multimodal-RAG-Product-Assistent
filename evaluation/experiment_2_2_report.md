# Phase 2.2 Experiment Report: Hybrid Retrieval (Dense + BM25 + RRF)

**Date:** October 2026  
**Corpus:** `data/amazon.csv` (1,465 products) in ChromaDB  
**Evaluation Set:** 25 Baseline v1 queries with Graded Relevance (0–3 scale)  

## 1. Hypothesis

> **H1:** Hybrid Dense + BM25 retrieval will improve ranking quality for queries containing exact product attributes, terminology, and constraints that dense semantic retrieval may blur (e.g. "over ear", specific model names, specifications).

## 2. Experimental Setup

- **Model A (Baseline):** Pure Dense Retrieval (`text-embedding-004`).
- **Model B (Metadata-Filtered):** Dense + Structured constraint filtering with dense fallback.
- **Model C (Hybrid):** Dense (top 20) + BM25Okapi (top 20) fused via Reciprocal Rank Fusion ($k=60$) -> Top 5.

## 3. Results Comparison Table

| Metric | Model A (Dense) | Model B (Metadata) | Model C (Hybrid RRF) | Delta Δ (C vs A) |
| :--- | :---: | :---: | :---: | :---: |
| **Hit Rate @ 1 (Rel >= 2)** | 100.0% | 100.0% | 100.0% | ** +0.0%** |
| **Hit Rate @ 3 (Rel >= 2)** | 100.0% | 100.0% | 100.0% | ** +0.0%** |
| **Hit Rate @ 5 (Rel >= 2)** | 100.0% | 100.0% | 100.0% | ** +0.0%** |
| **Ideal Match @ 1 (Rel=3)** | 100.0% | 100.0% |  96.0% | ** -4.0%** |
| **Ideal Match @ 3 (Rel=3)** | 100.0% | 100.0% | 100.0% | ** +0.0%** |
| **MRR** |  1.000 |  1.000 |  1.000 | **+0.000** |
| **Precision @ 3** |  97.3% |  98.7% |  96.0% | ** -1.3%** |
| **Precision @ 5** |  95.2% |  96.8% |  93.6% | ** -1.6%** |
| **NDCG @ 1** |  1.000 |  1.000 |  0.977 | **-0.023** |
| **NDCG @ 3** |  0.988 |  0.982 |  0.952 | **-0.036** |
| **NDCG @ 5** |  0.986 |  0.987 |  0.965 | **-0.022** |

## 4. Query-Level ΔNDCG Breakdown

- **Queries with Positive ΔNDCG:** 1
- **Queries with Negative ΔNDCG:** 4
- **Queries with Neutral ΔNDCG:** 20

### Queries Improved by Lexical BM25 Signal
- **[Q02] `best bluetooth earbuds under 1500`** (ΔNDCG@3 = +0.169):
  - Model A Grades: `[3, 2, 3, 3, 2]`
  - Model C Grades: `[3, 3, 3, 3, 3]`
### Queries Where Dense Was Superior
- **[Q05] `boAt Rockerz over ear bluetooth headphones`** (ΔNDCG@3 = -0.232):
  - Model A Grades: `[3, 1, 3, 1, 1]`
  - Model C Grades: `[3, 1, 1, 3, 1]`
- **[Q11] `budget android smartphone under 7000`** (ΔNDCG@3 = -0.254):
  - Model A Grades: `[3, 3, 3, 2, 2]`
  - Model C Grades: `[3, 1, 3, 0, 3]`
- **[Q16] `Logitech wireless mouse with USB nano receiver`** (ΔNDCG@3 = -0.169):
  - Model A Grades: `[3, 3, 3, 2, 2]`
  - Model C Grades: `[3, 2, 3, 3, 3]`
- **[Q22] `500 watt mixer grinder with 3 stainless steel jars`** (ΔNDCG@3 = -0.402):
  - Model A Grades: `[3, 3, 3, 3, 2]`
  - Model C Grades: `[2, 3, 2, 3, 3]`

## 5. Architectural Synthesis: The 3 Retrieval Pillars

| Retrieval Signal | Superpower | Blind Spot | Ideal Use Case |
| :--- | :--- | :--- | :--- | 
| **Dense Semantic** | High-level intent, conceptual similarity, synonyms | Exact lexical terms ("over-ear" vs "on-ear"), numeric thresholds | Broad & exploratory queries |
| **Lexical BM25** | Exact keywords, technical codes, exact model numbers | Numeric inequalities (< 1500), semantic synonyms | Attribute & terminology queries |
| **Structured Metadata** | Hard numerical bounds (price <= 2000), strict categories | Noisy/sparse catalog tags, taxonomy rigidity | Hard filterable constraints |

## 6. Next Step: Phase 2.3 — Cross-Encoder Reranking
Take the top candidates from the hybrid candidate pool and pass them through a cross-encoder to compute full query-document attention scoring.
