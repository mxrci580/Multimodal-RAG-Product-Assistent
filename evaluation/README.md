# Multimodal Product RAG — Evaluation Suite

A rigorous, scientifically validated evaluation framework for e-commerce Retrieval-Augmented Generation (RAG). This evaluation suite tracks an empirical progression from a naive dense semantic baseline to a multimodal vision-language pipeline.

---

## Quick Reference (2-Minute Overview)

### 1. What Was Evaluated?
We evaluated the impact of each retrieval and generation component in an e-commerce assistant. We specifically investigated whether incorporating physical product imagery (multimodal grounding) provides a measurable, statistically significant improvement over text-only RAG pipelines when answering product search queries with visual and numerical constraints.

### 2. What Systems Were Compared?
We evaluated 5 retrieval configurations and 3 generation pipelines head-to-head on the same frozen benchmark:
- **Retrieval:**
  1. *Dense Baseline* (Google `text-embedding-004` + ChromaDB)
  2. *Metadata Guardrails* (Deterministic regex budget/brand filtering + dense vector search)
  3. *Hybrid Search* (Dense + Okapi BM25 + Reciprocal Rank Fusion, $k=60$)
  4. *Cross-Encoder Reranking* (`ms-marco-MiniLM-L-12-v2` via FlashRank)
  5. *Multimodal Retrieval* (Hybrid + Cross-Encoder candidate generation + `gemini-1.5-flash` visual verification)
- **Generation:**
  1. *Dense-Gen* (Dense Top-5 + text generator)
  2. *Cross-Encoder-Gen* (Cross-Encoder Top-5 + text generator)
  3. *Multimodal-Gen* (Multimodal Top-5 + image and text generator)

### 3. What Dataset Was Used?
- **Catalog:** 1,465 Amazon products (`data/amazon.csv`) covering consumer electronics, audio, wearables, and accessories.
- **Benchmark:** Frozen 50-query golden dataset (`evaluation/golden_dataset/questions_v2_50.json`) stratified across:
  - **Evidence Tiers:** 15 IMAGE, 25 TEXT+IMAGE, 10 TEXT (Control).
  - **Difficulty Tiers:** 15 Easy, 15 Medium, 12 Hard, 8 Adversarial (keyword traps).

### 4. What Metrics Were Measured?
- **Retrieval:** Hit Rate@1 (HR@1), Hit Rate@5 (HR@5), MRR, Precision@3/5, NDCG@3/5, Ideal Match@K.
- **Generation:** Answer Correctness, Claim Faithfulness, Multi-Constraint Completeness, Hallucination Rate, Evidence Attribution Accuracy.
- **Statistical Tests:** Paired McNemar's exact test (binomial), Wilcoxon signed-rank test, and 10,000-sample bootstrap 95% Confidence Intervals.

---

## Key Experimental Results

### Retrieval Comparison (Frozen 50-Query Benchmark)

| Retrieval Architecture | HR@1 | HR@5 | MRR | NDCG@3 | NDCG@5 | Stat. Sig. vs Dense ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Baseline** | 32.0% | 76.0% | 0.501 | 0.493 | 0.562 | — |
| **Metadata Guardrails** | 28.0% | 72.0% | 0.458 | 0.451 | 0.524 | No ($p > 0.05$) |
| **Hybrid (Dense + BM25)** | 28.0% | 64.0% | 0.424 | 0.379 | 0.451 | No ($p > 0.05$) |
| **Cross-Encoder Reranker** | 38.0% | 60.0% | 0.465 | 0.427 | 0.465 | No ($p > 0.05$) |
| **Multimodal Retrieval** | **76.0%** | **82.0%** | **0.790** | **0.772** | **0.775** | **Yes ($p = 0.00001 < 0.001$)** |

### Generation Comparison

| Generation Architecture | Answer Correctness | Claim Faithfulness | Multi-Constraint Completeness | Hallucination Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Dense-Gen** | 40.0% | 73.5% | 68.1% | 26.5% |
| **Cross-Encoder-Gen** | 38.0% | 70.3% | 66.1% | 29.7% |
| **Multimodal-Gen** | **78.0%** | **82.4%** | **93.7%** | **17.6%** |
| *Uplift (MM vs Dense)* | *+38.0% ($p < 0.001$)* | *+8.9% ($p < 0.01$)* | *+25.6% ($p < 0.001$)* | *-8.9% ($p < 0.01$)* |

---

## Core Findings & Engineering Takeaways

### 1. Why Was Multimodal Retrieval Necessary?
- **Dense & Cross-Encoder Blindness:** Text models cannot evaluate physical specifications that are omitted or misrepresented in product marketing descriptions. On adversarial queries (e.g., smartwatches advertising "Apple Watch UI dial" despite possessing a square hardware body), text rerankers collapsed to **25.0% HR@1**.
- **Visual Grounding Solves the Trap:** By inspecting actual product images, the multimodal retriever elevated HR@1 to **75.0%** on adversarial queries and **80.0%** on visual queries.
- **Re-ranking, Not Recall Expansion:** Multimodal retrieval did not significantly expand 5-candidate recall (HR@5 delta $+6.0\%$, $p = 0.375$). Its true value is as an **ultra-precise rank-1 filter** ($+44.0\%$ HR@1, $p < 0.001$).

### 2. Zero Regression on Text Control
On the 10 text-sufficient control queries, Generation Correctness was identical (**30.0% vs. 30.0%**, $p = 1.000$). Introducing vision models caused zero negative degradation on purely textual tasks.

### 3. Disentangling: What Failed and Why?
We mapped all 50 queries across a retrieval-generation contingency matrix:
- **0% Generator Failures on Clean Retrieval:** In 100% of cases where retrieval succeeded at Rank 1 (38/38), the generator gave an accurate, hallucination-free answer.
- **The Remaining 22% Errors:** All 11 remaining generation failures were direct cascades of retrieval failures:
  - *Candidate Pool Truncation (8%):* 4 target products ranked between #21 and #30 in Stage 1, missing the $K=20$ candidate pool.
  - *Fine-Grained 2D Image Ambiguity (6%):* Single-angle product images obscured hidden tactile buttons or subtle textures.
  - *Catalog Synonymy (8%):* The system recommended a valid, near-identical alternative SKU (e.g., 64GB vs 128GB flash drive).

---

## Methodological Limitations

1. **Catalog Scale:** Benchmarked on 1,465 Amazon products (`data/amazon.csv`).
2. **Benchmark Construction:** The 50-query golden dataset was constructed via catalog inspection and author verification, not independent double-blind crowdsourced annotators.
3. **Hyperparameters:** Tested on a single frozen test benchmark without a dedicated separate split for hyperparameter tuning.

---

## Directory Navigation

- **[evaluation.md](file:///Users/drizy/Documents/Agentic%20Multimodala%20RAG%20with%20Evaluation/evaluation/evaluation.md)** — Canonical 17-section comprehensive evaluation report (Single Source of Truth).
- **[methodology.md](file:///Users/drizy/Documents/Agentic%20Multimodala%20RAG%20with%20Evaluation/evaluation/methodology.md)** — Complete dataset construction, tier definitions, and grading protocol.
- **[metrics.md](file:///Users/drizy/Documents/Agentic%20Multimodala%20RAG%20with%20Evaluation/evaluation/metrics.md)** — Mathematical definitions and formulas for all retrieval and generation metrics.
- **`reports/`** — Detailed analysis reports:
  - `phase4d_statistical_validation_report.md`: Complete statistical tables, CIs, and test statistics.
  - `benchmark_audit_report.md`: Audit of the frozen 50-query benchmark.
  - `error_taxonomy_and_disentangling_report.md`: 15-class error taxonomy and contingency matrix.
- **`results/`** — Raw JSON outputs for Phases 4B, 4C, and 4D.
- **`scripts/`** — Executable audit and analysis scripts.
