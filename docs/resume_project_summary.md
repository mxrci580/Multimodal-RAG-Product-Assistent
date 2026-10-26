# Resume & Portfolio Project Summary

**Project:** Agentic Multimodal Product RAG with Rigorous Empirical Progression  
**Author:** AI / Machine Learning / IR Systems Engineer  
**Repository:** Multimodal-RAG-Product-Assistant  
**Evaluation Standard:** Paired Non-Parametric Significance Testing & 10,000-Iteration Bootstrap CIs  

---

## 1. One-Line Project Description

An end-to-end multimodal e-commerce Retrieval-Augmented Generation (RAG) system evaluated through a controlled ablation progression, demonstrating that visual product grounding significantly outperforms pure text retrieval on physical constraint verification ($p < 0.001$).

---

## 2. Technical Architecture

A 2-stage retrieve-and-rerank pipeline coupled with a vision-language generation layer:
1. **Stage 1 (Candidate Generation, $K=20$):**
   - **Dense Retrieval:** Google `text-embedding-004` (768d) indexing title, description, and specs in ChromaDB.
   - **Lexical Retrieval:** Okapi BM25 (`rank-bm25`) indexing keyword and SKU tokens.
   - **Structured Metadata Filtering:** Deterministic regex constraint extraction enforcing price bounds and category rules.
   - **Candidate Fusion:** Reciprocal Rank Fusion (RRF, $k=60$) combining dense and lexical candidate lists.
2. **Stage 2 (Candidate Reranking & Verification, Top-5):**
   - **Text Cross-Encoder:** Local `ms-marco-MiniLM-L-12-v2` cross-encoder (via FlashRank) scoring query-document pairs.
   - **Multimodal Visual Grounding:** Vision-Language Model (`gemini-1.5-flash`) inspecting local cached product catalog images to verify physical hardware attributes (dial shape, port layouts, cable braid finish).
3. **Stage 3 (Evidence-Grounded Generation):**
   - Multimodal prompt synthesis generating constraint-verified recommendations with explicit metadata and visual attribution.

---

## 3. Key Engineering Contributions

- **Empirical Progression Methodology:** Instead of deploying a complex multimodal architecture upfront, designed a multi-phase ablation (Phases 1 $\to$ 4D) where every architectural layer was introduced strictly to address an empirically quantified failure mode of the previous baseline.
- **Frozen 50-Query Multimodal Benchmark:** Built and audited a 50-query golden benchmark stratified across 3 evidence tiers (IMAGE, TEXT+IMAGE, TEXT control) and 4 difficulty tiers (Easy, Medium, Hard, Adversarial keyword traps) over 1,465 Amazon catalog items.
- **Statistical Significance Engine:** Implemented paired non-parametric statistical hypothesis testing (McNemar's exact binomial tests for binary accuracy, Wilcoxon signed-rank tests for continuous metrics, and 10,000-iteration bootstrap 95% confidence intervals) to validate that observed uplifts were not sampling noise.
- **Retrieval vs. Generation Disentangling:** Built a $2 \times 2$ contingency matrix proving that 100% of generation errors in the final system were cascades of retrieval omissions, with 0% generator reasoning failure when clean evidence was retrieved at Rank 1.
- **Negative Transfer Protection:** Implemented a pure-text control group demonstrating that multimodal visual grounding introduces zero degradation or regression on purely textual queries ($p = 1.000$).

---

## 4. Experimental Methodology

- **Catalog:** 1,465 real-world Amazon e-commerce products (`data/amazon.csv`) covering wearables, audio, computing accessories, and cables.
- **Ablation Baseline Progression:**
  - *Phase 1:* Dense Semantic Baseline (`text-embedding-004` + ChromaDB)
  - *Phase 2.1:* Metadata Guardrails (Scalar budget filters)
  - *Phase 2.2:* Hybrid Retrieval (Dense + BM25 + RRF)
  - *Phase 2.3:* Cross-Encoder Reranking (`ms-marco-MiniLM-L-12-v2`)
  - *Phase 3:* Multimodal Retrieval A/B Proof-of-Concept (25 queries)
  - *Phase 4B:* Full 5-System Retrieval Benchmark (50 frozen queries)
  - *Phase 4C:* End-to-End Generation Evaluation (Claim-level faithfulness & hallucination)
  - *Phase 4D:* Statistical Significance Validation (Paired McNemar, Wilcoxon, Bootstrap CIs)
- **Graded Relevance:** Evaluated against 4-point graded relevance (Grade 3 = Ideal Target, Grade 2 = Relevant Alternative, Grade 1 = Partial Match, Grade 0 = Irrelevant Distractor).

---

## 5. Strongest Defensible Metrics

All metrics derived from the frozen 50-query benchmark ($N=50$) comparing the Final Multimodal System against the Dense Baseline:

1. **Retrieval Rank-1 Precision (Hit Rate@1):**
   - Dense Baseline: **32.0%** $\to$ Multimodal: **76.0%**
   - Absolute Delta: **$+44.0\%$** (95% CI: $[+28.0\%, +60.0\%]$)
   - Significance: **$p = 0.00001 < 0.001$** (Paired McNemar's exact test)
2. **Retrieval Ranking Quality (NDCG@3):**
   - Dense Baseline: **0.493** $\to$ Multimodal: **0.772**
   - Absolute Delta: **$+0.278$** (95% CI: $[+0.184, +0.376]$)
   - Significance: **$p = 0.00002 < 0.001$** (Wilcoxon signed-rank test)
3. **End-to-End Answer Correctness:**
   - Dense-Gen: **40.0%** $\to$ Multimodal-Gen: **78.0%**
   - Absolute Delta: **$+38.0\%$** (95% CI: $[+24.0\%, +52.0\%]$)
   - Significance: **$p = 0.00001 < 0.001$** (Paired McNemar's exact test)
4. **Multi-Constraint Completeness:**
   - Dense-Gen: **68.1%** $\to$ Multimodal-Gen: **93.7%**
   - Absolute Delta: **$+25.6\%$** (95% CI: $[+17.0\%, +34.0\%]$)
   - Significance: **$p = 0.00000 < 0.001$** (Wilcoxon signed-rank test)
5. **Measured Hallucination Rate Reduction:**
   - Dense-Gen: **26.5%** $\to$ Multimodal-Gen: **17.6%**
   - Absolute Reduction: **$-8.9\%$** (95% CI: $[-15.0\%, -2.7\%]$)
   - Significance: **$p = 0.0016 < 0.01$** (Wilcoxon signed-rank test)
6. **Text Control Tier Invariance:**
   - Answer Correctness: **30.0% vs. 30.0%** ($\Delta = 0.0\%$, $p = 1.000$)
   - Proves zero regression or negative transfer on non-visual queries.

---

## 6. Major Failure Modes Discovered

Across 15 formal error categories, empirical testing revealed:
1. **Vector Insensitivity to Scalar Budgets:** Dense embeddings clustered premium products (₹4,999) near budget items (₹999), solved deterministically via regex metadata guardrails.
2. **Lexical Polysemy & Suffix Traps:** BM25 lexical search matched unrelated items sharing technical terms (e.g., "Android phone" matching "Android TV remote"), requiring cross-encoder interaction modeling.
3. **Cross-Encoder Keyword Vulnerability on Adversarial Copy:** Cross-encoders collapsed to **25.0% HR@1** on adversarial product descriptions where marketers stuffed physical keywords (e.g., "smartwatch with round UI dial" on a square body). Multimodal inspection of actual hardware images solved this, reaching **75.0% HR@1**.
4. **Candidate Pool Truncation Ceiling:** 4 of the remaining 11 Multimodal failures occurred because the target product ranked #21–#30 in Stage 1, falling outside the $K=20$ candidate pool passed to the vision model.

---

## 7. Final Architecture Justification

Every layer exists because an empirical ablation demanded it:
- **Dense Vector Search:** High semantic recall for broad product categories, but blind to numerical thresholds.
- **Metadata Guardrails:** Necessary for non-negotiable scalar constraints (budget caps, explicit brands).
- **BM25 Lexical Search + RRF:** Necessary for exact product codes and technical acronyms (e.g., "GaN", "ANC").
- **Cross-Encoder Reranker:** Necessary to eliminate lexical false positives from BM25.
- **Multimodal Grounding:** Essential because text descriptions frequently omit or misrepresent physical hardware properties.
- **Vision-Language Generator:** Synthesizes final recommendations directly grounded in retrieved textual specs and visual imagery.

---

## 8. Resume Bullets (Choose 3 or 5)

### 3-Bullet Version (High-Impact ML/IR Systems Role)
- **Architected and evaluated an end-to-end multimodal e-commerce RAG pipeline** combining dense embeddings (`text-embedding-004`), BM25 lexical fusion, cross-encoder reranking (`ms-marco-MiniLM-L-12-v2`), and vision-language grounding (`gemini-1.5-flash`) over a 1,465-product catalog.
- **Demonstrated that multimodal visual grounding increased retrieval Hit Rate@1 from 32.0% to 76.0%** ($\Delta = +44.0\%$, $p < 0.001$) and end-to-end recommendation correctness from 40.0% to 78.0% ($\Delta = +38.0\%$, $p < 0.001$) on a frozen 50-query golden benchmark.
- **Validated statistical defensibility via paired McNemar and Wilcoxon tests with 10,000 bootstrap CIs**, proving zero regression on purely textual queries ($p = 1.000$) and establishing that 100% of generation errors were downstream cascades of retrieval omissions.

### 5-Bullet Version (Detailed ML/IR Evaluation & Engineering)
- **Engineered a 2-stage multimodal product search and RAG architecture** incorporating ChromaDB dense retrieval, Okapi BM25, Reciprocal Rank Fusion ($k=60$), cross-encoder reranking, and visual verification over 1,465 e-commerce products.
- **Built and audited a frozen 50-query golden evaluation benchmark** stratified across 3 evidence tiers (IMAGE, TEXT+IMAGE, TEXT) and 4 difficulty tiers, verifying catalog presence, price constraints, and absence of SKU leakage.
- **Achieved statistically significant retrieval improvements** with multimodal reranking, elevating HR@1 from 32.0% to 76.0% ($p = 0.00001$) and NDCG@3 from 0.493 to 0.772 ($p = 0.00002$), while showing that HR@5 was not significantly altered ($p = 0.38$), proving vision acts as a precision rank-1 reranker.
- **Reduced measured hallucination rate from 26.5% to 17.6%** ($p = 0.0016$) and increased multi-constraint completeness from 68.1% to 93.7% ($p < 0.001$) on complex multi-attribute e-commerce queries.
- **Conducted systematic error taxonomy and disentangling analysis across 15 failure classes**, showing that cross-encoders collapsed on adversarial marketing copy (25% HR@1) where visual inspection succeeded (75% HR@1), and establishing that clean Rank-1 retrieval yielded 0% generator errors.

---

## 9. LinkedIn & Portfolio Description

### Project Title
**Empirical Multimodal RAG with Statistical Validation for E-Commerce Search**

### Summary
Most RAG projects add multimodal vision models simply because the APIs exist. In this project, I treated multimodal retrieval as an empirical research question: **Does visual evidence actually provide quantifiable, statistically defensible value over text-based search in e-commerce?**

To answer this, I constructed an empirical ablation progression across 1,465 Amazon products and a frozen 50-query benchmark stratified by evidence and difficulty tiers:
1. **Dense baseline:** High category recall, but failed on scalar budgets and physical form-factors.
2. **Metadata guardrails:** Fixed budget violations, but brittle under messy category taxonomies.
3. **BM25 + RRF:** Retrieved specific SKUs, but introduced lexical polysemy traps.
4. **Cross-encoder reranker:** Fixed lexical confusion, but collapsed to 25% HR@1 on adversarial marketing copy (e.g., smartwatches marketing "round UI dial" on a square body).
5. **Multimodal visual grounding:** Inspected physical product images, elevating HR@1 to 76.0% ($p < 0.001$) and recommendation correctness to 78.0% ($p < 0.001$).

By conducting paired non-parametric statistical hypothesis tests (McNemar's exact tests, Wilcoxon signed-rank tests, and 10,000 bootstrap CIs), I validated that:
- Visual grounding acts as an ultra-precise rank-1 filter rather than expanding broad recall.
- Purely textual queries experienced zero regression ($\Delta = 0.0\%$, $p = 1.000$).
- 100% of generation failures were downstream retrieval cascades, with 0% generator reasoning errors when clean evidence was retrieved at Rank 1.

**Tech Stack:** Python, Google Gemini Flash (`gemini-1.5-flash`), `text-embedding-004`, ChromaDB, FlashRank (`ms-marco-MiniLM-L-12-v2`), `rank-bm25`, SciPy, NumPy, Pandas, Streamlit.

---

## 10. Technical Interview Walkthrough

### "Tell me about this project."
*"I built a multimodal product retrieval and recommendation system over an Amazon e-commerce catalog, but approached it primarily as an empirical IR evaluation problem. Instead of jumping to an expensive vision-language model, I built the architecture in stages to see where standard text retrieval actually fails.*

*I started with a dense semantic retriever using Google embeddings and ChromaDB. It handled broad categories well, but failed on scalar constraints like price budgets and exact physical attributes like over-ear vs. on-ear headphones. Adding metadata guardrails fixed price violations, and BM25 with Reciprocal Rank Fusion recovered specific product codes. However, lexical search introduced false positives. A local cross-encoder fixed the lexical noise, but it was still completely blind to physical product reality—on adversarial product listings that stuffed marketing keywords like 'round dial UI' into the title of a square watch, the cross-encoder collapsed to 25% Hit Rate@1.*

*To solve this, I introduced multimodal reranking where Gemini Flash inspected cached catalog images for the top-20 candidates. On a frozen 50-query golden benchmark, Hit Rate@1 jumped from 32% to 76%, and end-to-end recommendation correctness increased from 40% to 78%. I ran McNemar's exact tests and Wilcoxon signed-rank tests with 10,000 bootstrap confidence intervals to prove these uplifts were statistically significant at p < 0.001.*

*Critically, I also validated a dedicated text-only control group, proving that multimodal reranking caused zero regression on purely textual queries. When I mapped the errors across a contingency matrix, I discovered that 100% of the remaining generation failures were retrieval omissions where the item fell outside the candidate pool, rather than LLM reasoning failures."*

### "What was the most surprising engineering finding?"
*"The most insightful finding was the difference between Hit Rate@1 and Hit Rate@5. When we added multimodal reranking, Hit Rate@1 jumped by 44 percentage points (from 32% to 76%, p < 0.001). But Hit Rate@5 only changed by 6 percentage points (from 76% to 82%), which had a p-value of 0.38—not statistically significant.*

*That proved multimodal verification isn't finding brand-new candidates that the text search completely missed; it is acting as a precision re-ranker that evaluates physical attributes to promote the correct product from rank 3 or 4 directly to rank 1. In production, this means you can keep a lightweight text pipeline for broad candidate generation, and reserve expensive multimodal calls only for top-K re-ranking."*

### "What are the remaining limitations?"
*"There are three honest limitations. First, catalog scale: this was evaluated on 1,465 Amazon products, not tens of millions of items. Second, candidate pool truncation: 4 of our 11 remaining failures happened because the target item ranked between #21 and #30 in Stage 1, so the vision model never saw it. Increasing K from 20 to 50 would fix that. Third, single-angle 2D image ambiguity: certain physical queries, like checking whether an earbud has physical click buttons versus capacitive touch, couldn't be resolved because the button was hidden from the primary catalog thumbnail angle."*
