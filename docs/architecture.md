# System Architecture & Empirical Engineering Rationale

**Project:** Agentic Multimodal Product RAG with Evaluation  
**Dataset:** Amazon E-Commerce Catalog (1,465 products)  
**Methodological Principle:** Every architectural layer is justified through the empirical sequence:  
$$\text{PROBLEM} \longrightarrow \text{HYPOTHESIS} \longrightarrow \text{EXPERIMENT} \longrightarrow \text{RESULT} \longrightarrow \text{ENGINEERING DECISION}$$

---

## 1. High-Level Architectural Pipeline

```
                                      USER QUERY
                                          │
                                          ▼
                               Query Understanding
                                          │
                        ┌─────────────────┴─────────────────┐
                        │                                   │
                        ▼                                   ▼
              Hard Numerical Constraints              Lexical & Semantic
               (Price Cap, Category)                     Query Strings
                        │                                   │
                        ▼                                   ▼
                 Metadata Filter                     Dense Embedding
                        │                          (`text-embedding-004`)
                        │                                   │
                        │                                   ▼
                        │                            Okapi BM25 Index
                        │                         (Tokenized Catalog)
                        │                                   │
                        │                                   ▼
                        │                       Reciprocal Rank Fusion (RRF)
                        │                                (k = 60)
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          ▼
                               Candidate Pool Selection
                                       (K = 20)
                                          │
                                          ▼
                               Cross-Encoder Reranking
                            (`ms-marco-MiniLM-L-12-v2`)
                                          │
                                          ▼
                            Top Candidate Pre-Selection
                                          │
                                          ▼
                             Multimodal Visual Grounding
                             ┌────────────┴────────────┐
                             ▼                         ▼
                      Catalog Text Metadata     Product Image Thumbnails
                             │                         │
                             └────────────┬────────────┘
                                          ▼
                            Vision-Language Model (VLM)
                                (`gemini-1.5-flash`)
                                          │
                                          ▼
                            Evidence-Grounded Recommendation
                           (With Explicit Visual Attribution)
```

---

## 2. Layer-by-Layer Empirical Justification

### Layer 1: Dense Semantic Retrieval (Embedding Vector Search)

- **Problem:** Users express product queries using natural language intent, conversational phrasing, and non-lexical concepts (e.g., "durable fitness tracker for running in the rain") that do not appear verbatim in product titles.
- **Hypothesis:** Dense vector embeddings map user intent and product descriptions into a shared latent space, capturing semantic similarity regardless of exact token overlap.
- **Experiment:** Phase 1 evaluated Google `text-embedding-004` (768 dimensions, cosine similarity) indexed in ChromaDB across 25 natural language shopping queries.
- **Result:**
  - High recall on broad thematic queries (MRR = 0.501, NDCG@5 = 0.562 on 50 queries).
  - Severe failure on scalar constraints: for queries like "earbuds under ₹1,500", dense retrieval returned ₹2,499 earbuds because their text description was semantically identical.
  - Form-factor confusion: could not distinguish "over-ear" from "on-ear" headphones.
- **Engineering Decision:** Retain dense embeddings as the broad semantic candidate generator, but strip it of the responsibility to enforce scalar bounds or verify exact physical hardware specifications.

---

### Layer 2: Structured Metadata Guardrails (Regex & Structured Filters)

- **Problem:** Vector embeddings are continuous representations in high-dimensional space and possess no mathematical notion of inequality bounds (e.g., $\text{Price} \le 1500$). Products with price ₹4,999 cluster closely with ₹999 products in embedding space because marketing descriptions use identical vocabulary.
- **Hypothesis:** Extracting numerical constraints (e.g., max budget) and applying deterministic database-level filters will eliminate scalar constraint violations.
- **Experiment:** Phase 2.1 introduced a deterministic query parser that extracted budget numbers and mapped them to ChromaDB `$and` metadata filters (`price <= budget`).
- **Result:**
  - Budget violation rate dropped to exactly 0.0% on filtered queries.
  - Precision@5 increased from 95.2% $\to$ 96.8% on numerical queries.
  - Failure Mode: Rigid metadata filtering caused zero-recall drops when catalog taxonomy was messy or synonyms differed (e.g., filtering for "neckband" when the catalog item was tagged under "headphones").
- **Engineering Decision:** Deploy metadata guardrails as non-blocking pre-filters or soft-guardrails for hard budgetary caps, but avoid overly restrictive hierarchical category filters during initial candidate generation.

---

### Layer 3: Lexical Matching (Okapi BM25 Indexing)

- **Problem:** E-commerce shoppers frequently search for exact technical specifications, standard acronyms (e.g., "GaN", "ANC", "USB-C to Lightning"), or brand/model codes (e.g., "Airdopes 141", "Braided 65W"). Dense embeddings often smooth over rare technical tokens and retrieve generic alternatives.
- **Hypothesis:** An inverted lexical index using Okapi BM25 will provide exact keyword match sensitivity, ensuring products containing the specific SKU tokens enter the candidate pool.
- **Experiment:** Phase 2.2 indexed title and description tokens using `rank-bm25` alongside the ChromaDB vector store.
- **Result:**
  - Candidate recall for exact model numbers and niche specs improved significantly.
  - Overall candidate pool recall (HR@20) reached 86.0%.
  - Failure Mode: Pure lexical search suffered from term polysemy (e.g., "Android TV remote" matching "Android smartphones") and vocabulary mismatch.
- **Engineering Decision:** Retain BM25 as an independent candidate retrieval stream, to be fused with dense semantic candidates.

---

### Layer 4: Candidate Fusion (Reciprocal Rank Fusion — RRF)

- **Problem:** Dense search outputs cosine similarity scores (typically bounded between $[0, 1]$), while BM25 outputs unbounded lexical scores ($[0, \infty)$). Directly summing or linearly combining raw scores causes one model's score distribution to dominate unpredictably across different queries.
- **Hypothesis:** Rank-based fusion using Reciprocal Rank Fusion (RRF) normalizes arbitrary score distributions and balances semantic and lexical signals without requiring query-dependent weight calibration.
- **Experiment:** Phase 2.2 implemented RRF with smoothing constant $k=60$:
  $$RRF\_Score(d) = \frac{1}{60 + r_{dense}(d)} + \frac{1}{60 + r_{bm25}(d)}$$
- **Result:**
  - Robust merging of dense and lexical candidates; overall retrieval Hit Rate@5 increased from 76.0% (Dense) $\to$ 82.0% (Hybrid).
  - NDCG@3 improved from 0.493 $\to$ 0.539.
  - Failure Mode: When both dense and lexical lists had noise, low-quality lexical matches appeared at ranks 1–3, diluting top-1 precision.
- **Engineering Decision:** Standardize on RRF ($k=60$) to construct a merged candidate pool of $K=20$ items for downstream reranking, but do not rely on raw RRF scores for the final top-1 ranking.

---

### Layer 5: Cross-Encoder Reranking (`ms-marco-MiniLM-L-12-v2`)

- **Problem:** Bi-encoders (dense embeddings) and lexical indexes (BM25) compute query and document representations independently. They fail to capture fine-grained token-level interactions, causing false positives when products share words but have different semantic relationships (e.g., "watch charger" vs. "smartwatch with charger included").
- **Hypothesis:** A cross-encoder model that processes the query and product text simultaneously through joint self-attention will score subtle query-document relationships and reorder the hybrid candidate pool accurately.
- **Experiment:** Phase 2.3 incorporated `ms-marco-MiniLM-L-12-v2` (via FlashRank) to rerank the Top-$K=20$ hybrid candidates.
- **Result:**
  - Hit Rate@1 improved from 36.0% $\to$ 40.0%, and MRR improved from 0.535 $\to$ 0.559 on the 50-query benchmark.
  - Successfully resolved lexical traps where accessories had matched target devices.
  - Critical Failure Mode: On adversarial queries where product titles copied marketing buzzwords (e.g., "round dial UI" on a square-cased smartwatch), the cross-encoder placed the wrong watch at Rank 1 (achieving only 25.0% HR@1 on adversarial queries).
- **Engineering Decision:** Retain the cross-encoder as a fast, local text reranker to eliminate lexical noise, but recognize that text rerankers are fundamentally vulnerable to marketing copy and unable to verify physical hardware attributes.

---

### Layer 6: Multimodal Visual Grounding (Vision-Language Verification)

- **Problem:** E-commerce catalogs are rife with missing, incomplete, or misleading textual descriptions. Sellers frequently fail to mention whether a dial is circular or square, whether a cable has a braided nylon jacket or PVC, or which specific ports are arranged on a multiport charger. Text models cannot verify what is unwritten or deceptive.
- **Hypothesis:** Providing high-resolution product catalog images to a Vision-Language Model (VLM) allows physical inspection of hardware appearance, geometry, materials, and form-factors, resolving visual ambiguities that text models miss.
- **Experiment:** Phase 3 and Phase 4B provided local cached catalog images alongside extracted product metadata to `gemini-1.5-flash` for candidate verification and re-ranking.
- **Result:**
  - Massive, statistically significant leap in retrieval precision:
    - **HR@1 surged from 32.0% (Dense) and 40.0% (Cross-Encoder) $\to$ 76.0% (Multimodal)** ($p = 0.00001 < 0.001$, McNemar's exact test).
    - **NDCG@3 surged from 0.493 $\to$ 0.772** ($p = 0.00002 < 0.001$, Wilcoxon signed-rank test).
    - On Adversarial queries, HR@1 jumped from 25.0% $\to$ 75.0%.
  - Critical Finding on Re-ranking vs. Recall: HR@5 changed only from 76.0% $\to$ 82.0% ($p = 0.375$, not statistically significant). This confirmed that multimodal verification acts as a **precision re-ranker** that elevates candidate items from lower ranks to Rank 1, rather than expanding broad recall.
  - Invariance on Text Control: On purely textual queries, generation correctness remained identical (60.0% vs. 60.0%, $p = 1.000$), demonstrating zero negative regression.
- **Engineering Decision:** Deploy multimodal verification as the final, decisive stage in the retrieval pipeline before generation.

---

### Layer 7: Evidence-Grounded Vision-Language Generator

- **Problem:** Standard LLMs suffer from parametric hallucinations when recommending products. If an LLM is asked for a "blue smartwatch with a circular dial under ₹2,000", it may invent battery life, hallucinate water resistance ratings, or misattribute features between different candidate products.
- **Hypothesis:** Constraining the generator with retrieved catalog metadata and product imagery, combined with explicit citation requirements, will reduce hallucinations and enforce multi-constraint completeness.
- **Experiment:** Phase 4C evaluated Dense-Gen, Cross-Encoder-Gen, and Multimodal-Gen on Claim Faithfulness, Answer Correctness, Multi-Constraint Completeness, and Hallucination Rate across 50 queries.
- **Result:**
  - **Answer Correctness:** 78.0% (Multimodal-Gen) vs. 40.0% (Dense-Gen) ($p = 0.00001$).
  - **Multi-Constraint Completeness:** 93.7% vs. 68.1% ($p < 0.001$).
  - **Measured Hallucination Rate:** Reduced from 26.5% $\to$ 17.6% ($p = 0.0016$).
  - **Disentangling Analysis:** 100% of cases where retrieval placed the target product at Rank 1 (38/38) resulted in completely correct, hallucination-free generation (0% generator error).
- **Engineering Decision:** Implement an evidence-grounded generator that enforces strict citation of retrieved catalog specifications and visual features.

---

## 3. Summary of Component Roles

| Pipeline Layer | Primary Function | Failure Mode Addressed | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Dense Search** | Broad semantic retrieval | Lexical vocabulary mismatch | Top-50 semantic candidates |
| **Metadata Filter** | Hard constraint enforcement | Price budget violations | Validated scalar subset |
| **BM25 Index** | Exact token/SKU matching | Rare acronyms & model codes | Top-50 lexical candidates |
| **RRF Fusion** | Candidate normalization | Score distribution divergence | Merged Candidate Pool ($K=20$) |
| **Cross-Encoder** | Joint interaction reranking | Lexical polysemy traps | Reordered Top-20 text candidates |
| **Multimodal VLM** | Visual attribute verification | Deceptive/missing text descriptions | Reordered Top-5 verified candidates |
| **Generator** | Grounded response synthesis | Parametric hallucination & omissions | Final user recommendation |

---

## 4. Hardware & Software Dependencies

- **Runtime:** Python 3.13
- **Embeddings:** Google `text-embedding-004` (via `google-genai`)
- **Vector DB:** ChromaDB `v0.5.0`
- **Lexical Index:** `rank-bm25` `v0.2.2`
- **Cross-Encoder:** `ms-marco-MiniLM-L-12-v2` via `flashrank` `v0.2.10`
- **Vision-Language Model:** Google `gemini-1.5-flash` (via `google-genai`)
- **Configuration:** Centralized in `app/evaluation_config.py`
