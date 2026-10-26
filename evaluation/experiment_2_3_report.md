# Phase 2.3 Experiment Report: Cross-Encoder Reranking

**Date:** October 2026  
**Corpus:** `data/amazon.csv` (1,465 products) in ChromaDB  
**Evaluation Set:** 25 Baseline v1 queries with Graded Relevance (0–3 scale)  

## 1. Experimental Overview

In this experiment, we evaluated the full two-stage retrieval architecture:
$$\text{Query} \to \text{Dense (20)} + \text{BM25 (20)} \to \text{RRF Fusion} \to \text{Candidate Pool (20)} \to \text{Cross-Encoder (MiniLM-L12)} \to \text{Top 5}$$

## 2. Four-Way Comparative Benchmark Table

| Metric | Model A (Dense) | Model B (Metadata) | Model C (Hybrid RRF) | Model D (Cross-Encoder) | Δ (D vs C) | Δ (D vs A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hit Rate @ 1 (Rel >= 2)** | 100.0% | 100.0% | 100.0% | **100.0%** | ** +0.0%** | ** +0.0%** |
| **Hit Rate @ 3 (Rel >= 2)** | 100.0% | 100.0% | 100.0% | **100.0%** | ** +0.0%** | ** +0.0%** |
| **Hit Rate @ 5 (Rel >= 2)** | 100.0% | 100.0% | 100.0% | **100.0%** | ** +0.0%** | ** +0.0%** |
| **Ideal Match @ 1 (Rel=3)** | 100.0% | 100.0% |  96.0% | ** 96.0%** | ** +0.0%** | ** -4.0%** |
| **Ideal Match @ 3 (Rel=3)** | 100.0% | 100.0% | 100.0% | **100.0%** | ** +0.0%** | ** +0.0%** |
| **MRR** |  1.000 |  1.000 |  1.000 | ** 1.000** | **+0.000** | **+0.000** |
| **Precision @ 3** |  97.3% |  98.7% |  96.0% | ** 96.0%** | ** +0.0%** | ** -1.3%** |
| **Precision @ 5** |  95.2% |  96.8% |  93.6% | ** 92.0%** | ** -1.6%** | ** -3.2%** |
| **NDCG @ 1** |  1.000 |  1.000 |  0.977 | ** 0.977** | **+0.000** | **-0.023** |
| **NDCG @ 3** |  0.988 |  0.982 |  0.952 | ** 0.951** | **-0.001** | **-0.036** |
| **NDCG @ 5** |  0.986 |  0.987 |  0.965 | ** 0.951** | **-0.014** | **-0.036** |

## 3. Deep-Dive on the 5 Focus Queries

### `[Q02]` *"best bluetooth earbuds under 1500"*
- **NDCG@3 Progression:** Model A = `0.831` $\to$ Model C = `1.000` $\to$ **Model D = `1.000`**
- **Model A Grades:** `[3, 2, 3, 3, 2]`
- **Model C Grades:** `[3, 3, 3, 3, 3]`
- **Model D Grades:** `[3, 3, 3, 3, 1]`

| Rank | Grade | Product Name | Price | Cross-Encoder Score |
| :---: | :---: | :--- | :---: | :---: |
| 1 | **3** | boAt Airdopes 171 in Ear Bluetooth True Wireless Earbud | ₹1,199 | 0.3382 |
| 2 | **3** | boAt Airdopes 141 Bluetooth Truly Wireless in Ear Earbu | ₹1,499 | 0.2699 |
| 3 | **3** | Noise Buds Vs104 Bluetooth Truly Wireless in Ear Earbud | ₹1,299 | 0.1374 |

### `[Q05]` *"boAt Rockerz over ear bluetooth headphones"*
- **NDCG@3 Progression:** Model A = `0.862` $\to$ Model C = `0.629` $\to$ **Model D = `0.862`**
- **Model A Grades:** `[3, 1, 3, 1, 1]`
- **Model C Grades:** `[3, 1, 1, 3, 1]`
- **Model D Grades:** `[3, 1, 3, 1, 1]`

| Rank | Grade | Product Name | Price | Cross-Encoder Score |
| :---: | :---: | :--- | :---: | :---: |
| 1 | **3** | boAt Rockerz 550 Over Ear Bluetooth Headphones with Upt | ₹1,799 | 0.9999 |
| 2 | **1** | boAt Rockerz 370 On Ear Bluetooth Headphones with Upto  | ₹1,199 | 0.9998 |
| 3 | **3** | boAt Rockerz 450 Bluetooth On Ear Headphones with Mic,  | ₹1,220 | 0.9998 |

### `[Q11]` *"budget android smartphone under 7000"*
- **NDCG@3 Progression:** Model A = `1.000` $\to$ Model C = `0.746` $\to$ **Model D = `1.000`**
- **Model A Grades:** `[3, 3, 3, 2, 2]`
- **Model C Grades:** `[3, 1, 3, 0, 3]`
- **Model D Grades:** `[3, 3, 3, 1, 2]`

| Rank | Grade | Product Name | Price | Cross-Encoder Score |
| :---: | :---: | :--- | :---: | :---: |
| 1 | **3** | Redmi A1 (Light Blue, 2GB RAM, 32GB Storage) | Segment  | ₹6,499 | 0.0004 |
| 2 | **3** | Redmi A1 (Light Green, 2GB RAM 32GB ROM) | Segment Best | ₹6,499 | 0.0002 |
| 3 | **3** | Redmi A1 (Black, 2GB RAM, 32GB Storage) | Segment Best  | ₹6,499 | 0.0002 |

### `[Q16]` *"Logitech wireless mouse with USB nano receiver"*
- **NDCG@3 Progression:** Model A = `1.000` $\to$ Model C = `0.831` $\to$ **Model D = `0.866`**
- **Model A Grades:** `[3, 3, 3, 2, 2]`
- **Model C Grades:** `[3, 2, 3, 3, 3]`
- **Model D Grades:** `[3, 3, 2, 3, 3]`

| Rank | Grade | Product Name | Price | Cross-Encoder Score |
| :---: | :---: | :--- | :---: | :---: |
| 1 | **3** | Logitech B170 Wireless Mouse, 2.4 GHz with USB Nano Rec | ₹599 | 0.9999 |
| 2 | **3** | Logitech M331 Silent Plus Wireless Mouse, 2.4GHz with U | ₹1,295 | 0.9999 |
| 3 | **2** | Logitech MK240 Nano Wireless USB Keyboard and Mouse Set | ₹1,495 | 0.9995 |

### `[Q22]` *"500 watt mixer grinder with 3 stainless steel jars"*
- **NDCG@3 Progression:** Model A = `1.000` $\to$ Model C = `0.598` $\to$ **Model D = `0.697`**
- **Model A Grades:** `[3, 3, 3, 3, 2]`
- **Model C Grades:** `[2, 3, 2, 3, 3]`
- **Model D Grades:** `[3, 2, 2, 2, 2]`

| Rank | Grade | Product Name | Price | Cross-Encoder Score |
| :---: | :---: | :--- | :---: | :---: |
| 1 | **3** | Lifelong LLMG23 Power Pro 500-Watt Mixer Grinder with 3 | ₹1,299 | 0.9999 |
| 2 | **2** | Lifelong LLMG93 500 Watt Duos Mixer Grinder, 2 Stainles | ₹1,149 | 0.9998 |
| 3 | **2** | USHA RapidMix 500-Watt Copper Motor Mixer Grinder with  | ₹2,449 | 0.9998 |

## 4. Key Engineering Insights

1. **Candidate Pool Quality Dictates Reranking Upper Bound:** The cross-encoder can only reorder items present in the 20-candidate pool generated by Stage 1.
2. **Query-Document Cross-Attention Corrects Lexical Polysemy:** The cross-encoder evaluates query tokens simultaneously with document tokens, resolving the keyword trap that misled unweighted RRF.
3. **Specialized Roles in Production Architecture:**
   - **Stage 1 (High Recall):** Dense + BM25 retrieves broad candidate coverage.
   - **Stage 1.5 (Hard Constraints):** Metadata guardrails enforce strict budget and catalog boundaries.
   - **Stage 2 (High Precision):** Cross-Encoder computes fine-grained semantic relevance.
