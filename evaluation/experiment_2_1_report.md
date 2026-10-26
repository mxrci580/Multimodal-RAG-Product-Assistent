# Phase 2.1 Experiment Report: Metadata-Filtered Retrieval vs. Baseline Dense

**Date:** October 2026  
**Corpus:** `data/amazon.csv` (1,465 products) in ChromaDB  
**Evaluation Set:** 25 Baseline v1 queries with Graded Relevance (0–3 scale)  

## 1. Experiment Overview

- **Model A (Baseline):** Pure Dense Vector Retrieval (`text-embedding-004`).
- **Model B (Metadata-Filtered):** Structured constraint extraction (`max_price`, `subcategory`) + ChromaDB filtered search with anti-false-negative dense fallback.

## 2. Quantitative Results Table

| Metric | Model A (Dense Baseline) | Model B (Metadata-Filtered) | Delta (Δ) |
| :--- | :---: | :---: | :---: |
| **Hit Rate @ 1 (Rel >= 2)** | 100.0% | 100.0% | ** +0.0%** |
| **Hit Rate @ 3 (Rel >= 2)** | 100.0% | 100.0% | ** +0.0%** |
| **Hit Rate @ 5 (Rel >= 2)** | 100.0% | 100.0% | ** +0.0%** |
| **Ideal Match @ 1 (Rel=3)** | 100.0% | 100.0% | ** +0.0%** |
| **Ideal Match @ 3 (Rel=3)** | 100.0% | 100.0% | ** +0.0%** |
| **MRR** |  1.000 |  1.000 | **+0.000** |
| **Precision @ 3** |  97.3% |  98.7% | ** +1.3%** |
| **Precision @ 5** |  95.2% |  96.8% | ** +1.6%** |
| **NDCG @ 1** |  1.000 |  1.000 | **+0.000** |
| **NDCG @ 3** |  0.988 |  0.982 | **-0.005** |
| **NDCG @ 5** |  0.986 |  0.987 | **+0.000** |

## 3. Findings & Engineering Insights

### Key Metrics Summary
- **Precision @ 3 improved from 97.3% to 98.7% (+1.3%)**: Enforcing metadata constraints successfully eliminated irrelevant or out-of-budget products from the top positions.
- **Precision @ 5 improved from 95.2% to 96.8% (+1.6%)**: Deeper rank quality improved as price and category boundaries were maintained.
- **Hit Rate @ 1, 3, 5 and MRR remained at 100% / 1.000**: No relevant items were lost at Rank 1.
- **NDCG @ 3 slightly shifted (0.988 → 0.982, -0.005)**: Caused by a specific catalog categorization boundary in Q05.

---

### Deep-Dive on Affected Queries

#### 1. The Win: `[Q08]` *"boAt calling smartwatch under 2000"* (Price Constraint)
* **Model A (Baseline Dense):**
  - Rank 1: `boAt Wave Call` (₹1,999) — Grade 3 [Ideal]
  - Rank 2: `boAt Newly Launched Wave Electra` (₹2,499) — Grade 1 [Over Budget by ₹500]
  - Rank 3: `boAt Newly Launched Wave Electra` (₹2,499) — Grade 1 [Over Budget by ₹500]
* **Model B (Metadata-Filtered):**
  - Rank 1: `boAt Wave Call` (₹1,999) — Grade 3 [Ideal]
  - Rank 2: `boAt Wave Call` (₹1,999) — Grade 3 [Ideal]
  - Rank 3: `boAt Wave Call` (₹1,999) — Grade 3 [Ideal]
* **Insight:** Dense search was blind to the hard numerical price ceiling and suggested ₹2,499 smartwatches. Metadata filtering strictly enforced `numeric_price <= 2000`, replacing the over-budget options with valid ₹1,999 calling watches, boosting top-3 precision from 33% to 100%!

#### 2. The Nuance & Lesson: `[Q05]` *"boAt Rockerz over ear bluetooth headphones"* (Catalog Boundary)
* **Model A (Baseline Dense):**
  - Rank 1: `boAt Rockerz 550 Over Ear` (₹1,799) — Grade 3
  - Rank 2: `boAt Rockerz 400 On-Ear` (₹1,399) — Grade 1
  - Rank 3: `boAt Rockerz 450 On-Ear / Over-Ear` (₹1,220) — Grade 3
* **Model B (Metadata-Filtered):**
  - Rank 1: `boAt Rockerz 550 Over Ear` (₹1,799) — Grade 3
  - Rank 2: `Boult Audio Bass Buds Q2 Over-Ear` (₹649) — Grade 2
  - Rank 3: `boAt Rockerz 400 On-Ear` (₹1,399) — Grade 1 [From Fallback]
  - Rank 4: `boAt Rockerz 450 On-Ear` (₹1,220) — Grade 3 [From Fallback]
* **Insight:** In the raw Amazon catalog, only **2 products** in the entire dataset are formally tagged under `subcategory: 'Over-Ear'` (`boAt 550` and `Boult Q2`). Popular alternative `boAt Rockerz 450` was categorized by Amazon under `On-Ear`. Strict subcategory filtering isolated the 2 Over-Ear items and our anti-false-negative fallback backfilled the remaining slots, which pushed `boAt 450` from Rank 3 to Rank 4.
* **Core Takeaway:** This confirms the exact hypothesis: **Hard metadata filtering on imperfect e-commerce catalogs can create false negatives or artificial scarcity when products are misclassified or span hybrid form-factors.**

---

## 4. Next Step: Phase 2.2 — Hybrid Retrieval (Dense + BM25 with RRF)

Rather than rigid hard-filtering on catalog subcategory tags, **Hybrid Retrieval (Dense + BM25)** allows exact terminology like `"over ear"` to be strongly rewarded via lexical matching without discarding products that share overlapping category taxonomy.

