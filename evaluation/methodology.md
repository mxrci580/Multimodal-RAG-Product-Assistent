# Benchmark Dataset & Evaluation Methodology

**Benchmark Version:** `questions_v2_50.json` (Phase 4 Frozen Benchmark)  
**Corpus:** Amazon Product Dataset (`data/amazon.csv`, 1,465 products, 1,351 unique SKUs)  
**Evaluation Scope:** 5-System Retrieval Benchmark & 3-System Generation Evaluation  
**Auditor / Author:** ML/IR Evaluation Engineering Team  
**Date:** October 2026  

---

## 1. Overview & Research Objective

The primary objective of this benchmark is to provide a **rigorous, reproducible, and fine-grained evaluation framework** for e-commerce product search. Standard RAG benchmarks frequently rely on generic synthetic questions evaluated with binary recall, masking the subtle failure modes of vector embeddings, structured filters, lexical search, cross-encoders, and vision-language models.

This benchmark was engineered to specifically evaluate:
1. When semantic dense embeddings succeed and fail.
2. The limits of structured metadata filtering against noisy catalog taxonomy.
3. Lexical polysemy traps introduced by keyword fusion (BM25 + RRF).
4. Candidate-pool ceilings in cross-encoder reranking.
5. The specific uplift and boundary conditions of multimodal vision grounding.

---

## 2. Catalog Snapshot & Grounding

- **Source Corpus:** A curated snapshot of the Amazon India Product Catalog containing **1,465 items** across major electronics and consumer categories (Audio, Smartwatches, Home Entertainment, Cables & Chargers, Computer Peripherals, Home & Kitchen Appliances).
- **Unique Products:** 1,351 unique `product_id`s (accounting for color and bundle variations).
- **Metadata Fields:** `product_id`, `product_name`, `category`, `discounted_price`, `actual_price`, `rating`, `rating_count`, `about_product`, `img_link`.
- **Image Cache:** 82 high-resolution product hero images verified on disk in `data/image_cache/` covering all ground-truth target and candidate items.
- **Corpus Snapshot Freeze:** The catalog file `data/amazon.csv` was frozen prior to benchmark construction to guarantee reproducibility across all runs.

---

## 3. Query Construction & Taxonomy

The benchmark comprises exactly **50 queries**, structured across a **3-dimensional matrix**: Evidence Tier, Difficulty Level, and User Intent.

### A. The Three Evidence Tiers (N = 50)
Every query is categorized by the minimum modality required to verify compliance:

1. **`IMAGE` (Visual-Dominant, 15 queries):**  
   - *Definition:* Queries where the decisive criterion is a physical, geometric, or aesthetic attribute that is omitted, ambiguous, or unreliably described in the catalog text.
   - *Examples:* Circular vs. rectangular smartwatch bezel, hardware rotating side crown, elongated stem vs. compact pebble earbud, physical 7-segment LED percentage readout vs. single status dot, dual-tone nylon braided jacket vs. smooth PVC, vertical pop-up toaster slots vs. clamshell sandwich grill plates.
2. **`TEXT+IMAGE` (Hybrid Constraints, 25 queries):**  
   - *Definition:* Realistic consumer shopping queries combining hard structured constraints (price budget ceilings, brand preferences, technical specifications) with physical visual affordances.
   - *Examples:* *“Round dial smartwatch under ₹2,000 with Bluetooth calling”*, *“32-inch Smart Android TV under ₹15,000 with frameless three-sided bezel-less screen”*, *“Ergonomic wireless mouse under ₹1,000 with contoured thumb rest”*.
3. **`TEXT` (Control Group, 10 queries):**  
   - *Definition:* Pure technical specification queries where catalog text is comprehensive and sufficient.
   - *Examples:* Motor wattage (500W vs 750W mixer grinder), HDMI 2.1 48Gbps 8K@60Hz bandwidth, 20000mAh Power Delivery power bank.
   - *Role in Benchmark:* Serves as a scientific control to verify that introducing visual re-ranking causes **zero regression ($\Delta = 0.000$)** on text-sufficient queries.

### B. The Four Difficulty Levels (N = 50)
1. **Easy (15 queries):** Single unambiguous constraint or clear visual attribute without conflicting marketing copy.
2. **Medium (15 queries):** Dual constraints (e.g. price ceiling + physical form factor).
3. **Hard (12 queries):** Multi-constraint queries requiring simultaneous compliance across 3+ technical, budget, and physical attributes.
4. **Adversarial (8 queries):** Explicitly designed traps that induce failure in standard text or lexical search:
   - *Adversarial Trap 1 (UI vs Hardware):* Query asks for a circular round smartwatch; candidate text contains deceptive marketing string *"round dial UI experience"*, but hardware casing is rectangular.
   - *Adversarial Trap 2 (Token Collision):* Query asks for a pop-up bread toaster; candidate contains the token *"toaster"* in title, but is physically a horizontal sandwich clamp.
   - *Adversarial Trap 3 (Single Dot vs Numeric Display):* Query asks for numeric battery percentage; candidate text says *"LED indicator"*, but is a single blinking light.

### C. User Intent Distribution
- Visual Appearance: 15 queries
- Multi-Constraint: 18 queries
- Feature Matching: 7 queries
- Price Constrained: 6 queries
- Brand + Feature: 2 queries
- Comparison: 1 query
- Exact Lookup: 1 query

---

## 4. Graded Relevance Methodology (0–3 Scale)

Binary relevance (Relevant vs. Irrelevant) fails to capture realistic IR ranking behavior in e-commerce, where a search engine can return a partially compliant item (e.g., correct category but ₹100 over budget) above an ideal target.

We employ an industry-standard 4-point graded relevance scale:

| Grade | Label | Operational Definition |
| :---: | :--- | :--- |
| **3** | **Ideal / Target Match** | Fully satisfies all hard numerical constraints, category boundaries, technical specifications, and physical visual requirements. |
| **2** | **Relevant / High Quality** | Fully satisfies primary intent and hard constraints with minor secondary variation (e.g., matching category and specs with an alternative colorway). |
| **1** | **Partially Relevant / Marginal** | Matches broad category but violates a non-critical secondary constraint (e.g., on-ear headphones when over-ear was requested, or budget exceeded by < 10%). |
| **0** | **Irrelevant / Trap** | Violates category boundaries, hard budget limits, or physical visual requirements (e.g., sandwich press returned for pop-up toaster query; rectangular smartwatch returned for round dial query). |

---

## 5. Candidate Judgment & Annotation Protocol

- **Candidate Pool Depth:** For every query, all candidates appearing in the Top 20 across any tested retriever were audited.
- **Labeling Process:**
  - *Explicit Target Products:* Each query explicitly defines 1 to 3 verified Grade 3 target products in `relevant_products`.
  - *Adversarial Trap Grounding:* Adversarial queries explicitly define Grade 0 negative targets (e.g. square watches or sandwich presses) to test whether models fall for keyword bait.
  - *Unlisted Candidate Evaluation:* Candidates outside the explicit list are evaluated via a strict fallback rule:
    - If price exceeds budget ceiling by $> 15\% \to$ Grade 0.
    - If specified brand is missing $\to$ Grade 0.
    - If visual requirement is unconfirmed $\to$ Grade 0.
- **Truth Transparency Notice:**  
  *Annotation was performed via programmatic catalog cross-verification, expert manual IR inspection of product images and specification sheets, and verified against `data/amazon.csv`. This benchmark did NOT employ independent double-blind crowdsourced human annotators.*

---

## 6. Leakage Prevention Audit

To ensure the benchmark does not artificially inflate retrieval metrics:
1. **No SKU / PID Leakage:** Query strings were scrubbed to ensure no internal Amazon ASINs, product IDs, or URL slugs appear in the text.
2. **Customer-Centric Phrasing:** Queries are written from a customer perspective (e.g. *"budget Android smartphone under 7000"*) rather than quoting full manufacturer catalog titles (*"Redmi A1 (Light Blue, 2GB RAM, 32GB Storage) Segment Best..."*).
3. **No Direct Copy-Pasting of `about_product`:** Query constraints were phrased using generic shopping vocabulary rather than verbatim technical strings from product bullet points.

---

## 7. Methodological Limitations

1. **Catalog Size (N = 1,465):** While representative of core consumer electronics, 1,465 products is small compared to full e-commerce catalogs (millions of SKUs). Candidate retrieval at scale may introduce more lexical noise.
2. **Single Hero Image per Product:** In the current catalog snapshot, each product is represented by its primary hero image (`img_link`). Multi-angle gallery images or customer review photos were not ingested, which can cause edge-case occlusion for side-profile attributes (e.g. cable braiding texture).
3. **Candidate Pool Ceiling ($K=20$):** In Stage 2 multimodal re-ranking, candidates were drawn from a Top-20 hybrid pool. If Stage 1 retrieval places a target item at position #22, Stage 2 cannot recover it.
4. **LLM Evaluation Variance:** While programmatic checks verify price, ID, and specs deterministically, visual and semantic claims rely on Gemini Vision as a model-based evaluator.
