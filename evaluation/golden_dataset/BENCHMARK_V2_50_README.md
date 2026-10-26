# Golden Dataset v2 (50 Queries): Multimodal Product Benchmark

**File:** `evaluation/golden_dataset/questions_v2_50.json`  
**CSV Companion:** `evaluation/golden_dataset/questions_v2_50.csv`  
**Status:** **FROZEN & IMMUTABLE (Phase 4 Baseline)**  
**Corpus:** `data/amazon.csv` (1,465 items, 1,351 unique catalog products)  
**Image Cache:** `data/image_cache/` (82 catalog-verified product images)  
**Validation Suite:** `evaluation/validate_benchmark_v2.py`

---

## 1. Benchmark Taxonomy & Distribution

The 50 queries are systematically constructed across a 3-dimensional matrix: **Evidence Type**, **Difficulty Level**, and **User Intent**.

### A. Evidence Type Distribution (N = 50)
| Evidence Tier | Count | % of Benchmark | Description |
| :--- | :---: | :---: | :--- |
| **`IMAGE`** | **15** | 30% | Physical geometry, dial shape, cable braiding, port layouts, or visual traits omitted or unreliably described in catalog text. |
| **`TEXT+IMAGE`** | **25** | 50% | Realistic hybrid shopping queries combining structured constraints (budget ceilings, technical specs) with visual styling. |
| **`TEXT`** | **10** | 20% | Pure technical spec control group (wattage, HDMI 2.1, 20000mAh capacity) where catalog text is comprehensive. |

### B. Difficulty Distribution (N = 50)
| Difficulty | Count | Definition & Purpose |
| :--- | :---: | :--- |
| **Easy** | **15** | Single clear constraint or unambiguous visual/spec requirement. |
| **Medium** | **15** | Dual constraint (e.g. price ceiling + specific physical affordance). |
| **Hard** | **12** | Multi-constraint queries requiring simultaneous compliance across 3+ technical, budget, and visual attributes. |
| **Adversarial** | **8** | Keyword traps, polysemy traps, or contradictions (e.g., text advertising "round dial UI" on a square screen; sandwich clamps labeled as toasters; single-dot LEDs labeled as battery displays). |

### C. Intent Type Distribution
- **Visual Appearance:** 15 queries
- **Multi-Constraint:** 18 queries
- **Feature Matching:** 7 queries
- **Price Constrained:** 6 queries
- **Brand + Feature:** 2 queries
- **Comparison:** 1 query
- **Exact Lookup:** 1 query

---

## 2. Schema Specification

Every query in `questions_v2_50.json` strictly adheres to the following JSON schema:

```json
{
  "id": "MM-026",
  "question": "Find a 2-slice pop-up bread toaster under ₹1,500 with white heat-resistant plastic housing and browning dial.",
  "evidence_type": "TEXT+IMAGE",
  "difficulty": "medium",
  "intent_type": "multi_constraint",
  "category": "Home & Kitchen",
  "constraints": {
    "price_max": 1500,
    "category": "toaster",
    "visual": ["white plastic housing", "front browning dial", "top pop-up slots"]
  },
  "relevant_products": [
    {
      "product_id": "B06XPYRWV5",
      "product_name": "Pigeon by Stovekraft 2 Slice Auto Pop up Toaster. A Smart Bread Toaster (White)",
      "relevance": 3
    },
    {
      "product_id": "B0073QGKAS",
      "product_name": "Bajaj ATX 4 750-Watt Pop-up Toaster (White)",
      "relevance": 3
    },
    {
      "product_id": "B00935MGHS",
      "product_name": "Prestige PSMFB 800 Watt Sandwich Toaster with Fixed Plates, Black",
      "relevance": 0
    }
  ],
  "required_evidence": {
    "price": "text",
    "housing_color": "image",
    "control_dial": "image",
    "slot_type": "image"
  }
}
```

---

## 3. Graded Relevance Scale

- **Grade 3 (Ideal / Target Match):** Completely satisfies all hard numerical constraints, category boundaries, technical specifications, and visual traits.
- **Grade 2 (Relevant / High Quality Alternative):** Satisfies all primary constraints with minor non-blocking aesthetic variation (e.g., matching category and specs with a slightly different colorway).
- **Grade 1 (Partially Relevant / Marginal):** Matches high-level category but violates a secondary constraint (e.g., on-ear headphones when over-ear was requested, or budget exceeded by < 10%).
- **Grade 0 (Irrelevant / Adversarial Trap):** Violates category boundaries, hard budget limits, or visual requirements (e.g., sandwich maker returned for pop-up toaster query; rectangular smartwatch returned for round dial query).

---

## 4. Benchmark Verification Status

All 50 queries passed the automated verification suite (`evaluation/validate_benchmark_v2.py`):
1. **Catalog Existence:** 100% of the 82 referenced unique `product_id`s exist in `data/amazon.csv`.
2. **Price Constraint Integrity:** 0 violations. All Grade 3 products strictly satisfy `discounted_price <= price_max`.
3. **Image Cache Integrity:** All 82 product images exist in `data/image_cache/` and are verified valid image files.
4. **Leakage Audit:** Queries are phrased from the customer perspective without leaking internal catalog SKUs or IDs.
5. **Freeze Policy:** This dataset is now **frozen**. Any future retrieval comparison (Dense, Metadata, Hybrid RRF, Cross-Encoder, Multimodal) or Generation Evaluation must run against these exact 50 questions.
