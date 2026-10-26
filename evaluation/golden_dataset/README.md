# Golden Dataset for Retrieval Recall & Accuracy Evaluation

This directory contains curated evaluation questions to measure the retrieval performance of the Product Recommendation RAG system against [`data/amazon.csv`](../../data/amazon.csv).

## Files
- `questions.json`: Machine-readable golden test dataset with full metadata.
- `questions.csv`: Tabular version of the golden test dataset.

## Dataset Schema

Each question record includes:
- `id`: Unique identifier (e.g., `Q01`, `Q02`).
- `question`: The realistic search query or user intent.
- `category`: Product category (`Audio & Earbuds`, `Smartwatches`, `Smartphones`, `Cables & Accessories`, `Computers & Peripherals`, `Smart TVs`, `Home & Kitchen`).
- `query_type`: Intent pattern:
  - `exact_product`: Precise product name lookups.
  - `brand_category`: Brand name + product class (e.g., "boAt wired earphones").
  - `brand_feature`: Brand + specific specification/feature.
  - `feature_search`: Spec/feature without brand name (e.g., "60W fast charging braided type C cable").
  - `price_constraint`: Budget-filtered search (e.g., "best bluetooth earbuds under 1500").
  - `intent_usecase`: Functional purpose (e.g., "fast charging cable for iPhone lightning").
- `expected_product_ids`: Target Amazon ASIN(s) present in ChromaDB.
- `ground_truth`: Primary expected product title snippet.
- `keywords`: Distinctive matching tokens.

## Evaluation Metrics

1. **Recall@K**: Proportion of ground-truth relevant products retrieved within top $K$ results:
   $$\text{Recall}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$
2. **Hit Rate@K**: Proportion of queries where at least one ground-truth product appears in top $K$ ($1$ if hit, $0$ otherwise).
3. **MRR (Mean Reciprocal Rank)**: Measures how high the first relevant result is ranked:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
