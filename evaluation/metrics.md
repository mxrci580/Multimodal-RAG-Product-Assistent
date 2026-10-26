# Formal Metrics & Mathematical Formulations

This document specifies the exact mathematical definitions, thresholds, and operational implementations of all retrieval and generation metrics used throughout the evaluation suite.

---

## 1. Information Retrieval (IR) Metrics

Let $Q$ be the set of queries ($|Q| = N = 50$). For each query $q \in Q$, let $(d_1, d_2, \dots, d_K)$ be the ranked list of retrieved documents, and let $rel(d_i) \in \{0, 1, 2, 3\}$ denote the graded relevance score of document $d_i$.

### A. Hit Rate @ K (HR@K)
Binary indicator of whether at least one relevant product ($rel \ge 2$) appears within the top $K$ retrieved positions:

$$\text{Hit@K}(q) = \mathbb{I}\left(\max_{i \in \{1, \dots, K\}} rel(d_i) \ge 2\right)$$

$$\text{HR@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{Hit@K}(q)$$

- **HR@1:** Measures the precision of the single top recommendation.
- **HR@5:** Measures candidate availability in the top 5 results.

### B. Ideal Match @ K (Ideal@K)
Binary indicator of whether the single best / target product ($rel = 3$) appears within the top $K$ positions:

$$\text{Ideal@K}(q) = \mathbb{I}\left(\max_{i \in \{1, \dots, K\}} rel(d_i) = 3\right)$$

$$\text{Ideal@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{Ideal@K}(q)$$

### C. Precision @ K (P@K)
The proportion of top $K$ retrieved products that meet the relevance threshold ($rel \ge 2$):

$$\text{Precision@K}(q) = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(rel(d_i) \ge 2)$$

$$\text{P@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{Precision@K}(q)$$

### D. Mean Reciprocal Rank (MRR)
Measures the reciprocal rank of the first relevant document ($rel \ge 2$):

$$\text{RR}(q) = \begin{cases} \frac{1}{\min \{i \mid rel(d_i) \ge 2\}}, & \text{if } \exists i \le K \text{ with } rel(d_i) \ge 2 \\ 0, & \text{otherwise} \end{cases}$$

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)$$

### E. Normalized Discounted Cumulative Gain @ K (NDCG@K)
Measures the ranking quality of graded relevance, penalizing relevant items appearing lower in the ranking.

1. **Discounted Cumulative Gain (DCG@K):**
   $$\text{DCG@K}(q) = \sum_{i=1}^K \frac{2^{rel(d_i)} - 1}{\log_2(i + 1)}$$

2. **Ideal Discounted Cumulative Gain (IDCG@K):**
   Let $\pi^*$ be the sorting permutation that arranges all available ground-truth relevant products in descending order of relevance ($rel_{\pi^*(1)} \ge rel_{\pi^*(2)} \ge \dots$):
   $$\text{IDCG@K}(q) = \sum_{i=1}^{\min(K, |R_q|)} \frac{2^{rel_{\pi^*(i)}} - 1}{\log_2(i + 1)}$$

3. **Normalized DCG (NDCG@K):**
   $$\text{NDCG@K}(q) = \begin{cases} \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)}, & \text{if } \text{IDCG@K}(q) > 0 \\ 0, & \text{otherwise} \end{cases}$$

   $$\text{NDCG@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{NDCG@K}(q)$$

---

## 2. Generation & Answer Quality Metrics

For each query $q$, the generator produces a structured recommendation response consisting of:
- Recommended product ID $d^* \in \text{Catalog}$
- Stated price $p^*$
- Natural language explanation / rationale
- A decomposed set of $M$ atomic factual claims: $C = \{c_1, c_2, \dots, c_M\}$

### A. Answer Correctness
Evaluates whether the primary recommended product $d^*$ satisfies the user's requirements, mapped directly to ground-truth relevance grades:

$$\text{Correctness}(q) = \begin{cases} 
1.00, & \text{if } rel(d^*) = 3 \quad (\text{Ideal Match}) \\
0.67, & \text{if } rel(d^*) = 2 \quad (\text{Relevant Alternative}) \\
0.33, & \text{if } rel(d^*) = 1 \quad (\text{Partially Relevant}) \\
0.00, & \text{if } rel(d^*) = 0 \quad (\text{Irrelevant or Unlisted Failure})
\end{cases}$$

### B. Claim-Level Faithfulness / Groundedness
Let $C_q = \{c_1, c_2, \dots, c_M\}$ be the atomic claims in the generated answer. Each claim $c_j$ is evaluated for factual support against the retrieved evidence (catalog text specs and verified image pixels):

$$\text{Supported}(c_j) = \begin{cases} 1, & \text{if } c_j \text{ is factually verified by the retrieved evidence} \\ 0, & \text{if } c_j \text{ is ungrounded, fabricated, or contradicts evidence} \end{cases}$$

$$\text{Faithfulness}(q) = \frac{1}{|C_q|} \sum_{j=1}^{|C_q|} \text{Supported}(c_j)$$

### C. Hallucination Rate
The proportion of factual claims made by the model that are unsupported or contradicted by the evidence:

$$\text{Hallucination Rate}(q) = 1.0 - \text{Faithfulness}(q) = \frac{1}{|C_q|} \sum_{j=1}^{|C_q|} (1 - \text{Supported}(c_j))$$

Specific hallucination categories tracked:
- **Price Hallucination:** Model quotes an incorrect price ($|p^* - p_{\text{catalog}}| > ₹50$).
- **Visual Hallucination:** Model asserts physical geometry (e.g. circular dial, braided wire) not present in the product image.
- **Spec Hallucination:** Model asserts technical features (e.g. ANC, Type-C) not documented in catalog text.

### D. Multi-Constraint Completeness
Let $F_q = \{f_1, f_2, \dots, f_T\}$ be the set of $T$ structured constraint facets defined for query $q$ (e.g. budget ceiling, brand, specs, physical form factor):

$$\text{Satisfied}(f_k) = \begin{cases} 1, & \text{if the recommended product } d^* \text{ satisfies facet } f_k \\ 0, & \text{otherwise} \end{cases}$$

$$\text{Completeness}(q) = \frac{1}{|F_q|} \sum_{k=1}^{|F_q|} \text{Satisfied}(f_k)$$

### E. Evidence Attribution Accuracy
Evaluates whether the generator correctly associates its claims with their appropriate source of evidence:

$$\text{Evidence Attribution}(q) = \frac{\text{Correctly Attributed Claims}}{|C_q|}$$

- Physical geometry, dial shape, wire braiding, stem shape, color, LED illumination ring $\to$ Must be attributed to `"product_image"`.
- Price, wattage, model name, warranty, Bluetooth version, battery capacity $\to$ Must be attributed to `"catalog_text"`.

---

## 3. Statistical Testing Formulations

### A. McNemar's Test (Exact Binomial for Paired Binary Outcomes)
Used for paired binary query outcomes (e.g. $\text{Hit@1}_{\text{Base}}(q)$ vs $\text{Hit@1}_{\text{New}}(q)$):

$$\begin{array}{c|cc}
& \text{New Succeeded (1)} & \text{New Failed (0)} \\
\hline
\text{Base Succeeded (1)} & a & c \\
\text{Base Failed (0)} & b & d
\end{array}$$

Discordant pairs:
- $b$: Base failed, New succeeded (New win)
- $c$: Base succeeded, New failed (Base win)

Under null hypothesis $H_0: p(b) = p(c) = 0.5$, the exact binomial test statistic is:

$$p = 2 \times \min\left(\sum_{k=0}^b \binom{b+c}{k} 0.5^{b+c}, \sum_{k=b}^{b+c} \binom{b+c}{k} 0.5^{b+c}\right)$$

### B. Wilcoxon Signed-Rank Test (Paired Continuous / Graded Outcomes)
Used for paired differences $D_q = X_{\text{New}}(q) - X_{\text{Base}}(q)$ in continuous metrics (NDCG, MRR, Faithfulness, Correctness).

1. Exclude zero differences ($D_q \ne 0$).
2. Rank absolute differences $|D_q|$: $R_q = \text{rank}(|D_q|)$.
3. Sum of positive ranks $W^+$ and negative ranks $W^-$:
   $$W^+ = \sum_{D_q > 0} R_q, \quad W^- = \sum_{D_q < 0} R_q$$
   Test statistic: $W = \min(W^+, W^-)$.
4. Rank-biserial correlation effect size:
   $$r = \frac{W^+ - W^-}{W^+ + W^-} \in [-1, +1]$$

### C. Bootstrap Confidence Intervals
10,000 paired resamples with replacement $\mathcal{S}_b = \{D_1^*, \dots, D_N^*\}$. The 95% confidence interval is the 2.5th and 97.5th percentiles of the bootstrap distribution:

$$\text{CI}_{95\%} = [\text{Percentile}(\bar{D}^*, 2.5\%), \text{Percentile}(\bar{D}^*, 97.5\%)]$$
