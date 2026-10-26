from typing import Dict, List, Any, Optional
from flashrank import Ranker, RerankRequest
from app.hybrid_retrieval import HybridProductRetriever


class RerankedProductRetriever:
    """
    Model D: Two-Stage Hybrid + Cross-Encoder Reranker.
    Stage 1 (Retrieval): Dense (top 20) + BM25 (top 20) merged via RRF (k=60) -> Candidate Pool of 20.
    Stage 2 (Reranking): Cross-Encoder evaluates full query-document attention over candidate pool -> Top 5.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        candidate_pool_k: int = 20,
        rrf_k: int = 60
    ):
        self.candidate_pool_k = candidate_pool_k
        self.rrf_k = rrf_k
        self.hybrid_retriever = HybridProductRetriever(
            rrf_k=rrf_k,
            dense_candidates=candidate_pool_k,
            bm25_candidates=candidate_pool_k
        )
        import os
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models_cache"))
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    @staticmethod
    def format_passage(doc: str, meta: Dict[str, Any]) -> str:
        """Format candidate metadata and text for cross-encoder attention."""
        name = meta.get("product_name", "")
        cat = meta.get("category", "")
        price = meta.get("price", "")
        # Compact, information-dense passage representation
        return f"{name}. Category: {cat}. Price: {price}. {str(doc)[:350]}"

    def search(self, query: str, top_k: int = 5, candidate_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute two-stage retrieval:
        1. Retrieve candidate pool from Hybrid RRF
        2. Rerank candidate pool using Cross-Encoder
        3. Return top_k reranked products
        """
        c_k = candidate_k or self.candidate_pool_k

        # Stage 1: Hybrid candidate retrieval
        hybrid_res = self.hybrid_retriever.search(
            query=query,
            top_k=c_k,
            dense_k=c_k,
            bm25_k=c_k,
            rrf_k=self.rrf_k
        )

        pids = hybrid_res["ids"][0] if hybrid_res and "ids" in hybrid_res else []
        docs = hybrid_res["documents"][0] if hybrid_res and "documents" in hybrid_res else []
        metas = hybrid_res["metadatas"][0] if hybrid_res and "metadatas" in hybrid_res else []

        if not pids:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "scores": [[]]}

        # Stage 2: Cross-Encoder Reranking
        passages = []
        meta_by_id = {}
        doc_by_id = {}
        for rrf_rank, (pid, doc, meta) in enumerate(zip(pids, docs, metas), start=1):
            meta_copy = dict(meta)
            meta_copy["rrf_rank"] = rrf_rank
            meta_by_id[pid] = meta_copy
            doc_by_id[pid] = doc
            passages.append({
                "id": pid,
                "text": self.format_passage(doc, meta_copy)
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)[:top_k]

        final_ids = []
        final_docs = []
        final_metas = []
        final_scores = []

        for rerank_rank, item in enumerate(reranked_results, start=1):
            pid = item["id"]
            score = float(item["score"])
            m = meta_by_id[pid]
            m["cross_encoder_score"] = score
            m["rerank_rank"] = rerank_rank

            final_ids.append(pid)
            final_docs.append(doc_by_id[pid])
            final_metas.append(m)
            final_scores.append(score)

        return {
            "ids": [final_ids],
            "documents": [final_docs],
            "metadatas": [final_metas],
            "scores": [final_scores]
        }
