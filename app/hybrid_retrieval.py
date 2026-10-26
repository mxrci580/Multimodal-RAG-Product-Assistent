import re
from typing import Dict, List, Any, Optional
from rank_bm25 import BM25Okapi
from app.retrieval import ProductRetriever


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for lexical matching."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", str(text).lower())


class BM25Retriever:
    """
    Lexical BM25 retriever built on the ChromaDB document corpus.
    Fast exact term matching without requiring GPU/vector inference.
    """

    def __init__(self, collection=None):
        if collection is None:
            self.base_retriever = ProductRetriever()
            self.collection = self.base_retriever.collection
        else:
            self.collection = collection

        # Load all documents once from ChromaDB
        data = self.collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

        self.id_to_index = {pid: idx for idx, pid in enumerate(self.ids)}
        self.tokenized_corpus = [tokenize_text(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Return top_k document matches by BM25 score."""
        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "rank": rank,
                "id": self.ids[idx],
                "score": float(scores[idx]),
                "document": self.documents[idx],
                "metadata": self.metadatas[idx]
            })
        return results


class HybridProductRetriever:
    """
    Hybrid Retriever (Dense + BM25 with Reciprocal Rank Fusion).
    Combines semantic embeddings with exact keyword BM25 retrieval.
    RRF Formula: score(d) = sum(1 / (rrf_k + rank(d)))
    """

    def __init__(self, rrf_k: int = 60, dense_candidates: int = 20, bm25_candidates: int = 20):
        self.dense_retriever = ProductRetriever()
        self.collection = self.dense_retriever.collection
        self.bm25_retriever = BM25Retriever(collection=self.collection)
        self.rrf_k = rrf_k
        self.dense_candidates = dense_candidates
        self.bm25_candidates = bm25_candidates

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
        rrf_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute hybrid search:
        1. Fetch top dense_k candidates from ChromaDB
        2. Fetch top bm25_k candidates from BM25
        3. Merge with Reciprocal Rank Fusion (RRF)
        4. Return top_k fused results
        """
        d_k = dense_k or self.dense_candidates
        b_k = bm25_k or self.bm25_candidates
        r_k = rrf_k or self.rrf_k

        # 1. Dense retrieval
        dense_res = self.dense_retriever.search(query, top_k=d_k)
        dense_ids = dense_res["ids"][0] if dense_res and "ids" in dense_res else []
        dense_metas = dense_res["metadatas"][0] if dense_res and "metadatas" in dense_res else []
        dense_docs = dense_res["documents"][0] if dense_res and "documents" in dense_res else []

        # 2. BM25 retrieval
        bm25_res = self.bm25_retriever.search(query, top_k=b_k)

        # Lookup caches for metadata and documents
        doc_lookup = {}
        meta_lookup = {}
        for pid, doc, meta in zip(dense_ids, dense_docs, dense_metas):
            doc_lookup[pid] = doc
            meta_lookup[pid] = meta
        for item in bm25_res:
            pid = item["id"]
            if pid not in doc_lookup:
                doc_lookup[pid] = item["document"]
                meta_lookup[pid] = item["metadata"]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        dense_rank_map: Dict[str, int] = {}
        bm25_rank_map: Dict[str, int] = {}

        for rank, pid in enumerate(dense_ids, start=1):
            dense_rank_map[pid] = rank
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (r_k + rank))

        for item in bm25_res:
            pid = item["id"]
            rank = item["rank"]
            bm25_rank_map[pid] = rank
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (r_k + rank))

        # Sort by final RRF score descending
        ranked_pids = sorted(rrf_scores.keys(), key=lambda p: rrf_scores[p], reverse=True)[:top_k]

        final_ids = ranked_pids
        final_docs = [doc_lookup[p] for p in ranked_pids]
        final_metas = []
        for p in ranked_pids:
            meta = dict(meta_lookup[p])
            # Store retrieval diagnostic provenance
            meta["rrf_score"] = rrf_scores[p]
            meta["dense_rank"] = dense_rank_map.get(p)
            meta["bm25_rank"] = bm25_rank_map.get(p)
            final_metas.append(meta)

        return {
            "ids": [final_ids],
            "documents": [final_docs],
            "metadatas": [final_metas],
            "scores": [[rrf_scores[p] for p in ranked_pids]]
        }
