import re
from typing import Dict, Any, Optional
from app.retrieval import ProductRetriever


class FilteredProductRetriever:
    """
    Metadata-Filtered Product Retriever (Phase 2.1 Experiment).
    Extracts structured constraints (price thresholds, product categories)
    from natural language queries and executes filtered dense retrieval.
    Includes an anti-false-negative fallback if filters over-constrain the search.
    """

    def __init__(self):
        self.base_retriever = ProductRetriever()
        self.client = self.base_retriever.client
        self.chroma_client = self.base_retriever.chroma_client
        self.collection = self.base_retriever.collection

    @staticmethod
    def extract_constraints(query: str) -> Dict[str, Any]:
        """
        Extract structured constraints from user query:
        - max_price: numeric upper bound
        - min_price: numeric lower bound
        - subcategory: standard category identifier in ChromaDB
        """
        q_lower = query.lower()
        constraints = {}

        # 1. Price constraint extraction
        # Matches: 'under 1500', 'below 7000', 'under ₹2000', '< 1500', 'within 3000'
        price_match = re.search(r'(?:under|below|less than|within|\<)\s*₹?\s*(\d+)', q_lower)
        if price_match:
            constraints["max_price"] = float(price_match.group(1))

        # 2. Subcategory constraint extraction (ordered from specific to general)
        if re.search(r'\b(?:hdmi)\b', q_lower):
            constraints["subcategory"] = "HDMICables"
        elif re.search(r'\b(?:cable|charger|cord|wire|lightning|type-c|type c)\b', q_lower):
            constraints["subcategory"] = "USBCables"
        elif re.search(r'\b(?:over-ear|over ear)\b', q_lower):
            constraints["subcategory"] = "Over-Ear"
        elif re.search(r'\b(?:on-ear|on ear)\b', q_lower):
            constraints["subcategory"] = "On-Ear"
        elif re.search(r'\b(?:earbuds|earphones|in-ear|in ear|airdopes|airbass)\b', q_lower):
            constraints["subcategory"] = "In-Ear"
        elif re.search(r'\b(?:smart watch|smartwatch)\b', q_lower):
            constraints["subcategory"] = "SmartWatches"
        elif re.search(r'\b(?:smartphone|android phone|mobile phone)\b', q_lower) or (
            re.search(r'\bphone\b', q_lower) and "iphone" not in q_lower
        ):
            constraints["subcategory"] = "Smartphones"
        elif re.search(r'\b(?:smart tv|tv|television|led tv)\b', q_lower):
            constraints["subcategory"] = "SmartTelevisions"
        elif re.search(r'\b(?:mouse)\b', q_lower):
            constraints["subcategory"] = "Mice"
        elif re.search(r'\b(?:keyboard)\b', q_lower):
            constraints["subcategory"] = "Keyboards"
        elif re.search(r'\b(?:mixer grinder|mixer)\b', q_lower):
            constraints["subcategory"] = "MixerGrinders"
        elif re.search(r'\b(?:kettle)\b', q_lower):
            constraints["subcategory"] = "ElectricKettles"
        elif re.search(r'\b(?:water heater|geyser)\b', q_lower):
            constraints["subcategory"] = "InstantWaterHeaters"
        elif re.search(r'\b(?:lint remover|lint shaver|fabric shaver)\b', q_lower):
            constraints["subcategory"] = "LintShavers"

        return constraints

    @staticmethod
    def build_where_clause(constraints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB compatible where clause from constraints."""
        conditions = []

        if "subcategory" in constraints:
            conditions.append({"subcategory": {"$eq": constraints["subcategory"]}})

        if "max_price" in constraints:
            conditions.append({"numeric_price": {"$lte": constraints["max_price"]}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def generate_embedding(self, text: str):
        return self.base_retriever.generate_embedding(text)

    def search(self, query: str, top_k: int = 5, use_filters: bool = True) -> Dict[str, Any]:
        """
        Execute search with or without metadata filtering.
        Includes automated fallback if filtered candidates < top_k.
        """
        query_embedding = self.generate_embedding(query)

        if not use_filters:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

        constraints = self.extract_constraints(query)
        where_clause = self.build_where_clause(constraints)

        if not where_clause:
            # No structured constraints found, proceed with dense search
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

        # Attempt filtered search
        try:
            filtered_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            retrieved_ids = filtered_results["ids"][0] if filtered_results and "ids" in filtered_results else []

            # If filtered search returned sufficient items, return it
            if len(retrieved_ids) >= top_k:
                return filtered_results

            # Anti-false-negative safeguard: backfill missing slots with unfiltered dense results
            unfiltered_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            existing_ids = set(retrieved_ids)

            # Combine without duplicates
            combined_ids = list(retrieved_ids)
            combined_docs = list(filtered_results["documents"][0])
            combined_metas = list(filtered_results["metadatas"][0])
            combined_dists = list(filtered_results.get("distances", [[]])[0])

            unf_ids = unfiltered_results["ids"][0]
            unf_docs = unfiltered_results["documents"][0]
            unf_metas = unfiltered_results["metadatas"][0]
            unf_dists = unfiltered_results.get("distances", [[]])[0] if "distances" in unfiltered_results else []

            for i, u_id in enumerate(unf_ids):
                if u_id not in existing_ids:
                    combined_ids.append(u_id)
                    combined_docs.append(unf_docs[i])
                    combined_metas.append(unf_metas[i])
                    if unf_dists:
                        combined_dists.append(unf_dists[i])
                    existing_ids.add(u_id)
                if len(combined_ids) >= top_k:
                    break

            return {
                "ids": [combined_ids[:top_k]],
                "documents": [combined_docs[:top_k]],
                "metadatas": [combined_metas[:top_k]],
                "distances": [combined_dists[:top_k]] if combined_dists else []
            }

        except Exception as e:
            # Fallback to pure dense search on query exception
            print(f"Warning: filtered query error ({e}), falling back to dense search.")
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
