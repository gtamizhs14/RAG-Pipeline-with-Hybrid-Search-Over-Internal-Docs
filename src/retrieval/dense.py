"""
Dense retriever: embeds the query and searches ChromaDB by cosine similarity.

Returns ranked candidates that will be fused with BM25 results via RRF.
ChromaDB distance = 1 - cosine_similarity, so we convert back to similarity
for a human-readable score.
"""

import logging

from src.ingestion.embedder import EmbeddingModel
from src.ingestion.store import DocumentStore
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever:

    def __init__(self, store: DocumentStore = None, embedder: EmbeddingModel = None):
        self.store = store or DocumentStore()
        self.embedder = embedder or EmbeddingModel()

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        if self.store.collection.count() == 0:
            logger.warning("ChromaDB collection is empty.")
            return []

        query_vec = self.embedder.embed_one(query)

        results = self.store.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, self.store.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, dists):
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    source=meta.get("source", ""),
                    doc_id=meta.get("doc_id", ""),
                    score=1.0 - dist,  # cosine similarity
                    metadata=meta,
                )
            )

        logger.debug(f"Dense retrieved {len(search_results)} results for: {query[:60]}")
        return search_results
