"""
Sparse retriever: keyword search using BM25 (bag-of-words, TF-IDF variant).

BM25 catches exact term matches that dense search misses — version numbers,
error codes, proper nouns, acronyms. Complements dense search via RRF fusion.

After scoring, we fetch content/metadata from ChromaDB by chunk ID so both
retrievers return the same SearchResult shape for the fusion layer.
"""

import logging
import pickle
from pathlib import Path

import numpy as np

from src.config import settings
from src.ingestion.store import DocumentStore, tokenize
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class SparseRetriever:

    def __init__(self, store: DocumentStore = None):
        # Store needed only for fetching content by ID after BM25 scoring
        self.store = store or DocumentStore()

    def _load_bm25(self) -> dict:
        p = Path(settings.bm25_index_path)
        if not p.exists():
            return {"corpus": [], "doc_ids": [], "bm25": None}
        with open(p, "rb") as f:
            return pickle.load(f)

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        data = self._load_bm25()

        if data["bm25"] is None or not data["doc_ids"]:
            logger.warning("BM25 index is empty or not built yet.")
            return []

        tokens = tokenize(query)
        scores = data["bm25"].get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_indices = [i for i in top_indices if scores[i] > 0]

        if not top_indices:
            return []

        top_ids = [data["doc_ids"][i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]

        # Fetch content and metadata from ChromaDB by chunk ID
        chroma_results = self.store.collection.get(
            ids=top_ids,
            include=["documents", "metadatas"],
        )

        id_to_doc = {}
        for cid, content, meta in zip(
            chroma_results["ids"],
            chroma_results["documents"],
            chroma_results["metadatas"],
        ):
            id_to_doc[cid] = (content, meta)

        search_results = []
        for chunk_id, score in zip(top_ids, top_scores):
            if chunk_id not in id_to_doc:
                continue
            content, meta = id_to_doc[chunk_id]
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    source=meta.get("source", ""),
                    doc_id=meta.get("doc_id", ""),
                    score=score,
                    metadata=meta,
                )
            )

        logger.debug(f"Sparse retrieved {len(search_results)} results for: {query[:60]}")
        return search_results
