"""
HybridRetriever — the single entry point for all retrieval queries.

Full pipeline:
  1. Dense retrieval  (ChromaDB cosine similarity) → top_k candidates
  2. Sparse retrieval (BM25 keyword matching)      → top_k candidates
  3. RRF fusion                                    → top_k merged, deduplicated
  4. Cross-encoder rerank (optional)               → top_n final

WHY share DocumentStore and EmbeddingModel between dense and sparse:
  DocumentStore holds the ChromaDB connection. Opening it twice would create
  two separate client objects pointing at the same files — wasteful and
  potentially problematic on Windows where file locking can be strict.
  Injecting one shared instance avoids both issues.

WHY expose retrieve_with_trace() separately:
  The eval framework (Phase 4) needs intermediate counts (how many dense,
  sparse, fused, and final results) to diagnose retrieval quality. Rather
  than cluttering retrieve() with optional output parameters, we offer a
  dedicated method that returns a RetrievalTrace dataclass. This keeps the
  hot path lean and the diagnostic path explicit.

WHY allow per-call overrides for top_k, top_n, use_reranker:
  The eval framework will call retrieve() with reranking disabled to isolate
  fusion quality from reranking quality. API endpoints may want a different
  top_n depending on the UI context. .env defaults cover 99% of cases; the
  overrides handle the rest without code changes.
"""

import logging
from dataclasses import dataclass

from src.config import settings
from src.ingestion.embedder import EmbeddingModel
from src.ingestion.store import DocumentStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import SearchResult
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTrace:
    """
    Diagnostic snapshot of a single retrieval run.

    Used by the evaluation framework to measure retrieval quality at each stage
    without re-running the pipeline multiple times.
    """

    query: str
    dense_count: int
    sparse_count: int
    fused_count: int
    final_count: int
    reranked: bool
    results: list[SearchResult]


class HybridRetriever:

    def __init__(
        self,
        store: DocumentStore = None,
        embedder: EmbeddingModel = None,
        use_reranker: bool = None,
    ):
        # Share one store and one embedder across all sub-retrievers to avoid
        # duplicate ChromaDB connections and duplicate model loads.
        shared_store = store or DocumentStore()
        shared_embedder = embedder or EmbeddingModel()

        self.dense = DenseRetriever(store=shared_store, embedder=shared_embedder)
        self.sparse = SparseRetriever(store=shared_store)
        self.reranker = CrossEncoderReranker()

        # Instance-level default; can be overridden per call
        self._use_reranker = (
            use_reranker if use_reranker is not None else settings.use_reranker
        )

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        top_n: int = None,
        use_reranker: bool = None,
    ) -> list[SearchResult]:
        """
        Run the full hybrid retrieval pipeline and return final results.

        Parameters
        ----------
        query       : natural language question or search string
        top_k       : candidates fetched from each retriever (pre-fusion count)
        top_n       : final results returned after reranking
        use_reranker: override the instance default for this call only
        """
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        top_n = top_n if top_n is not None else settings.rerank_top_n
        do_rerank = use_reranker if use_reranker is not None else self._use_reranker

        # ── Stage 1: candidate retrieval ─────────────────────────────────────
        # Both retrievers run sequentially here. In a high-traffic production
        # system these would run in parallel (asyncio or ThreadPoolExecutor)
        # since they touch different indexes. For a portfolio demo, sequential
        # is simpler and the latency difference is negligible (<50ms on local).
        dense_results = self.dense.retrieve(query, top_k=top_k)
        sparse_results = self.sparse.retrieve(query, top_k=top_k)

        logger.info(
            f"Stage 1 — dense: {len(dense_results)}, sparse: {len(sparse_results)}"
        )

        # ── Stage 2: RRF fusion ───────────────────────────────────────────────
        # Keep top_k candidates going into the reranker so the reranker has a
        # generous shortlist to choose from. top_n cut happens in stage 3.
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_n=top_k,
        )

        logger.info(f"Stage 2 — fused: {len(fused)} unique candidates")

        # ── Stage 3: cross-encoder rerank ────────────────────────────────────
        if do_rerank and fused:
            final = self.reranker.rerank(query, fused, top_n=top_n)
        else:
            # Reranker disabled: take the top_n from RRF ranking directly
            final = fused[:top_n]

        logger.info(f"Stage 3 — final: {len(final)} results (reranked={do_rerank})")
        return final

    def retrieve_with_trace(
        self,
        query: str,
        top_k: int = None,
        top_n: int = None,
        use_reranker: bool = None,
    ) -> RetrievalTrace:
        """
        Same pipeline as retrieve() but also returns intermediate counts.

        The RetrievalTrace lets the eval framework measure:
          - Did dense and sparse retrieve enough candidates?
          - Did RRF deduplicate effectively (fused_count < dense + sparse)?
          - What was the final precision after reranking?
        """
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        top_n = top_n if top_n is not None else settings.rerank_top_n
        do_rerank = use_reranker if use_reranker is not None else self._use_reranker

        dense_results = self.dense.retrieve(query, top_k=top_k)
        sparse_results = self.sparse.retrieve(query, top_k=top_k)

        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_n=top_k,
        )

        if do_rerank and fused:
            final = self.reranker.rerank(query, fused, top_n=top_n)
        else:
            final = fused[:top_n]

        return RetrievalTrace(
            query=query,
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            fused_count=len(fused),
            final_count=len(final),
            reranked=do_rerank,
            results=final,
        )
