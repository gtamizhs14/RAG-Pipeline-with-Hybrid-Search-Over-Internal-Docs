"""
Cross-encoder reranker — precision pass after RRF fusion.

WHY a two-stage retrieval pipeline:

  Stage 1 (RECALL): bi-encoders (dense) + keyword (sparse) → top-K candidates
    - Bi-encoder: query and document are embedded INDEPENDENTLY, then compared
      by dot product. Fast because you embed the corpus once and store vectors.
      Weak because query and document never attend to each other — the model
      can't see "does this specific sentence answer this specific question?"

  Stage 2 (PRECISION): cross-encoder → top-N final
    - Cross-encoder: concatenates [query, doc] into ONE input and runs full
      transformer attention across both. It can see every word of the query
      while reading every word of the document.
    - Dramatically better at "does this passage directly answer the question?"
    - Slow: O(candidates) inference at query time, so you can only run it on
      the shortlist (20 candidates → 5 final), never the whole corpus.

WHY this model (ms-marco-MiniLM-L-6-v2):
  Trained on MS MARCO passage ranking, a 500k+ query-passage relevance dataset
  from real Bing search logs. Best balance of accuracy and inference speed for
  passage re-ranking. Runs on CPU in ~50ms for 20 candidates.
  Configurable via RERANKER_MODEL in .env if you want a larger/different model.

WHY cached_property for the model:
  Loading a transformer from disk takes ~1 second. cached_property loads it
  once on first use and reuses the same instance for all subsequent queries
  in the same process — avoids paying the startup cost on every request.
"""

import logging
from functools import cached_property

from src.config import settings
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model

    @cached_property
    def _model(self):
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading cross-encoder: {self.model_name}")
        # activation_fct=None → raw logits. The relative order is all we need;
        # applying sigmoid would squash scores but wouldn't change the ranking.
        model = CrossEncoder(self.model_name)
        logger.info("Cross-encoder ready")
        return model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = None,
    ) -> list[SearchResult]:
        """
        Score every (query, document) pair and return top_n by descending score.

        The .score field on returned results holds the cross-encoder logit,
        replacing the RRF composite score from the previous stage.
        """
        if not results:
            return []

        top_n = top_n if top_n is not None else settings.rerank_top_n

        # CrossEncoder.predict takes a list of [query, passage] pairs.
        # Sending all pairs in one call lets the model batch internally
        # (much faster than calling predict() per pair).
        pairs = [[query, r.content] for r in results]
        raw_scores = self._model.predict(pairs)

        # zip scores back to results, sort descending, take top_n
        scored = sorted(
            zip(raw_scores, results),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        reranked: list[SearchResult] = []
        for score, result in scored[:top_n]:
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    source=result.source,
                    doc_id=result.doc_id,
                    score=float(score),
                    metadata=result.metadata,
                )
            )

        logger.debug(
            f"Cross-encoder reranked {len(results)} candidates → {len(reranked)} final"
        )
        return reranked
