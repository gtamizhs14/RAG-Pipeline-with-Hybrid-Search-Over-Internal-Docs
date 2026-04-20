"""
Reciprocal Rank Fusion (RRF) — merges dense and sparse ranked lists.

WHY RRF instead of a weighted average of raw scores:
  BM25 scores and cosine similarities live on completely different scales.
  BM25 can return 14.3 for a good match; cosine returns 0.87. A naive
  weighted average would let whichever retriever produces bigger numbers
  dominate — regardless of semantic quality.

  RRF uses RANK POSITION, not raw scores. Each document's contribution is
  1 / (k + rank). This is scale-invariant: rank 1 from BM25 and rank 1
  from dense search contribute equally, regardless of their raw scores.

WHY k=60:
  Standard constant from Cormack, Clarke & Buettcher (SIGIR 2009).
  Empirically tuned to prevent rank-1 results from completely dominating.
  At k=60, the difference between rank 1 and rank 2 is 1/61 vs 1/62 (~1.6%),
  which is meaningful but not overwhelming. Lower k → rank 1 dominates more.
  Higher k → all ranks treated nearly equally (diminishing returns).

WHY weights on top of RRF:
  The spec calls for 0.7 dense / 0.3 sparse. For Q&A over prose documents,
  dense semantic search usually matters more. For technical docs with version
  numbers and error codes, you'd increase sparse weight. Making it configurable
  in .env lets you tune per corpus without code changes.
"""

import logging

from src.config import settings
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = None,
    dense_weight: float = None,
    sparse_weight: float = None,
    top_n: int = None,
) -> list[SearchResult]:
    """
    Merge two ranked lists using weighted RRF.

    Each document's score = sum over lists of: weight_i / (k + rank_i).
    Documents appearing in only one list still get scored; they just miss
    the contribution from the other list.

    Returns up to top_n SearchResult objects sorted by descending RRF score.
    The .score field is replaced with the composite RRF score so downstream
    code sees a single unified ranking.
    """
    k = k if k is not None else settings.rrf_k
    dense_weight = dense_weight if dense_weight is not None else settings.dense_weight
    sparse_weight = sparse_weight if sparse_weight is not None else settings.sparse_weight
    top_n = top_n if top_n is not None else settings.retrieval_top_k

    # Build a unified lookup so we can recover content/metadata for any chunk_id.
    # Dense result takes priority if the same id somehow appears in both (shouldn't
    # happen since both retrievers pull from the same underlying store, but safe).
    all_results: dict[str, SearchResult] = {}
    for r in dense_results:
        all_results[r.chunk_id] = r
    for r in sparse_results:
        if r.chunk_id not in all_results:
            all_results[r.chunk_id] = r

    # Accumulate weighted RRF contribution from each list.
    # Initialise to 0.0 so chunks only in one list still appear in the dict.
    rrf_scores: dict[str, float] = {chunk_id: 0.0 for chunk_id in all_results}

    for rank, result in enumerate(dense_results, start=1):
        rrf_scores[result.chunk_id] += dense_weight / (k + rank)

    for rank, result in enumerate(sparse_results, start=1):
        rrf_scores[result.chunk_id] += sparse_weight / (k + rank)

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_n]

    fused: list[SearchResult] = []
    for chunk_id in sorted_ids:
        r = all_results[chunk_id]
        fused.append(
            SearchResult(
                chunk_id=r.chunk_id,
                content=r.content,
                source=r.source,
                doc_id=r.doc_id,
                score=rrf_scores[chunk_id],
                metadata=r.metadata,
            )
        )

    logger.debug(
        f"RRF fused {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"→ {len(fused)} unique candidates"
    )
    return fused
