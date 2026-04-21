"""
Retrieval quality metrics: Precision@K, Recall@K, MRR, NDCG@K.

All functions operate on doc_id lists (not chunk_ids) so that the eval dataset
can be written at the document level. A retrieved chunk "matches" if its doc_id
appears in the relevant_doc_ids set.

WHY NDCG over MAP:
  Mean Average Precision assumes a binary relevance judgement and penalises
  late relevant results evenly regardless of rank. NDCG uses a logarithmic
  discount so a relevant result at rank 1 is worth much more than at rank 5.
  For a RAG system where the top-1 chunk matters most (it dominates the LLM
  context), NDCG better captures the rank quality we care about.

WHY MRR alongside NDCG:
  MRR is the simplest "how fast did you find the first relevant result" metric.
  It's easy to explain: MRR=1.0 means the first result is always relevant;
  MRR=0.5 means you found it at rank 2 on average. Interviewers love it.
"""

import math


def _to_relevance_list(retrieved_doc_ids: list[str], relevant_set: set[str]) -> list[int]:
    """Binary relevance indicators: 1 if relevant, 0 otherwise."""
    return [1 if doc_id in relevant_set else 0 for doc_id in retrieved_doc_ids]


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    """
    Fraction of the top-k retrieved results that are relevant.
    P@K = |relevant ∩ retrieved[:k]| / k
    """
    if k == 0:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / k


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    """
    Fraction of relevant documents found in the top-k results.
    R@K = |relevant ∩ retrieved[:k]| / |relevant|
    """
    if not relevant_doc_ids:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank — reciprocal of the rank of the first relevant result.
    MRR = 1/rank_of_first_hit  (0.0 if no hit)
    """
    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    NDCG@K = DCG@K / IDCG@K
    Binary relevance: relevant=1, irrelevant=0.
    """
    if k == 0 or not relevant_doc_ids:
        return 0.0

    relevant_set = set(relevant_doc_ids)
    top_k = retrieved_doc_ids[:k]

    # Actual DCG
    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(
            (1 if d in relevant_set else 0 for d in top_k), start=1
        )
    )

    # Ideal DCG: all relevant docs at the top (up to k)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_all(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    k: int,
) -> dict[str, float]:
    """Compute all four metrics in one call. Returns a dict for easy logging."""
    return {
        "precision_at_k": precision_at_k(retrieved_doc_ids, relevant_doc_ids, k),
        "recall_at_k": recall_at_k(retrieved_doc_ids, relevant_doc_ids, k),
        "mrr": mrr(retrieved_doc_ids, relevant_doc_ids),
        "ndcg_at_k": ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k),
    }
