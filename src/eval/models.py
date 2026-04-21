"""
Data models for the evaluation framework.

WHY EvalSample uses doc_ids (not chunk_ids) as the relevance signal:
  Users write eval datasets at the document level ("Q about doc X"), not at
  the chunk level (which chunk inside doc X?). Chunk IDs are an implementation
  detail that changes if re-chunking strategy changes. Document IDs are stable.
  Retrieval metrics are computed by checking whether any chunk from the relevant
  doc_ids appears in the retrieved results.

WHY separate RetrievalEval and GenerationEval per sample:
  Retrieval metrics can be computed offline (no LLM needed if you have ground-
  truth relevant docs). Generation metrics require an LLM judge call. Separating
  them lets the eval runner skip generation eval when --retrieval-only is passed,
  saving API cost during rapid iteration on the retrieval stack.

WHY EvalReport stores per-sample results AND aggregates:
  Aggregates (mean Precision@5, mean faithfulness) are what you put in a README.
  Per-sample results are what you look at when debugging: "which questions did
  the retriever fail on?" The report carries both so you don't need two passes.
"""

from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """
    One (question, answer, relevant docs) triple from the eval dataset.

    Fields
    ------
    id                  : unique identifier for this sample
    question            : natural-language question
    ground_truth_answer : reference answer (used for answer relevance scoring)
    relevant_doc_ids    : document IDs that contain the answer; at least one
                          chunk from these docs should appear in retrieval results
    """

    id: str
    question: str
    ground_truth_answer: str
    relevant_doc_ids: list[str]


@dataclass
class RetrievalEval:
    """Per-sample retrieval quality metrics."""

    sample_id: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]
    hit: bool  # True if any relevant doc appeared in top-k


@dataclass
class GenerationEval:
    """
    Per-sample generation quality metrics (require LLM judge calls).

    faithfulness      : is the answer grounded in retrieved context? [0, 1]
    answer_relevance  : does the answer address the question? [0, 1]
    """

    sample_id: str
    faithfulness: float
    faithfulness_reason: str
    answer_relevance: float
    answer_relevance_reason: str
    composite_score: float  # mean of faithfulness + answer_relevance


@dataclass
class EvalResult:
    """Full evaluation result for a single sample."""

    sample: EvalSample
    retrieval: RetrievalEval
    generation: GenerationEval | None  # None if generation eval was skipped
    answer: str
    latency_ms: float
    confidence_composite: float


@dataclass
class EvalReport:
    """
    Aggregated evaluation report across all samples.

    Aggregates are means across all samples unless otherwise noted.
    """

    num_samples: int
    k: int

    # Retrieval aggregates
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    hit_rate: float  # fraction of samples with at least one relevant chunk retrieved

    # Generation aggregates (None if generation eval was skipped)
    mean_faithfulness: float | None
    mean_answer_relevance: float | None
    mean_generation_composite: float | None

    # Pipeline aggregates
    mean_confidence_composite: float
    mean_latency_ms: float

    # Per-sample breakdown for debugging
    results: list[EvalResult] = field(default_factory=list)
