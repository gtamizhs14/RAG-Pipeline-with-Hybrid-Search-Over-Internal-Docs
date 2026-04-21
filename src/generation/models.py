"""
Data models for the generation layer.

WHY separate CitedSource from SearchResult:
  SearchResult is a retrieval-layer concept — it carries chunk_id, raw score,
  and metadata. CitedSource is a generation-layer concept — it carries a
  citation number (the [1], [2] the LLM actually used in its answer) tied
  back to retrieval metadata. Keeping them separate means the retrieval
  layer has no knowledge of citation numbering, and the generation layer
  doesn't expose raw BM25/cosine scores to callers.

WHY VerifiedCitation extends CitedSource fields rather than wrapping it:
  Dataclass inheritance adds complexity without benefit here. VerifiedCitation
  is a flat struct with all CitedSource fields plus verification fields — easy
  to serialise to JSON and inspect without navigating nested objects.

WHY store both cited_sources and all_sources in RAGResponse:
  cited_sources is what the user sees (only the chunks the LLM referenced).
  all_sources is what the eval framework needs (every retrieved chunk, so it
  can measure recall — were the right chunks even retrieved, even if not cited?)
  Returning both avoids re-running the pipeline twice for eval.

WHY track latency_ms at the response level:
  End-to-end latency is the most important production metric. Breaking it down
  per stage (retrieval vs generation) requires the RetrievalTrace from Phase 2.
  latency_ms on RAGResponse is the wall-clock total an API client would observe.

WHY AnswerConfidence is a separate dataclass:
  Callers (API, UI, eval harness) often need to inspect individual score
  components to understand *why* confidence is low. A single float loses that
  signal. The composite_score is still available for simple threshold checks.
"""

from dataclasses import dataclass, field

from src.retrieval.models import SearchResult


@dataclass
class CitedSource:
    """
    A single source that the LLM explicitly cited in its answer.

    citation_number matches the [n] token the LLM wrote — callers can
    render "Sources: [1] filename.pdf p.3" from this without parsing the
    answer text themselves.
    """

    citation_number: int
    chunk_id: str
    source: str
    doc_id: str
    content: str
    score: float


@dataclass
class VerifiedCitation:
    """
    CitedSource extended with LLM-as-judge verification result.

    verified=True   → judge confirmed the chunk supports the claim
    verified=False  → judge found the chunk does NOT support the claim
    verified=None   → judge call failed (treat as unknown / uncertain)
    """

    citation_number: int
    chunk_id: str
    source: str
    doc_id: str
    content: str
    score: float
    verified: bool | None
    verification_reason: str


@dataclass
class AnswerConfidence:
    """
    Composite confidence score for a single RAG answer.

    All component scores are in [0, 1].

    retrieval_confidence : average relevance score of retrieved chunks
    citation_coverage    : fraction of citations verified as SUPPORTED
    completeness_score   : LLM-as-judge estimate of answer completeness
    composite_score      : weighted average (0.35 / 0.40 / 0.25)
    """

    retrieval_confidence: float
    citation_coverage: float
    completeness_score: float
    composite_score: float


@dataclass
class RAGResponse:
    """
    Complete output of one RAGPipeline.answer() call.

    Fields
    ------
    question        : the original user question
    answer          : generated text with inline [n] citations
    cited_sources   : cited chunks with verification status, in citation-number order
    all_sources     : every chunk returned by retrieval (for eval / debugging)
    confidence      : composite confidence score with component breakdown
    latency_ms      : wall-clock milliseconds from query received to response ready
    model           : the Groq model ID that generated the answer
    """

    question: str
    answer: str
    cited_sources: list[VerifiedCitation]
    all_sources: list[SearchResult]
    confidence: AnswerConfidence
    latency_ms: float
    model: str
