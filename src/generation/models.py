"""
Data models for the generation layer.

WHY separate CitedSource from SearchResult:
  SearchResult is a retrieval-layer concept — it carries chunk_id, raw score,
  and metadata. CitedSource is a generation-layer concept — it carries a
  citation number (the [1], [2] the LLM actually used in its answer) tied
  back to retrieval metadata. Keeping them separate means the retrieval
  layer has no knowledge of citation numbering, and the generation layer
  doesn't expose raw BM25/cosine scores to callers.

WHY store both cited_sources and all_sources in RAGResponse:
  cited_sources is what the user sees (only the chunks the LLM referenced).
  all_sources is what the eval framework needs (every retrieved chunk, so it
  can measure recall — were the right chunks even retrieved, even if not cited?)
  Returning both avoids re-running the pipeline twice for eval.

WHY track latency_ms at the response level:
  End-to-end latency is the most important production metric. Breaking it down
  per stage (retrieval vs generation) requires the RetrievalTrace from Phase 2.
  latency_ms on RAGResponse is the wall-clock total an API client would observe.
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
class RAGResponse:
    """
    Complete output of one RAGPipeline.answer() call.

    Fields
    ------
    question      : the original user question
    answer        : generated text with inline [n] citations
    cited_sources : only the chunks the LLM cited, in citation-number order
    all_sources   : every chunk returned by retrieval (for eval / debugging)
    latency_ms    : wall-clock milliseconds from query received to response ready
    model         : the Groq model ID that generated the answer
    """

    question: str
    answer: str
    cited_sources: list[CitedSource]
    all_sources: list[SearchResult]
    latency_ms: float
    model: str
