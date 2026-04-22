"""
Pydantic request/response schemas for the FastAPI layer.

WHY separate schemas from generation models:
  RAGResponse is an internal dataclass — it carries SearchResult objects and
  VerifiedCitation objects that reference internal types. The API schemas are
  public contracts: flat, JSON-serialisable, versioned. Keeping them separate
  means we can evolve the internal pipeline without breaking API consumers, and
  vice versa.

WHY include confidence breakdown in QueryResponse:
  Callers (the Streamlit UI, future integrations) need individual score
  components to render a meaningful confidence indicator. A single composite
  float hides whether low confidence came from poor retrieval or a hallucinating
  LLM — the breakdown tells the operator which lever to pull.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int | None = Field(None, ge=1, le=100, description="Candidates per retriever")
    top_n: int | None = Field(None, ge=1, le=20, description="Final chunks after reranking")
    use_reranker: bool | None = Field(None, description="Override reranker toggle")
    skip_verification: bool = Field(False, description="Skip LLM-as-judge citation check (faster)")


class SourceSchema(BaseModel):
    citation_number: int
    source: str
    doc_id: str
    chunk_id: str
    content: str
    score: float
    verified: bool | None
    verification_reason: str


class ConfidenceSchema(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    completeness_score: float
    composite_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    cited_sources: list[SourceSchema]
    confidence: ConfidenceSchema
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
