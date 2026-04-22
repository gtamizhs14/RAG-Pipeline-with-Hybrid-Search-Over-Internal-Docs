"""
FastAPI route handlers.

WHY a module-level _pipeline variable set via set_pipeline():
  FastAPI's dependency injection works best for per-request dependencies.
  RAGPipeline is expensive to construct (loads models, opens ChromaDB) — it
  should be created once at startup and reused. We store it at module level and
  set it during the lifespan event in main.py. The Depends(get_pipeline) pattern
  then hands the same instance to every request handler without rebuilding it.

WHY not use FastAPI's BackgroundTasks for citation verification:
  Verification is part of the answer's trust signal. If we run it in the
  background and return before it completes, the caller receives an answer with
  no verification status — they'd need to poll a second endpoint. For a portfolio
  project, synchronous is simpler and correct. Async verification is a Phase 5+
  optimisation if latency becomes a concern.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import ConfidenceSchema, HealthResponse, QueryRequest, QueryResponse, SourceSchema
from src.generation.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

_pipeline: RAGPipeline | None = None


def set_pipeline(pipeline: RAGPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return _pipeline


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(status="ok", pipeline_ready=_pipeline is not None)


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    Run the full RAG pipeline for a question.

    Returns the generated answer with inline citations, per-source verification
    status, and a composite confidence score.
    """
    logger.info(f"POST /query — question={req.question!r}")
    try:
        resp = pipeline.answer(
            question=req.question,
            top_k=req.top_k,
            top_n=req.top_n,
            use_reranker=req.use_reranker,
            skip_verification=req.skip_verification,
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        question=resp.question,
        answer=resp.answer,
        cited_sources=[
            SourceSchema(
                citation_number=cs.citation_number,
                source=cs.source,
                doc_id=cs.doc_id,
                chunk_id=cs.chunk_id,
                content=cs.content,
                score=cs.score,
                verified=cs.verified,
                verification_reason=cs.verification_reason,
            )
            for cs in resp.cited_sources
        ],
        confidence=ConfidenceSchema(
            retrieval_confidence=resp.confidence.retrieval_confidence,
            citation_coverage=resp.confidence.citation_coverage,
            completeness_score=resp.confidence.completeness_score,
            composite_score=resp.confidence.composite_score,
        ),
        latency_ms=resp.latency_ms,
        model=resp.model,
    )
