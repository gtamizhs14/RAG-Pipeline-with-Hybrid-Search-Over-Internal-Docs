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

WHY the ingest endpoint saves to a temp file then calls IngestionPipeline:
  IngestionPipeline expects a file path (it reads from disk). Saving the upload
  to a NamedTemporaryFile keeps the pipeline interface unchanged and avoids
  loading the full file into memory twice. The temp file is deleted after
  ingestion regardless of success or failure.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.api.schemas import (
    ConfidenceSchema,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceSchema,
)
from src.generation.pipeline import RAGPipeline
from src.ingestion.pipeline import IngestionPipeline

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


@router.get("/documents", response_model=DocumentListResponse, tags=["ingestion"])
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)):
    """List all documents currently indexed in the vector store."""
    docs = pipeline.retriever.store.list_documents()
    stats = pipeline.retriever.store.get_stats()
    return DocumentListResponse(
        total_documents=len(docs),
        total_chunks=stats["chroma_count"],
        documents=[
            DocumentInfo(
                doc_id=d["doc_id"],
                source=d["source"],
                chunk_count=d["chunk_count"],
            )
            for d in docs
        ],
    )


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(
    file: UploadFile = File(..., description="Document to index (.txt, .md, .pdf, .html)"),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Upload and index a new document into the vector store.

    Accepts .txt, .md, .pdf, and .html files. The document is chunked,
    embedded, deduplicated, and written to ChromaDB + BM25 atomically.
    The shared DocumentStore ensures the new document is immediately
    available to all subsequent /query requests without restarting.
    """
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    allowed = {".txt", ".md", ".pdf", ".html", ".htm"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(allowed))}",
        )

    # Save upload to a temp file — IngestionPipeline expects a file path
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)

        ingestion = IngestionPipeline(
            store=pipeline.store,
            embedder=pipeline.embedder,
        )
        stats = ingestion.run(tmp_path)

        store_stats = pipeline.retriever.store.get_stats()
        return IngestResponse(
            documents_loaded=stats.documents_loaded,
            chunks_created=stats.chunks_created,
            chunks_added=stats.chunks_added,
            chunks_skipped_duplicate=stats.chunks_skipped_duplicate,
            total_chunks_in_store=store_stats["chroma_count"],
            errors=stats.errors,
        )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    Run the full RAG pipeline for a question.

    Returns the generated answer with inline citations, per-source verification
    status, and a composite confidence score.
    """
    logger.info(
        f"POST /query — question={req.question!r} mode={req.retrieval_mode}"
    )
    try:
        resp = pipeline.answer(
            question=req.question,
            top_k=req.top_k,
            top_n=req.top_n,
            use_reranker=req.use_reranker,
            skip_verification=req.skip_verification,
            dense_only=(req.retrieval_mode == "dense_only"),
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
        retrieval_mode=req.retrieval_mode,
    )
