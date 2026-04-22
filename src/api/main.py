"""
FastAPI application entry point.

Start with:
    uvicorn src.api.main:app --reload --port 8000

WHY lifespan instead of @app.on_event("startup"):
  on_event is deprecated in FastAPI ≥ 0.93. The lifespan context manager is
  the current idiom and plays nicely with pytest's AsyncClient — both startup
  and shutdown run within the test fixture's scope.

WHY initialise RAGPipeline (and therefore ChromaDB + embedding model) at startup:
  The embedding model (~90MB) and ChromaDB client take 2–5 seconds to load.
  Loading on the first request would cause that request to time out in production.
  Startup initialisation makes the first request as fast as all subsequent ones.

WHY allow_origins=["*"] in development:
  The Streamlit UI runs on a different port (8501) and needs to call the API.
  In production, lock this down to the Streamlit container's origin or a
  configured allow-list. For a local portfolio demo, wildcard is fine.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_pipeline
from src.generation.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: initialising RAGPipeline (loads models + ChromaDB)…")
    pipeline = RAGPipeline()
    set_pipeline(pipeline)
    logger.info("Startup: pipeline ready")
    yield
    logger.info("Shutdown: releasing resources")


app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Hybrid search RAG pipeline with citation verification and confidence scoring. "
        "Uses Sentence Transformers for local embeddings, ChromaDB + BM25 for retrieval, "
        "and any OpenAI-compatible LLM (default: Groq llama-3.3-70b-versatile) for grounded generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
