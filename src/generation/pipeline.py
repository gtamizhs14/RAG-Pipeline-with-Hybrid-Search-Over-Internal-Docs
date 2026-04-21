"""
RAGPipeline — end-to-end orchestration: retrieve → prompt → generate → cite.

Full pipeline:
  1. HybridRetriever    → top_n ranked chunks (Phase 2)
  2. PromptBuilder      → numbered context block + grounded system prompt
  3. GroqClient         → LLM generation with inline [n] citations
  4. CitationParser     → maps [n] tokens back to source chunks
  5. RAGResponse        → structured answer + attribution returned to caller

WHY inject dependencies (store, embedder, groq_client) instead of constructing them:
  FastAPI (Phase 5) will create one RAGPipeline at startup and reuse it across
  requests. If we constructed DocumentStore inside __init__ every time, every
  test instantiation would open a new ChromaDB connection. Injection lets tests
  pass lightweight fakes and lets the API layer share one warm instance.

WHY measure wall-clock latency here rather than in the API layer:
  The API layer adds network overhead (serialisation, HTTP). Measuring inside
  RAGPipeline gives us a clean "RAG work" number that's comparable across
  different API frameworks, useful for Phase 4 evaluation.

WHY not catch exceptions in answer():
  Explicit is better than silent. If retrieval fails (ChromaDB unreachable) or
  Groq raises an auth error, the exception should propagate to the caller (API
  layer or CLI) which can decide how to surface it to the user. Swallowing
  exceptions here would hide bugs behind empty responses.

Interview question this answers:
  "Walk me through how a question becomes an answer in your RAG pipeline."
  Answer: The question hits HybridRetriever (dense + sparse → RRF → reranker,
  Phase 2). The top-5 chunks are numbered [1]–[5] and injected into a grounded
  system prompt. Groq's llama3-70b generates an answer citing those numbers
  inline. A regex parser extracts the cited numbers and maps them back to source
  metadata. The caller receives a RAGResponse with the answer, cited sources,
  and all retrieval results for auditability.
"""

import logging
import time

from src.config import settings
from src.generation.citations import CitationParser
from src.generation.groq_client import GroqClient
from src.generation.models import RAGResponse
from src.generation.prompt import SYSTEM_PROMPT, PromptBuilder
from src.ingestion.embedder import EmbeddingModel
from src.ingestion.store import DocumentStore
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(
        self,
        store: DocumentStore = None,
        embedder: EmbeddingModel = None,
        groq_client: GroqClient = None,
    ):
        shared_store = store or DocumentStore()
        shared_embedder = embedder or EmbeddingModel()

        self.retriever = HybridRetriever(store=shared_store, embedder=shared_embedder)
        self.prompt_builder = PromptBuilder(max_context_chars=settings.max_context_chars)
        self.groq_client = groq_client or GroqClient()
        self.citation_parser = CitationParser()

    def answer(
        self,
        question: str,
        top_k: int = None,
        top_n: int = None,
        use_reranker: bool = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a single question.

        Parameters
        ----------
        question     : natural-language question from the user
        top_k        : candidates fetched per retriever (passed through to HybridRetriever)
        top_n        : final results after reranking (passed through to HybridRetriever)
        use_reranker : override reranker flag for this call only

        Returns
        -------
        RAGResponse with generated answer, cited sources, and retrieval results.
        """
        t0 = time.perf_counter()

        # ── Stage 1: hybrid retrieval ─────────────────────────────────────────
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
            top_n=top_n,
            use_reranker=use_reranker,
        )
        logger.info(f"Stage 1 — retrieved {len(results)} chunks")

        # ── Stage 2: prompt construction ──────────────────────────────────────
        # used_results may be a subset of results if context limit was reached.
        user_message, used_results = self.prompt_builder.build(question, results)
        logger.info(f"Stage 2 — prompt built with {len(used_results)} chunks")

        # ── Stage 3: LLM generation ───────────────────────────────────────────
        answer_text = self.groq_client.complete(
            system=SYSTEM_PROMPT,
            user=user_message,
        )
        logger.info("Stage 3 — generation complete")

        # ── Stage 4: citation parsing ─────────────────────────────────────────
        cited_sources = self.citation_parser.parse(answer_text, used_results)
        logger.info(f"Stage 4 — {len(cited_sources)} unique citations extracted")

        latency_ms = (time.perf_counter() - t0) * 1_000

        return RAGResponse(
            question=question,
            answer=answer_text,
            cited_sources=cited_sources,
            all_sources=results,
            latency_ms=round(latency_ms, 2),
            model=self.groq_client.model,
        )
