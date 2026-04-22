"""
RAGPipeline — end-to-end orchestration: retrieve → prompt → generate → cite → verify → score.

Full pipeline:
  1. HybridRetriever    → top_n ranked chunks (Phase 2)
  2. PromptBuilder      → numbered context block + grounded system prompt
  3. GroqClient         → LLM generation with inline [n] citations
  4. CitationParser     → maps [n] tokens back to source chunks
  5. CitationVerifier   → LLM-as-judge: does chunk [n] support its claim?
  6. AnswerConfidenceScorer → composite confidence (retrieval + citations + completeness)

WHY inject dependencies (store, embedder, llm_client) instead of constructing them:
  FastAPI (Phase 5) will create one RAGPipeline at startup and reuse it across
  requests. If we constructed DocumentStore inside __init__ every time, every
  test instantiation would open a new ChromaDB connection. Injection lets tests
  pass lightweight fakes and lets the API layer share one warm instance.

WHY structured "I don't know" when retrieval confidence is low:
  Instruction-following ("say I don't know if unsure") is fragile — the LLM
  may still hallucinate. A hard threshold on retrieval confidence lets the
  pipeline return a structured response before wasting an LLM call, and
  surfaces actionable guidance ("check these documents manually") to the caller.

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
  system prompt. The LLM generates an answer citing those numbers
  inline. A regex parser extracts the cited numbers and maps them back to source
  metadata. A judge model then verifies each citation, and a composite confidence
  score is computed from retrieval quality, citation coverage, and completeness.
"""

import logging
import time

from src.config import settings
from src.generation.citations import CitationParser
from src.generation.llm_client import LLMClient
from src.generation.models import AnswerConfidence, RAGResponse, VerifiedCitation
from src.generation.prompt import SYSTEM_PROMPT, PromptBuilder
from src.generation.scorer import AnswerConfidenceScorer
from src.generation.verifier import CitationVerifier
from src.ingestion.embedder import EmbeddingModel
from src.ingestion.store import DocumentStore
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

_IDK_ANSWER = (
    "I don't have enough relevant information in the provided documents to "
    "answer this question confidently. The retrieved content had low relevance "
    "scores, which suggests the documents may not cover this topic. "
    "Please check your source documents or rephrase the question."
)


class RAGPipeline:

    def __init__(
        self,
        store: DocumentStore = None,
        embedder: EmbeddingModel = None,
        llm_client: LLMClient = None,
        judge_client: LLMClient = None,
    ):
        shared_store = store or DocumentStore()
        shared_embedder = embedder or EmbeddingModel()
        shared_llm = llm_client or LLMClient()
        # Judge can share the same client instance when models are identical.
        shared_judge = judge_client or (
            shared_llm
            if settings.llm_judge_model == settings.llm_model
            else LLMClient(model=settings.llm_judge_model)
        )

        self.store = shared_store
        self.embedder = shared_embedder
        self.retriever = HybridRetriever(store=shared_store, embedder=shared_embedder)
        self.prompt_builder = PromptBuilder(max_context_chars=settings.max_context_chars)
        self.llm_client = shared_llm
        self.citation_parser = CitationParser()
        self.verifier = CitationVerifier(judge_client=shared_judge)
        self.scorer = AnswerConfidenceScorer(judge_client=shared_judge)

    def answer(
        self,
        question: str,
        top_k: int = None,
        top_n: int = None,
        use_reranker: bool = None,
        skip_verification: bool = False,
        dense_only: bool = False,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a single question.

        Parameters
        ----------
        question          : natural-language question from the user
        top_k             : candidates fetched per retriever (passed to HybridRetriever)
        top_n             : final results after reranking (passed to HybridRetriever)
        use_reranker      : override reranker flag for this call only
        skip_verification : if True, skip LLM-as-judge verification and scoring
                            (faster; useful for high-throughput eval runs)
        dense_only        : skip BM25 + RRF, use raw dense retrieval only
                            (enables hybrid vs. dense A/B comparison in the UI)

        Returns
        -------
        RAGResponse with generated answer, verified citations, confidence score,
        and retrieval results.
        """
        t0 = time.perf_counter()

        # ── Stage 1: hybrid retrieval ─────────────────────────────────────────
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
            top_n=top_n,
            use_reranker=use_reranker,
            dense_only=dense_only,
        )
        logger.info(f"Stage 1 — retrieved {len(results)} chunks")

        # ── Low-confidence early exit ──────────────────────────────────────────
        retrieval_conf = (
            sum(max(0.0, min(1.0, r.score)) for r in results) / len(results)
            if results
            else 0.0
        )
        if retrieval_conf < settings.retrieval_confidence_threshold:
            logger.info(
                f"Retrieval confidence {retrieval_conf:.3f} below threshold "
                f"{settings.retrieval_confidence_threshold} — returning IDK response"
            )
            latency_ms = (time.perf_counter() - t0) * 1_000
            zero_conf = AnswerConfidence(
                retrieval_confidence=round(retrieval_conf, 4),
                citation_coverage=0.0,
                completeness_score=0.0,
                composite_score=0.0,
            )
            return RAGResponse(
                question=question,
                answer=_IDK_ANSWER,
                cited_sources=[],
                all_sources=results,
                confidence=zero_conf,
                latency_ms=round(latency_ms, 2),
                model=self.llm_client.model,
            )

        # ── Stage 2: prompt construction ──────────────────────────────────────
        user_message, used_results = self.prompt_builder.build(question, results)
        logger.info(f"Stage 2 — prompt built with {len(used_results)} chunks")

        # ── Stage 3: LLM generation ───────────────────────────────────────────
        answer_text = self.llm_client.complete(
            system=SYSTEM_PROMPT,
            user=user_message,
        )
        logger.info("Stage 3 — generation complete")

        # ── Stage 4: citation parsing ─────────────────────────────────────────
        cited_sources = self.citation_parser.parse(answer_text, used_results)
        logger.info(f"Stage 4 — {len(cited_sources)} unique citations extracted")

        # ── Stage 5: citation verification ───────────────────────────────────
        if skip_verification or not cited_sources:
            verified_citations: list[VerifiedCitation] = [
                VerifiedCitation(
                    citation_number=cs.citation_number,
                    chunk_id=cs.chunk_id,
                    source=cs.source,
                    doc_id=cs.doc_id,
                    content=cs.content,
                    score=cs.score,
                    verified=None,
                    verification_reason="verification_skipped",
                )
                for cs in cited_sources
            ]
        else:
            verified_citations = self.verifier.verify(answer_text, cited_sources)
        logger.info(f"Stage 5 — {len(verified_citations)} citations verified")

        # ── Stage 6: confidence scoring ───────────────────────────────────────
        if skip_verification:
            confidence = AnswerConfidence(
                retrieval_confidence=round(retrieval_conf, 4),
                citation_coverage=0.5,
                completeness_score=0.5,
                composite_score=round(0.35 * retrieval_conf + 0.40 * 0.5 + 0.25 * 0.5, 4),
            )
        else:
            confidence = self.scorer.score(
                question=question,
                answer_text=answer_text,
                all_sources=results,
                verified_citations=verified_citations,
            )
        logger.info(f"Stage 6 — composite confidence={confidence.composite_score:.3f}")

        latency_ms = (time.perf_counter() - t0) * 1_000

        return RAGResponse(
            question=question,
            answer=answer_text,
            cited_sources=verified_citations,
            all_sources=results,
            confidence=confidence,
            latency_ms=round(latency_ms, 2),
            model=self.llm_client.model,
        )
