"""
EvalRunner — orchestrates evaluation of the full RAG pipeline on a test set.

Evaluation flow for each EvalSample:
  1. Run RAGPipeline.answer() with skip_verification=True (avoids double judge
     calls — we do our own faithfulness + relevance scoring here instead)
  2. Extract retrieved doc_ids from all_sources
  3. Compute Precision@K, Recall@K, MRR, NDCG@K against relevant_doc_ids
  4. (If not --retrieval-only) Score faithfulness and answer relevance with
     GenerationMetricsScorer
  5. Assemble EvalResult; accumulate into EvalReport

WHY skip_verification=True in the pipeline call:
  The pipeline's CitationVerifier and the eval framework's GenerationMetrics
  scorer both call the judge model. Running both would double the LLM cost.
  The eval harness's faithfulness scorer is more comprehensive (it checks the
  whole answer against all context, not just per-citation pairs), so we use it
  exclusively during eval and rely on the per-citation verifier only in
  production serving.

WHY pass the full context to the faithfulness judge (not just cited chunks):
  The LLM may have hallucinated a fact from a chunk it chose not to cite. Using
  all retrieved chunks as context gives the judge the full picture.

WHY log progress per sample:
  Eval runs can be long (N samples × 2–3 LLM calls each). Per-sample logs let
  operators monitor progress and kill the run early if they see systematic
  failures, without waiting for the full batch.
"""

import json
import logging
import time
from pathlib import Path

from src.eval.generation_metrics import GenerationMetricsScorer
from src.eval.models import (
    EvalReport,
    EvalResult,
    EvalSample,
    GenerationEval,
    RetrievalEval,
)
from src.eval.retrieval_metrics import compute_all
from src.generation.groq_client import GroqClient
from src.generation.pipeline import RAGPipeline
from src.config import settings

logger = logging.getLogger(__name__)


def load_eval_dataset(path: str | Path) -> list[EvalSample]:
    """
    Load eval samples from a JSON file.

    Expected format — a JSON array of objects:
    [
      {
        "id": "q1",
        "question": "What is ...?",
        "ground_truth_answer": "...",
        "relevant_doc_ids": ["doc_a", "doc_b"]
      },
      ...
    ]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    for item in data:
        samples.append(
            EvalSample(
                id=item["id"],
                question=item["question"],
                ground_truth_answer=item["ground_truth_answer"],
                relevant_doc_ids=item["relevant_doc_ids"],
            )
        )
    logger.info(f"Loaded {len(samples)} eval samples from {path}")
    return samples


class EvalRunner:

    def __init__(
        self,
        pipeline: RAGPipeline = None,
        judge_client: GroqClient = None,
        k: int = None,
        eval_generation: bool = True,
    ):
        """
        Parameters
        ----------
        pipeline        : RAGPipeline instance (creates one with defaults if None)
        judge_client    : GroqClient for generation metric scoring
        k               : top-k for retrieval metric computation (defaults to rerank_top_n)
        eval_generation : if False, skip faithfulness and answer relevance scoring
        """
        self.pipeline = pipeline or RAGPipeline()
        self.k = k if k is not None else settings.rerank_top_n
        self.eval_generation = eval_generation
        self._gen_scorer = (
            GenerationMetricsScorer(judge_client=judge_client)
            if eval_generation
            else None
        )

    def run(self, samples: list[EvalSample]) -> EvalReport:
        """
        Evaluate the pipeline on all samples and return an EvalReport.

        Parameters
        ----------
        samples : list of EvalSample (from load_eval_dataset)
        """
        results: list[EvalResult] = []

        for i, sample in enumerate(samples, start=1):
            logger.info(f"[{i}/{len(samples)}] Evaluating: {sample.id!r}")
            result = self._eval_sample(sample)
            results.append(result)
            logger.info(
                f"  P@{self.k}={result.retrieval.precision_at_k:.3f} "
                f"R@{self.k}={result.retrieval.recall_at_k:.3f} "
                f"MRR={result.retrieval.mrr:.3f} "
                + (
                    f"faith={result.generation.faithfulness:.3f} "
                    f"rel={result.generation.answer_relevance:.3f}"
                    if result.generation
                    else "generation_eval=skipped"
                )
            )

        return self._aggregate(results)

    def _eval_sample(self, sample: EvalSample) -> EvalResult:
        t0 = time.perf_counter()

        # ── RAG pipeline call ─────────────────────────────────────────────────
        # skip_verification=True avoids duplicate judge calls during eval.
        rag_response = self.pipeline.answer(
            question=sample.question,
            skip_verification=True,
        )

        latency_ms = (time.perf_counter() - t0) * 1_000

        # ── Retrieval metrics ─────────────────────────────────────────────────
        retrieved_doc_ids = [r.doc_id for r in rag_response.all_sources]
        metrics = compute_all(retrieved_doc_ids, sample.relevant_doc_ids, self.k)

        retrieval_eval = RetrievalEval(
            sample_id=sample.id,
            precision_at_k=metrics["precision_at_k"],
            recall_at_k=metrics["recall_at_k"],
            mrr=metrics["mrr"],
            ndcg_at_k=metrics["ndcg_at_k"],
            k=self.k,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=sample.relevant_doc_ids,
            hit=metrics["recall_at_k"] > 0.0,
        )

        # ── Generation metrics ────────────────────────────────────────────────
        gen_eval = None
        if self.eval_generation and self._gen_scorer:
            context_chunks = [r.content for r in rag_response.all_sources]
            faith, faith_reason = self._gen_scorer.faithfulness(
                context_chunks, rag_response.answer
            )
            rel, rel_reason = self._gen_scorer.answer_relevance(
                sample.question, rag_response.answer
            )
            gen_eval = GenerationEval(
                sample_id=sample.id,
                faithfulness=faith,
                faithfulness_reason=faith_reason,
                answer_relevance=rel,
                answer_relevance_reason=rel_reason,
                composite_score=round((faith + rel) / 2, 4),
            )

        return EvalResult(
            sample=sample,
            retrieval=retrieval_eval,
            generation=gen_eval,
            answer=rag_response.answer,
            latency_ms=round(latency_ms, 2),
            confidence_composite=rag_response.confidence.composite_score,
        )

    def _aggregate(self, results: list[EvalResult]) -> EvalReport:
        n = len(results)
        if n == 0:
            raise ValueError("No eval results to aggregate")

        def mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        ret = [r.retrieval for r in results]
        gen = [r.generation for r in results if r.generation is not None]

        return EvalReport(
            num_samples=n,
            k=self.k,
            mean_precision_at_k=mean([r.precision_at_k for r in ret]),
            mean_recall_at_k=mean([r.recall_at_k for r in ret]),
            mean_mrr=mean([r.mrr for r in ret]),
            mean_ndcg_at_k=mean([r.ndcg_at_k for r in ret]),
            hit_rate=mean([1.0 if r.hit else 0.0 for r in ret]),
            mean_faithfulness=mean([g.faithfulness for g in gen]) if gen else None,
            mean_answer_relevance=mean([g.answer_relevance for g in gen]) if gen else None,
            mean_generation_composite=mean([g.composite_score for g in gen]) if gen else None,
            mean_confidence_composite=mean([r.confidence_composite for r in results]),
            mean_latency_ms=mean([r.latency_ms for r in results]),
            results=results,
        )
