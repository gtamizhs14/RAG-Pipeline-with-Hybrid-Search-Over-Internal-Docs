"""
Generation quality metrics evaluated with an LLM-as-judge.

Two metrics:

  1. Faithfulness — is every claim in the answer grounded in the retrieved
     context? A high-faithfulness answer contains no hallucinations.
     Judge prompt: "Given only the context below, is this answer supported?"

  2. Answer Relevance — does the answer actually address the question asked?
     An answer can be faithful (no hallucinations) but still off-topic.
     Judge prompt: "Does this answer address the question below?"

WHY LLM-as-judge instead of BLEU/ROUGE:
  BLEU and ROUGE measure n-gram overlap with a reference answer. They penalise
  valid paraphrases and reward wrong answers that share keywords. An LLM judge
  reads semantically — it can tell "Python is interpreted" and "Python is not
  compiled" are equivalent, and it can tell a hallucinated fact from a true one.
  For RAG evaluation, faithfulness (no hallucination) matters more than
  surface-level similarity to a reference answer.

WHY return both a score and a reason:
  A score alone tells you the system is bad. A reason tells you *why* — was the
  LLM citing a chunk that doesn't contain the fact, or was the answer completely
  off-topic? The reason surfaces in the EvalReport for human review.

WHY 0.0–1.0 float instead of binary:
  A single hallucinated sentence in a mostly-correct answer shouldn't score the
  same as a completely fabricated response. The judge is asked for a decimal
  score so partial credit propagates correctly into aggregate metrics.
"""

import logging
import re

from src.generation.llm_client import LLMClient
from src.config import settings

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\b([01](?:\.\d+)?|\d?\.\d+)\b")

_FAITHFULNESS_SYSTEM = (
    "You are a strict fact-checking judge evaluating a RAG system. "
    "You will be given a CONTEXT (retrieved chunks) and an ANSWER. "
    "Judge whether every factual claim in the answer is supported by the context. "
    "Ignore stylistic choices; focus only on factual grounding. "
    "Reply with a decimal score between 0.0 and 1.0 on the first line "
    "(1.0 = fully grounded, 0.0 = completely hallucinated). "
    "Then on the second line write one short sentence explaining your score."
)

_RELEVANCE_SYSTEM = (
    "You are a strict QA evaluator. "
    "You will be given a QUESTION and an ANSWER. "
    "Judge whether the answer directly addresses the question. "
    "Reply with a decimal score between 0.0 and 1.0 on the first line "
    "(1.0 = fully addresses the question, 0.0 = completely off-topic). "
    "Then on the second line write one short sentence explaining your score."
)


def _parse_score_and_reason(reply: str, label: str) -> tuple[float, str]:
    lines = reply.strip().splitlines()
    score_line = lines[0].strip() if lines else ""
    reason = lines[1].strip() if len(lines) > 1 else reply.strip()
    match = _SCORE_RE.search(score_line)
    if match:
        val = max(0.0, min(1.0, float(match.group(1))))
        return val, reason
    logger.warning(f"{label} judge unexpected format: {reply!r}")
    return 0.5, reply.strip()[:200]


class GenerationMetricsScorer:

    def __init__(self, judge_client: LLMClient = None):
        self._judge = judge_client or LLMClient(model=settings.llm_judge_model)

    def faithfulness(
        self,
        context_chunks: list[str],
        answer: str,
    ) -> tuple[float, str]:
        """
        Score how faithfully the answer is grounded in the retrieved context.

        Parameters
        ----------
        context_chunks : list of chunk content strings shown to the LLM
        answer         : the generated answer to evaluate

        Returns
        -------
        (score: float, reason: str)
        """
        context_text = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )
        user_msg = f"CONTEXT:\n{context_text}\n\nANSWER:\n{answer}"
        try:
            reply = self._judge.complete(
                system=_FAITHFULNESS_SYSTEM,
                user=user_msg,
                temperature=0.0,
                max_tokens=100,
            )
            return _parse_score_and_reason(reply, "faithfulness")
        except Exception as exc:
            logger.warning(f"Faithfulness judge failed: {exc}")
            return 0.5, f"judge_error: {exc}"

    def answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> tuple[float, str]:
        """
        Score how well the answer addresses the question.

        Returns
        -------
        (score: float, reason: str)
        """
        user_msg = f"QUESTION: {question}\n\nANSWER:\n{answer}"
        try:
            reply = self._judge.complete(
                system=_RELEVANCE_SYSTEM,
                user=user_msg,
                temperature=0.0,
                max_tokens=100,
            )
            return _parse_score_and_reason(reply, "answer_relevance")
        except Exception as exc:
            logger.warning(f"Answer relevance judge failed: {exc}")
            return 0.5, f"judge_error: {exc}"
