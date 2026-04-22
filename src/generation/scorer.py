"""
AnswerConfidenceScorer — composite confidence score for each RAG answer.

Three components (all in [0, 1]):
  1. retrieval_confidence  — average relevance score of retrieved chunks.
     High = the retriever found strongly-matching content; low = weak matches,
     answer may be hallucinated or off-topic.

  2. citation_coverage     — fraction of cited sources the judge verified as
     SUPPORTED. 1.0 = every citation checks out; 0.0 = none do.
     None-verdict citations (judge errors) count as 0.5 (uncertain).

  3. completeness_score    — LLM-as-judge answer: did the response address all
     parts of the question? Sent as a separate judge call so it is independent
     of citation verification.

composite_score = weighted average:
  0.35 * retrieval_confidence
  + 0.40 * citation_coverage
  + 0.25 * completeness_score

WHY these weights:
  Citation coverage is the strongest signal — an answer citing chunks that
  don't support it is actively misleading. Retrieval confidence matters but
  a high-scoring retrieval can still produce a bad answer. Completeness is
  a secondary quality signal and is the most expensive to compute (one more
  LLM call), so it gets the smallest weight.

WHY return a dataclass instead of a float:
  Callers (API, UI, eval harness) often want to inspect individual components
  to understand *why* confidence is low. A single float loses that signal.

Interview question this answers:
  "How does your system tell users when to trust the answer?"
  Answer: Every response carries a composite confidence score broken down
  into retrieval quality, citation verification coverage, and answer
  completeness. The API surfaces all three, so the UI can show a simple
  traffic-light alongside the detailed breakdown for users who want it.
"""

import logging
import re

from src.generation.llm_client import LLMClient
from src.generation.models import AnswerConfidence, VerifiedCitation
from src.retrieval.models import SearchResult
from src.config import settings

logger = logging.getLogger(__name__)

_COMPLETENESS_SYSTEM = (
    "You are a strict QA evaluator. "
    "You will be given a QUESTION and an ANSWER. "
    "Judge whether the answer addresses all parts of the question. "
    "Reply with a single decimal number between 0.0 and 1.0 on the first line "
    "(1.0 = fully addressed, 0.0 = not addressed at all). "
    "Then on the second line write one short sentence explaining your score."
)

_SCORE_RE = re.compile(r"\b([01](?:\.\d+)?|\d?\.\d+)\b")


class AnswerConfidenceScorer:

    def __init__(self, judge_client: LLMClient = None):
        self._judge = judge_client or LLMClient(model=settings.llm_judge_model)

    def score(
        self,
        question: str,
        answer_text: str,
        all_sources: list[SearchResult],
        verified_citations: list[VerifiedCitation],
    ) -> AnswerConfidence:
        """
        Compute a composite confidence score for the answer.

        Parameters
        ----------
        question           : original user question
        answer_text        : LLM-generated answer
        all_sources        : all retrieved chunks (used for retrieval_confidence)
        verified_citations : output of CitationVerifier.verify()
        """
        retrieval_confidence = self._retrieval_confidence(all_sources)
        citation_coverage = self._citation_coverage(verified_citations)
        completeness_score = self._completeness(question, answer_text)

        composite = round(
            0.35 * retrieval_confidence
            + 0.40 * citation_coverage
            + 0.25 * completeness_score,
            4,
        )

        logger.info(
            f"Confidence: retrieval={retrieval_confidence:.3f} "
            f"citations={citation_coverage:.3f} "
            f"completeness={completeness_score:.3f} "
            f"composite={composite:.3f}"
        )

        return AnswerConfidence(
            retrieval_confidence=round(retrieval_confidence, 4),
            citation_coverage=round(citation_coverage, 4),
            completeness_score=round(completeness_score, 4),
            composite_score=composite,
        )

    @staticmethod
    def _retrieval_confidence(sources: list[SearchResult]) -> float:
        if not sources:
            return 0.0
        raw_scores = [s.score for s in sources]
        # Reranker scores can be negative; shift to [0, 1] via sigmoid-like clamp
        # For cosine/RRF scores already in [0,1] this is a no-op.
        clipped = [max(0.0, min(1.0, s)) for s in raw_scores]
        return sum(clipped) / len(clipped)

    @staticmethod
    def _citation_coverage(verified: list[VerifiedCitation]) -> float:
        if not verified:
            return 0.0
        scores = []
        for vc in verified:
            if vc.verified is True:
                scores.append(1.0)
            elif vc.verified is False:
                scores.append(0.0)
            else:
                scores.append(0.5)  # unknown / judge error
        return sum(scores) / len(scores)

    def _completeness(self, question: str, answer_text: str) -> float:
        user_msg = f"QUESTION: {question}\n\nANSWER:\n{answer_text}"
        try:
            reply = self._judge.complete(
                system=_COMPLETENESS_SYSTEM,
                user=user_msg,
                temperature=0.0,
                max_tokens=80,
            )
            lines = reply.strip().splitlines()
            score_line = lines[0].strip() if lines else ""
            match = _SCORE_RE.search(score_line)
            if match:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            logger.warning(f"Completeness judge unexpected format: {reply!r}")
            return 0.5
        except Exception as exc:
            logger.warning(f"Completeness judge call failed: {exc}")
            return 0.5
