"""
CitationVerifier — LLM-as-judge that checks whether each cited chunk
actually supports the claim the LLM attached it to.

WHY LLM-as-judge instead of a similarity score:
  Cosine similarity tells you a chunk is topically close to the question,
  not that it *supports a specific claim*. A chunk about "Python speed" can
  be retrieved for "Is Python fast?" but may say "Python is slow for CPU
  tasks" — high similarity, wrong support. The judge reads both the claim
  and the chunk text, giving a binary SUPPORTED/UNSUPPORTED verdict with a
  short reason.

WHY extract claim sentences from the answer:
  The LLM writes "Alpha is fast [1]. Beta is slow [2]." — each citation is
  attached to a specific sentence. We extract the sentence(s) containing [n]
  as the "claim" to verify against chunk n. This keeps the verification
  prompt focused on a single claim/chunk pair rather than the whole answer,
  reducing judge hallucination.

WHY use groq_judge_model (not groq_model):
  The judge model can be the same model or a stronger one. Having a
  dedicated config key lets operators swap in a bigger model for judging
  (higher quality verdicts) while keeping generation fast with a smaller
  model. Both default to llama3-70b-8192 so it's a no-op unless explicitly
  configured.

WHY not fail on judge errors:
  If the judge call throws (rate limit, transient network error), we mark
  that citation as unverified=None (unknown) rather than crashing the whole
  pipeline. The caller can still return the answer with a caveat; losing
  one verification is better than losing the whole response.

Interview question this answers:
  "How do you know the LLM isn't just making up citations?"
  Answer: A second judge-model pass checks each (claim, chunk) pair and
  flags citations where the chunk doesn't support the claim. This gives
  users a per-source trust signal, not just a confidence score.
"""

import logging
import re

from src.generation.llm_client import LLMClient
from src.generation.models import CitedSource, VerifiedCitation
from src.config import settings

logger = logging.getLogger(__name__)

_VERDICT_RE = re.compile(r"\b(SUPPORTED|UNSUPPORTED)\b", re.IGNORECASE)

_JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. "
    "You will be given a CLAIM and a SOURCE CHUNK. "
    "Decide whether the chunk directly supports the claim. "
    "Reply with exactly one word on the first line: SUPPORTED or UNSUPPORTED. "
    "Then on the second line write one short sentence explaining why."
)


def _extract_claim_for(citation_number: int, answer_text: str) -> str:
    """Return the sentence(s) in answer_text that contain [n]."""
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    tag = f"[{citation_number}]"
    matched = [s for s in sentences if tag in s]
    return " ".join(matched) if matched else answer_text[:300]


class CitationVerifier:

    def __init__(self, judge_client: LLMClient = None):
        self._judge = judge_client or LLMClient(model=settings.llm_judge_model)

    def verify(
        self,
        answer_text: str,
        cited_sources: list[CitedSource],
    ) -> list[VerifiedCitation]:
        """
        For each CitedSource, ask the judge model whether the chunk
        actually supports the claim the LLM made when citing it.

        Returns a list of VerifiedCitation objects in the same order as
        cited_sources, with `verified` and `verification_reason` populated.

        Parameters
        ----------
        answer_text   : the full LLM-generated answer text (used to extract claims)
        cited_sources : CitedSource list from CitationParser
        """
        results: list[VerifiedCitation] = []

        for cs in cited_sources:
            claim = _extract_claim_for(cs.citation_number, answer_text)
            verified, reason = self._judge_pair(claim, cs.content, cs.citation_number)
            results.append(
                VerifiedCitation(
                    citation_number=cs.citation_number,
                    chunk_id=cs.chunk_id,
                    source=cs.source,
                    doc_id=cs.doc_id,
                    content=cs.content,
                    score=cs.score,
                    verified=verified,
                    verification_reason=reason,
                )
            )

        return results

    def _judge_pair(
        self, claim: str, chunk_content: str, citation_number: int
    ) -> tuple[bool | None, str]:
        """Call the judge model; return (verified_bool_or_None, reason_str)."""
        user_msg = (
            f"CLAIM: {claim}\n\n"
            f"SOURCE CHUNK [{citation_number}]:\n{chunk_content}"
        )
        try:
            reply = self._judge.complete(
                system=_JUDGE_SYSTEM,
                user=user_msg,
                temperature=0.0,
                max_tokens=80,
            )
            lines = reply.strip().splitlines()
            verdict_line = lines[0].strip() if lines else ""
            reason = lines[1].strip() if len(lines) > 1 else reply.strip()

            match = _VERDICT_RE.search(verdict_line)
            if match:
                verified = match.group(1).upper() == "SUPPORTED"
                logger.debug(f"[{citation_number}] judge={verified} reason={reason!r}")
                return verified, reason
            else:
                logger.warning(
                    f"[{citation_number}] judge returned unexpected format: {reply!r}"
                )
                return None, reply.strip()[:200]

        except Exception as exc:
            logger.warning(f"[{citation_number}] judge call failed: {exc}")
            return None, f"verification_error: {exc}"
