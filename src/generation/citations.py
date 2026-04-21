"""
CitationParser — maps [n] tokens in the LLM answer back to source chunks.

WHY parse citations post-generation instead of structured output:
  Structured output (JSON with a "citations" array) forces the LLM to emit a
  schema rather than natural prose. The [n] inline style lets the model write
  naturally ("the policy states [1] that…") while still providing a machine-
  readable link. It also survives model updates — any model that can follow
  "cite with [1]" instructions works without schema changes.

WHY deduplicate citation numbers:
  The LLM often writes [1] multiple times in one answer when the same chunk
  supports multiple claims. We want one CitedSource per unique number, not one
  per occurrence, so the sources list shown to the user is clean.

WHY sort by citation number, not by score:
  The user's reading order follows the citation numbers as they appear in the
  answer. Sorting by score would scramble that reading order. The score is still
  available on CitedSource if the caller wants to re-sort for debugging.

WHY silently ignore out-of-range citation numbers:
  The LLM occasionally hallucinates a [7] when only 5 chunks were provided.
  Raising an exception would crash the response; returning nothing for that
  citation number is safe — the answer still contains the text, just without
  a broken source link. We log a warning so it shows up in monitoring.

Interview question this answers:
  "How do you attribute sources in a RAG response?"
  Answer: Each context chunk is labelled [1], [2], … in the prompt. The LLM
  is instructed to cite inline. After generation, a regex extracts every [n]
  from the answer text. We deduplicate, validate the index is in-range, and
  return CitedSource objects that the UI can render as "Sources: [1] doc.pdf".
"""

import logging
import re

from src.generation.models import CitedSource
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


class CitationParser:

    def parse(
        self,
        answer: str,
        used_results: list[SearchResult],
    ) -> list[CitedSource]:
        """
        Extract inline citations from the answer and map them to source chunks.

        Parameters
        ----------
        answer       : raw LLM output text (may contain [1], [2], …)
        used_results : the chunks that were actually included in the prompt,
                       in the same order they were numbered (index 0 → [1])

        Returns
        -------
        Deduplicated list of CitedSource, sorted by citation number.
        Citations that reference a number beyond len(used_results) are dropped.
        """
        raw_numbers = {int(m) for m in _CITATION_RE.findall(answer)}
        sorted_numbers = sorted(raw_numbers)

        cited: list[CitedSource] = []
        for n in sorted_numbers:
            idx = n - 1  # [1] → index 0
            if idx < 0 or idx >= len(used_results):
                logger.warning(
                    f"LLM cited [{n}] but only {len(used_results)} chunks were "
                    "provided — dropping phantom citation"
                )
                continue
            r = used_results[idx]
            cited.append(
                CitedSource(
                    citation_number=n,
                    chunk_id=r.chunk_id,
                    source=r.source,
                    doc_id=r.doc_id,
                    content=r.content,
                    score=r.score,
                )
            )

        logger.debug(
            f"CitationParser: found {len(raw_numbers)} unique citation numbers, "
            f"{len(cited)} valid"
        )
        return cited
