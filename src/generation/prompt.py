"""
PromptBuilder — assembles the messages sent to the LLM.

WHY number the context chunks as [1], [2], …:
  The LLM is instructed to cite inline using these numbers. After generation,
  CitationParser scans the answer text for [n] tokens and maps each back to the
  corresponding SearchResult. Without numbering, the LLM would have to reproduce
  filenames or chunk IDs verbatim — fragile and noisy.

WHY a character-based truncation limit (max_context_chars):
  Token counting requires running the tokenizer, which adds latency and couples
  this layer to a specific tokenizer implementation. Characters are a reasonable
  proxy: llama3 uses ~4 characters per token on average English text. Setting
  max_context_chars=12000 keeps the context under ~3 000 tokens, leaving plenty
  of headroom for the system prompt and generated answer within the model's
  context window.

WHY return used_results alongside the user message:
  The citation parser needs to know which chunks were actually included in the
  prompt — if the context was truncated, chunks beyond the cutoff were never
  seen by the LLM, so a [6] citation in the answer would be meaningless.
  Returning used_results lets the caller pass only the relevant slice to
  CitationParser, preventing phantom citations.

WHY keep temperature low (set in config, 0.1):
  Factual Q&A over documents rewards consistency over creativity. A temperature
  of 0.1 makes the model reliably cite the context rather than free-associating.

Interview question this answers:
  "How do you prevent the LLM from hallucinating in a RAG system?"
  Answer: Grounding. The system prompt explicitly restricts the model to the
  provided context and instructs it to say "I don't know" if the answer isn't
  there. Low temperature reduces variance. Citations create an auditable trail
  back to source chunks so hallucinations are easy to spot.
"""

import logging

from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the context chunks provided below. "
    "Cite your sources inline using [number] notation — e.g. write [1] or [2] "
    "immediately after any claim that comes from that chunk. "
    "If multiple chunks support the same claim, cite all of them: [1][3]. "
    "If the context does not contain enough information to answer, respond with: "
    "'I don't have enough information in the provided context to answer this question.' "
    "Do not invent facts, do not draw on external knowledge."
)


class PromptBuilder:

    def __init__(self, max_context_chars: int = 12_000):
        self.max_context_chars = max_context_chars

    def build(
        self,
        query: str,
        results: list[SearchResult],
    ) -> tuple[str, list[SearchResult]]:
        """
        Construct the user-turn message and return the subset of results used.

        Parameters
        ----------
        query   : the user's question
        results : ranked retrieval results (highest-score first)

        Returns
        -------
        (user_message, used_results)
          user_message  — the full text to send as the user turn
          used_results  — the chunks actually included (may be fewer than results
                          if max_context_chars was reached)
        """
        context_parts: list[str] = []
        used: list[SearchResult] = []
        total_chars = 0

        for i, result in enumerate(results, start=1):
            chunk_text = f"[{i}]\n{result.content}"
            if total_chars + len(chunk_text) > self.max_context_chars:
                logger.debug(
                    f"Context limit reached after {i - 1} chunks "
                    f"({total_chars} / {self.max_context_chars} chars)"
                )
                break
            context_parts.append(chunk_text)
            used.append(result)
            total_chars += len(chunk_text)

        context_block = "\n\n".join(context_parts)
        user_message = (
            f"Context chunks:\n\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer (cite sources inline as [1], [2], etc.):"
        )

        logger.info(
            f"PromptBuilder: {len(used)}/{len(results)} chunks included, "
            f"{total_chars} chars"
        )
        return user_message, used
