"""
GroqClient — thin wrapper around the Groq Python SDK.

WHY Groq instead of OpenAI:
  Groq runs on custom LPU (Language Processing Unit) hardware optimised for
  transformer inference. On llama3-70b-8192 it delivers ~300 tokens/second —
  roughly 10–20× faster than comparable GPU-hosted providers. For a RAG
  pipeline where retrieval already takes ~200ms, generation latency matters.

WHY cached_property for the SDK client:
  The Groq() constructor validates the API key and opens an HTTPX session.
  Doing that on every request would add overhead and flood the connection pool.
  cached_property initialises it once on first use and reuses the session for
  the lifetime of the process — same pattern as the cross-encoder in Phase 2.

WHY separate temperature and max_tokens from the constructor:
  Different call sites may want different values. The eval harness in Phase 4
  may call with temperature=0.0 for deterministic outputs; the API endpoint
  will use the .env defaults. Constructor params would bake in a single value;
  per-call params with .env fallbacks give flexibility without config bloat.

WHY not stream:
  Streaming is great for UX (tokens appear progressively) but complicates
  citation parsing — you need the full answer before you can run a regex over
  it. For Phase 3 we return the complete answer. Streaming can be layered on
  in Phase 5 (FastAPI + WebSocket) without changing this layer.

Interview question this answers:
  "Why did you choose Groq over the OpenAI API?"
  Answer: Speed. Groq's LPU delivers sub-second generation for 70B-parameter
  models, making the full RAG pipeline fast enough for interactive use without
  sacrificing model quality. The interface is API-compatible with OpenAI's
  ChatCompletion spec, so switching later is a one-line config change.
"""

import logging
from functools import cached_property

from groq import Groq

from src.config import settings

logger = logging.getLogger(__name__)


class GroqClient:

    def __init__(self, model: str = None):
        self.model = model or settings.groq_model

    @cached_property
    def _client(self) -> Groq:
        logger.info(f"Initialising Groq client (model={self.model})")
        return Groq(api_key=settings.groq_api_key)

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Send a two-turn (system + user) chat request and return the text reply.

        Parameters
        ----------
        system      : system prompt (role constraints, citation instructions)
        user        : user-turn message (context chunks + question)
        temperature : sampling temperature; defaults to settings.generation_temperature
        max_tokens  : max tokens to generate; defaults to settings.generation_max_tokens
        """
        temperature = temperature if temperature is not None else settings.generation_temperature
        max_tokens = max_tokens if max_tokens is not None else settings.generation_max_tokens

        logger.debug(
            f"Groq request: model={self.model}, temp={temperature}, "
            f"max_tokens={max_tokens}, user_chars={len(user)}"
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content
        logger.info(
            f"Groq response: {len(answer)} chars, "
            f"usage={response.usage.total_tokens} tokens"
        )
        return answer
