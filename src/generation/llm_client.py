"""
LLMClient — provider-agnostic wrapper over any OpenAI-compatible chat API.

WHY OpenAI SDK instead of provider-specific SDKs:
  Groq, OpenAI, Together AI, Ollama, LM Studio, Anyscale, and most hosted
  LLM APIs all expose the same OpenAI ChatCompletion interface. Using the
  openai SDK with a configurable base_url means switching providers is a
  one-line .env change — no code changes required.

WHY base_url + api_key as the abstraction boundary:
  These two fields are all the OpenAI SDK needs to route requests. Every
  OpenAI-compatible provider documents them. The model name is also provider-
  specific but already in .env. Together, LLM_BASE_URL + LLM_API_KEY +
  LLM_MODEL fully specify a provider without any provider-specific code.

Provider presets (set in .env — see .env.example):
  Groq    : LLM_BASE_URL=https://api.groq.com/openai/v1
  OpenAI  : LLM_BASE_URL=https://api.openai.com/v1
  Ollama  : LLM_BASE_URL=http://localhost:11434/v1  (LLM_API_KEY=ollama)
  Together: LLM_BASE_URL=https://api.together.xyz/v1

WHY cached_property for the SDK client:
  The OpenAI() constructor validates the API key and opens an HTTPX session.
  Doing that on every request adds overhead and floods the connection pool.
  cached_property initialises once on first use and reuses the session for
  the lifetime of the process.

WHY not stream:
  Streaming is great for UX but complicates citation parsing — the full answer
  is needed before running the [n] regex over it. Streaming can be layered on
  later (FastAPI SSE / WebSocket) without changing this client.

Interview question this answers:
  "Is your LLM provider swappable?"
  Answer: Yes. LLMClient uses the openai SDK pointed at a configurable
  base_url. Switching from Groq to OpenAI to a local Ollama instance is a
  single .env change — the pipeline never imports a provider-specific SDK.
"""

import logging
from functools import cached_property

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(self, model: str = None):
        self.model = model or settings.llm_model

    @cached_property
    def _client(self) -> OpenAI:
        logger.info(
            f"Initialising LLMClient (model={self.model}, "
            f"base_url={settings.llm_base_url})"
        )
        return OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

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
            f"LLM request: model={self.model}, base_url={settings.llm_base_url}, "
            f"temp={temperature}, max_tokens={max_tokens}, user_chars={len(user)}"
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
            f"LLM response: {len(answer)} chars, "
            f"usage={response.usage.total_tokens} tokens"
        )
        return answer
