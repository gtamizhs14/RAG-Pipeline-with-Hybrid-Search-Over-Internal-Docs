# LLM APIs: Groq, OpenAI, and OpenAI-Compatible Providers

## Groq

Groq provides low-latency LLM inference using custom Language Processing Units (LPUs). It achieves approximately 500 tokens per second for llama-3.3-70b-versatile — much faster than typical GPU providers. This speed is critical for RAG pipelines where the LLM is called multiple times per query for generation and citation verification.

### Context Window

llama-3.3-70b-versatile has a 128,000-token context window. The pipeline reserves space for the system prompt (~200 tokens), the user question (~100 tokens), and the generated answer (~500 tokens), leaving the remainder for retrieved context. max_context_chars is set to 12,000 characters (~3,000 tokens) to stay well within the typical LLM context budget.

### Token Usage per Query

A typical RAG query uses: 2,000–4,000 tokens for retrieved context, ~200 tokens for the system prompt, 20–100 tokens for the user question, 200–500 tokens for the generated answer, and ~300 tokens per cited source for citation verification. Total is approximately 3,000–6,000 tokens per query.

### Free Tier

Groq's free tier allows approximately 6,000 tokens per minute for most models. Since each RAG query uses 3,000–6,000 tokens (context + generation + verification), this limits development throughput to roughly 1–2 queries per minute. For evaluation runs with many samples, the skip_verification flag reduces token usage by skipping the per-citation judge calls.

## OpenAI-Compatible API Abstraction

The pipeline uses the openai Python SDK pointed at a configurable base_url. Any OpenAI-compatible API works with zero code changes — just update the environment variables:

| Provider | LLM_BASE_URL |
|----------|-------------|
| Groq | https://api.groq.com/openai/v1 |
| OpenAI | https://api.openai.com/v1 |
| Ollama | http://localhost:11434/v1 |
| Together AI | https://api.together.xyz/v1 |

## System Prompt Design

The system prompt for RAG generation instructs the model to:
- Answer only using the numbered sources provided
- Cite every claim with [n] inline
- Forbid using prior knowledge not in the sources
- Say "I don't know based on the provided documents" rather than guessing when sources do not contain enough information

Low temperature (e.g. 0.1) prioritises factual consistency over creativity, reducing hallucination variance. For grounded Q&A the model should closely follow the retrieved context rather than generating diverse paraphrases, so high temperature is counterproductive.

## Choosing a Model

For citation verification (LLM-as-judge), a smaller model can be used since the task is structured (SUPPORTED/UNSUPPORTED JSON output) and does not require creative generation. Using a separate judge model avoids self-serving bias from grading one's own outputs.

For generation, a 70B-class model (llama-3.3-70b-versatile, gpt-4o-mini) provides a good balance of instruction-following quality and token speed. Instruction-following quality matters more than raw knowledge because the model's job is to synthesize provided context, not recall from weights.
