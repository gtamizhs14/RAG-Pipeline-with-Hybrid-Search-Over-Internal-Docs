# LLM-as-Judge Evaluation

## What is LLM-as-Judge?

LLM-as-judge uses one language model to evaluate the outputs of another. It has become the standard approach for automatic RAG evaluation because human evaluation is expensive, and string-matching metrics like BLEU miss paraphrases. A judge model is prompted with the question, answer, and context, and returns a structured score with a reason.

## Citation Verification

Citation verification detects two failure modes:

1. **Hallucinated citations** — the LLM cites source [3] for a claim, but chunk [3] says nothing about it
2. **Misattribution** — the claim is true but cited from the wrong source

Both make answers look grounded when they are not.

The pipeline runs citation verification per (claim, chunk) pair. For each citation in the generated answer, the judge model checks whether the cited chunk actually supports the specific claim. The verdict is SUPPORTED or UNSUPPORTED with a one-sentence reason.

## JSON Output Format

Free-text judge outputs are hard to parse reliably at scale. JSON output allows the application to extract structured fields — score, verdict (SUPPORTED/UNSUPPORTED), and reason — without fragile string parsing. If the judge returns malformed text, the application can fail explicitly rather than silently producing wrong scores.

## Self-Serving Bias

Self-serving bias occurs when the judge model is the same as the generation model — the model may validate its own outputs rather than objectively assessing them. This effect is small for factual citation verification but larger for subjective quality judgments. Using a stronger or different judge model is preferred for production evaluation.

## Effective Judge Prompt Design

An effective judge prompt should specify:

1. The evaluation criterion being assessed
2. The evidence the judge may use (context passages, question, answer)
3. The required output format (structured JSON with score and reason)
4. Examples of SUPPORTED vs. UNSUPPORTED verdicts to anchor the judge's interpretation

## skip_verification Flag

When skip_verification=True, the pipeline skips the LLM-as-judge citation verification step and uses neutral placeholder confidence values (0.5 for citation coverage and completeness). This reduces latency and LLM token usage, making it suitable for high-throughput evaluation runs where generation metrics are assessed separately.

## Faithfulness Evaluation

A faithfulness score measures whether every claim in the generated answer is supported by at least one retrieved chunk. A judge prompt for faithfulness presents the full answer and all retrieved chunks, asking the judge to identify any claims that cannot be traced back to the context.

## Answer Relevance Evaluation

An answer relevance score measures whether the answer actually addresses the question that was asked. A judge prompt for relevance presents the question and answer only (not the context), asking the judge to score how directly the answer responds to the question.
