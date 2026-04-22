# LLM and RAG Evaluation

## Why Evaluation Matters

Evaluation is what separates a demo from a production system. Without metrics, you cannot tell whether a change improved retrieval quality, made hallucination worse, or degraded latency. The RAG pipeline measures both retrieval quality (no LLM calls needed) and generation quality (LLM-as-judge calls).

## Retrieval Metrics

Retrieval metrics compare retrieved doc_ids against a pre-defined list of relevant_doc_ids from the evaluation dataset. This comparison is pure Python arithmetic — no language model is needed, making retrieval evaluation fast and cheap to run repeatedly.

### Precision@K

Precision@K measures what fraction of the K retrieved documents are actually relevant — it penalizes returning irrelevant results. High precision means few false positives.

### Recall@K

Recall@K measures what fraction of all relevant documents were found in the top K — it penalizes missing relevant documents. High recall means few false negatives.

### Mean Reciprocal Rank (MRR)

MRR is the average of the reciprocal rank of the first relevant document across all queries. If a relevant document appears at rank 1, it contributes 1/1 = 1.0; at rank 2, it contributes 0.5; at rank 3, it contributes 0.33, and so on. MRR focuses on how quickly the system surfaces the first relevant result.

### NDCG@K

NDCG (Normalized Discounted Cumulative Gain) measures retrieval quality accounting for rank position — a relevant result at rank 1 is worth more than one at rank 5. DCG sums relevance scores discounted by log2(rank+1), and NDCG normalises by the ideal DCG. This rewards systems that put the most relevant results at the top.

### Hit Rate

Hit rate is the fraction of queries where at least one relevant document appears in the top K retrieved results. Unlike MRR or NDCG, it does not consider how many relevant documents were found or at what rank — it is a binary pass/fail per query. Hit rate is useful as a quick sanity check on whether the retriever is finding anything relevant at all.

## Generation Metrics

### Faithfulness vs. Factual Correctness

Faithfulness means the answer only makes claims supported by the retrieved context, regardless of whether that context is itself correct. Factual correctness means the answer is true in the real world. RAG systems are typically evaluated for faithfulness because context is available at eval time, whereas factual correctness requires external ground truth.

### Answer Relevance vs. Faithfulness

An answer can be faithful (only citing content from the context) yet not relevant (answering a different question than was asked). Conversely, an answer can address the question but include hallucinated claims not in the context. Measuring both separately identifies which component is failing: retrieval (wrong documents), faithfulness (hallucination), or relevance (off-topic generation).

## Composite Confidence Score

The composite score is: **0.35 × retrieval_confidence + 0.40 × citation_coverage + 0.25 × completeness_score**.

- **Retrieval confidence**: the average clamped relevance of retrieved chunks
- **Citation coverage**: the fraction of cited sources verified as SUPPORTED
- **Completeness**: an LLM-as-judge score for whether the answer fully addresses the question

Confidence thresholds:
- **HIGH** (≥ 0.7): shown in green
- **MEDIUM** (0.4–0.7): shown in orange
- **LOW** (< 0.4): shown in red

## Evaluation Dataset Properties

A good RAG evaluation dataset should have 50+ samples covering:
- Simple factual questions (single-document answers)
- Multi-hop questions (require synthesizing across documents)
- Out-of-scope questions (correct answer is "I don't know")
- Adversarial questions (test hallucination resistance)

Each sample needs a question, ground truth answer, and relevant_doc_ids.

## Running Evaluation

```bash
# Full eval (retrieval + generation metrics, uses LLM-as-judge)
python run_eval.py

# Retrieval metrics only (no LLM calls — fast)
python run_eval.py --retrieval-only
```
