"""
Seed script: writes sample AI engineering documents to data/raw/ and ingests them.

Run once before the first demo to populate the vector store:
    python scripts/seed.py

WHY a seed script instead of bundling documents in the repo:
  Real documents may be copyrighted or contain sensitive information. A seed
  script generates synthetic but realistic content that:
    1. Demonstrates every retrieval path (dense + sparse + hybrid)
    2. Exercises citation verification with verifiable claims
    3. Lets reviewers run the full pipeline without needing their own corpus

WHY write to data/raw/ and then call IngestionPipeline:
  Using the same IngestionPipeline that production uses means the seed data
  exercises the exact chunking and dedup logic reviewers will see live.
  It also means the seed can be re-run safely — duplicates are filtered.
"""

import sys
from pathlib import Path

# Allow running as `python scripts/seed.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline  # noqa: E402

RAW_DIR = Path("data/raw")

# ── Sample documents ──────────────────────────────────────────────────────────
DOCUMENTS = {
    "retrieval_augmented_generation.md": """
# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that improves large language model (LLM)
outputs by grounding them in retrieved external knowledge. Instead of relying solely on
information encoded in model weights during training, a RAG system first retrieves relevant
document chunks from a knowledge base, then passes those chunks as context to the LLM.

The core motivation is factual accuracy: LLMs trained on static datasets hallucinate when
asked about events after their knowledge cutoff or about domain-specific facts not well
represented in training data. RAG addresses this by dynamically fetching up-to-date,
authoritative information at inference time.

## RAG Pipeline Stages

A standard RAG pipeline has six stages:

1. **Ingestion**: Documents are loaded, chunked into passages, embedded into dense vectors,
   and indexed in a vector store (e.g., ChromaDB) alongside a sparse BM25 index.

2. **Retrieval**: A user query is embedded and used to search the vector store. Hybrid
   search combines dense retrieval (cosine similarity) with sparse BM25 keyword matching,
   then fuses results with Reciprocal Rank Fusion (RRF).

3. **Reranking**: A cross-encoder model (e.g., ms-marco-MiniLM-L-6-v2) re-scores the top
   fused candidates. Cross-encoders are slower than bi-encoders but more accurate because
   they jointly encode the query and passage together.

4. **Prompt construction**: Retrieved chunks are numbered [1]–[N] and inserted into a
   grounded system prompt that instructs the LLM to cite sources inline.

5. **Generation**: The LLM produces an answer with inline numeric citations. Groq's
   llama3-70b-8192 offers 8 192-token context at high throughput.

6. **Verification and scoring**: An LLM-as-judge checks whether each cited chunk actually
   supports the claim it was cited for. A composite confidence score is computed from
   retrieval quality, citation coverage, and answer completeness.

## Advantages of RAG

- **Reduced hallucination**: The model is constrained to content in retrieved passages.
- **Updateable knowledge**: New documents can be indexed without retraining the LLM.
- **Attributable answers**: Citations allow users to verify every claim.
- **Cost-effective**: Smaller fine-tuned models often outperform larger base models when
  augmented with good retrieval.
""",

    "hybrid_search.md": """
# Hybrid Search: Combining Dense and Sparse Retrieval

## Why Hybrid Search?

Neither dense nor sparse retrieval is strictly superior. They fail in complementary ways:

- **Dense retrieval** (embedding similarity) excels at semantic matching — it understands
  that "vehicle collision" and "car accident" mean the same thing. It struggles with rare
  technical terms, version numbers, or entity names not well-represented in training data.

- **Sparse retrieval** (BM25 keyword matching) excels at exact-match lookups — finding
  "errno ECONNREFUSED" or "LangGraph v0.2.1" verbatim. It fails at paraphrase and synonym
  matching because it operates purely on token overlap.

Hybrid search runs both and merges the results, capturing the strengths of each.

## BM25 (Best Match 25)

BM25 is the industry-standard sparse retrieval algorithm, used by Elasticsearch and
Lucene. It extends TF-IDF with two saturation functions:

- **Term frequency saturation** (k1 parameter): prevents a term appearing 100 times from
  being 100× more relevant than one appearing once.
- **Document length normalization** (b parameter): shorter documents are not penalized
  relative to longer documents containing the same information.

The BM25 score for a query term t in document d is:

    score(t, d) = IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d| / avgdl))

where IDF is inverse document frequency, tf is term frequency, |d| is document length,
and avgdl is average document length across the corpus.

## Reciprocal Rank Fusion (RRF)

RRF is a score-free fusion algorithm that combines ranked lists from multiple retrievers.
For each document, its RRF score is the sum of 1/(k + rank_i) across all ranked lists:

    RRF(d) = Σ_i  1 / (k + rank_i(d))

The constant k (typically 60) dampens the impact of very high ranks. RRF outperforms
linear score interpolation because it does not require score normalization — scores from
different retrievers are often on incompatible scales.

## Dense Retrieval with Sentence Transformers

Dense retrieval encodes both queries and documents into fixed-length vectors using a
bi-encoder architecture. At inference time, only the query is encoded; document embeddings
are pre-computed and cached. Cosine similarity between query and document vectors gives
the relevance score.

The all-MiniLM-L6-v2 model produces 384-dimensional embeddings. It is fast (CPU inference
in ~5ms per query), lightweight (~90MB), and achieves approximately 90% of the retrieval
quality of OpenAI's text-embedding-3-small on standard benchmarks.
""",

    "vector_databases.md": """
# Vector Databases for Semantic Search

## What is a Vector Database?

A vector database stores and indexes high-dimensional embedding vectors, enabling
sub-second approximate nearest-neighbor (ANN) search at scale. Traditional relational
databases support exact equality and range queries. Vector databases support similarity
queries: "find the 20 documents most similar to this embedding."

## ChromaDB

ChromaDB is an open-source, embeddable vector database designed for LLM applications.
Key characteristics:

- **Embedding storage**: ChromaDB stores vectors alongside the original document text and
  arbitrary key-value metadata (source filename, doc_id, chunk_index, etc.).
- **HNSW indexing**: Hierarchical Navigable Small World (HNSW) graphs enable logarithmic-
  time ANN search. ChromaDB defaults to cosine distance (distance = 1 − similarity).
- **Persistent storage**: PersistentClient stores the index to disk between process
  restarts. No separate server process is required for single-machine deployments.
- **Metadata filtering**: Queries can filter by metadata fields before vector search,
  enabling scoped retrieval (e.g., only search within a specific document).

## ChromaDB vs. Alternatives

| Database     | Deployment   | ANN Algorithm | Notes                         |
|--------------|-------------|---------------|-------------------------------|
| ChromaDB     | Embedded/server | HNSW       | Best for local dev + portfolios |
| Pinecone     | Managed SaaS  | Proprietary | Zero ops, but paid at scale   |
| Weaviate     | Self-hosted   | HNSW        | Rich schema, hybrid built-in  |
| Qdrant       | Self-hosted   | HNSW        | Fast, Rust-based              |
| pgvector     | PostgreSQL ext | IVFFlat    | Good if already on Postgres   |

## Deduplication

A practical concern when building RAG systems is duplicate chunks degrading retrieval
quality and wasting storage. Cosine similarity deduplication checks whether a new chunk's
embedding is within a threshold (e.g., 0.95) of any existing chunk. If it is, the chunk
is skipped. This handles re-ingested documents and near-duplicate content without
requiring exact string comparison.
""",

    "llm_evaluation.md": """
# Evaluating RAG Systems

## Why Evaluation Matters

A RAG system that returns plausible-sounding but incorrect answers is worse than one that
says "I don't know." Systematic evaluation distinguishes systems that look good on demos
from systems that are reliable in production.

## Retrieval Metrics

Retrieval quality is measured independently of the LLM. These metrics require only a set
of (question, relevant_doc_ids) pairs — no LLM calls needed:

**Precision@K**: Fraction of retrieved documents that are relevant.
    P@K = |{retrieved_K} ∩ {relevant}| / K

**Recall@K**: Fraction of all relevant documents that appear in the top K results.
    R@K = |{retrieved_K} ∩ {relevant}| / |{relevant}|

**Mean Reciprocal Rank (MRR)**: Average of 1/rank_of_first_relevant across queries.
    MRR = (1/N) Σ 1/rank_i

**NDCG@K** (Normalized Discounted Cumulative Gain): Accounts for rank position.
    DCG@K = Σ rel_i / log2(i+1), NDCG = DCG / IDCG

**Hit Rate**: Fraction of queries where at least one relevant document appears in top K.

## Generation Metrics

Generation quality is harder to measure. LLM-as-judge is the standard approach:

**Faithfulness**: Does the answer contain only claims that are supported by the retrieved
context? A judge model is asked to evaluate each sentence in the answer against the
provided passages, returning a score from 0 to 1 and a reason.

**Answer Relevance**: Does the answer actually address the question asked? The judge
evaluates whether the answer is on-topic, complete, and directly responsive.

**Citation Coverage**: What fraction of cited sources are verified as supporting their
corresponding claims? This is measured by the citation verification component.

## Composite Confidence Score

The pipeline computes a composite confidence score to signal answer trustworthiness:

    composite = 0.35 × retrieval_confidence + 0.40 × citation_coverage + 0.25 × completeness

- **Retrieval confidence**: Average clamped relevance score of retrieved chunks (0–1).
- **Citation coverage**: Fraction of cited sources verified as SUPPORTED.
- **Completeness**: LLM-as-judge score for whether the answer fully addresses the question.

Scores above 0.7 are HIGH confidence; 0.4–0.7 are MEDIUM; below 0.4 are LOW.

## Evaluation Dataset Design

A good evaluation dataset for RAG has 50+ samples covering:
- Simple factual questions (single-document answers)
- Multi-hop questions (require synthesizing across documents)
- Out-of-scope questions (correct answer: "I don't know")
- Adversarial questions (test against hallucination)

Each sample needs: question, ground_truth_answer, and relevant_doc_ids.
""",

    "llm_as_judge.md": """
# LLM-as-Judge for RAG Evaluation

## What is LLM-as-Judge?

LLM-as-judge uses one LLM to evaluate the outputs of another. It has become the standard
approach for automatic evaluation of RAG systems because:

1. Human evaluation is expensive, slow, and hard to scale.
2. String-matching metrics (BLEU, ROUGE) miss paraphrases and penalize valid synonyms.
3. Modern LLMs are reliable judges for factual accuracy, relevance, and completeness.

## Citation Verification

Citation verification asks the judge: "Does the content of chunk [n] actually support the
claim that cites it?" This catches two failure modes:

1. **Hallucinated citations**: The LLM cites [3] for a claim, but chunk [3] says nothing
   about it. The answer looks grounded but isn't.
2. **Misattribution**: The claim is true but cited from the wrong source.

The verifier prompts the judge with the (claim_sentence, chunk_content) pair and asks for
SUPPORTED or UNSUPPORTED with a one-sentence reason. The reason is returned to the caller
so users can see why a citation was flagged.

## Faithfulness vs. Factual Correctness

These are related but distinct:

- **Faithful** means the answer only makes claims that appear in the retrieved context,
  regardless of whether the context itself is correct.
- **Factually correct** means the answer is true in the world.

RAG evaluation typically measures faithfulness because we can assess it automatically
(context is available). Factual correctness requires external ground truth.

A highly faithful answer from a corpus of incorrect documents is still faithfully wrong.
This is why retrieval quality is a prerequisite for generation quality.

## Prompt Design for Judges

Effective judge prompts specify:
1. The evaluation criterion (faithfulness, relevance, etc.)
2. The evidence the judge may use (context passages, question, answer)
3. The output format (structured JSON with score + reason)
4. Examples of SUPPORTED vs. UNSUPPORTED verdicts

JSON output is critical: free-text judge outputs are hard to parse reliably at scale.

## Using the Same Model for Generation and Judging

When the judge model is the same as the generation model, there is a risk of self-serving
bias — the model may validate its own outputs. In practice, this effect is small for
factual citation verification but larger for subjective quality judgments. For production
systems, using a stronger or different judge model is preferred.
""",

    "groq_and_llms.md": """
# LLM APIs and Groq

## Groq API

Groq provides low-latency LLM inference using custom Language Processing Units (LPUs).
Key characteristics:

- **Speed**: Groq achieves ~500 tokens/second for llama3-70b-8192, compared to ~50–100
  tokens/second on typical GPU providers. This is critical for RAG pipelines where the
  LLM is called 2–4 times per query (generation + citation verification).
- **Free tier**: Generous rate limits (6 000 tokens per minute for free accounts) are
  sufficient for portfolio projects and development.
- **OpenAI-compatible API**: The Groq client library mirrors the OpenAI SDK interface,
  making it easy to swap providers.

## llama3-70b-8192

Meta's Llama 3 70B model with an 8 192-token context window is the primary generation
model in this pipeline. Characteristics:

- **Context window**: 8 192 tokens allows approximately 6 000 tokens of retrieved context
  (about 12 chunks of 500 tokens each) plus system prompt and generated answer.
- **Instruction following**: Strong adherence to citation format instructions
  ("[1]", "[2]") with low hallucination rates on grounded Q&A tasks.
- **Temperature**: 0.1 is recommended for factual RAG tasks. Higher temperatures increase
  response diversity but also hallucination rate.

## Prompt Engineering for Grounded Generation

The system prompt for RAG generation should:
1. Instruct the model to ONLY use information from the provided numbered sources.
2. Require inline citations in the format [n].
3. Explicitly forbid the model from using prior knowledge not in the sources.
4. Specify the "I don't know" response for questions not covered by the sources.

Example system prompt excerpt:
    "Answer ONLY using the numbered sources provided. Cite every claim with [n].
    If the sources do not contain enough information, say 'I don't know based on
    the provided documents' rather than guessing."

## Token Cost Optimization

Each RAG query uses tokens for:
- Retrieved context: ~2 000–4 000 tokens
- System prompt: ~200 tokens
- User question: ~20–100 tokens
- Generated answer: ~200–500 tokens
- Citation verification: ~300 tokens × number of cited sources

Total per query: approximately 3 000–6 000 tokens. At Groq's free tier (6 000 TPM),
budget for 1–2 queries per minute in development.
""",
}


def write_documents() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for filename, content in DOCUMENTS.items():
        path = RAW_DIR / filename
        path.write_text(content.strip(), encoding="utf-8")
        written += 1
    print(f"Wrote {written} documents to {RAW_DIR}/")


def ingest_documents() -> None:
    print("Starting ingestion pipeline...")
    pipeline = IngestionPipeline()
    stats = pipeline.run(RAW_DIR)

    print(f"\nIngestion complete:")
    print(f"  Documents loaded : {stats.documents_loaded}")
    print(f"  Chunks created   : {stats.chunks_created}")
    print(f"  Chunks added     : {stats.chunks_added}")
    print(f"  Chunks skipped   : {stats.chunks_skipped_duplicate} (deduplication)")
    if stats.errors:
        print(f"  Errors           : {len(stats.errors)}")
        for err in stats.errors:
            print(f"    - {err}")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline — Seed Script")
    print("=" * 60)
    write_documents()
    ingest_documents()
    print("\nDone. Start the API server and Streamlit UI to query the documents.")
    print("  uvicorn src.api.main:app --reload --port 8000")
    print("  streamlit run streamlit_app.py")
