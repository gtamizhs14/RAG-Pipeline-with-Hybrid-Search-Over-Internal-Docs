# Hybrid Search: Dense + Sparse Retrieval

## Why Hybrid Search?

Dense retrieval excels at semantic matching but struggles with rare technical terms, version numbers, and entity names not well-represented in training data. Sparse BM25 handles exact-match lookups for terms like error codes or library versions. Hybrid search combines both, capturing semantic meaning and exact keyword matches.

For technical documentation, hybrid search outperforms pure dense retrieval because both patterns appear: natural language questions (semantic) and precise identifiers (keyword).

## BM25

BM25 extends TF-IDF with two saturation functions:

- **Term frequency saturation** (k1 parameter): prevents a term appearing many times from dominating linearly
- **Document length normalization** (b parameter): prevents longer documents from being unfairly penalized

These make BM25 more robust than raw TF-IDF for passage retrieval.

The rank_bm25 library does not have a native persistence format. Pickling the BM25Okapi object alongside the corpus and doc_ids is the simplest reliable persistence approach. The index is rebuilt from the full corpus on each new batch so IDF scores remain correct as the corpus grows.

## Reciprocal Rank Fusion (RRF)

RRF combines ranked lists from multiple retrievers by computing 1/(k + rank_i) for each document in each list, where k is typically 60. Scores are summed across lists per document and results are re-ranked by combined score. RRF is robust to score scale differences because it operates only on rank positions, not raw scores.

The k constant (typically 60) dampens the impact of very high ranks. Without it, the top-ranked document would receive a disproportionately large boost. k=60 is the value from the original RRF paper and has been empirically validated across many retrieval benchmarks.

## Embedding Model

The pipeline uses all-MiniLM-L6-v2 from SentenceTransformers. It produces 384-dimensional embeddings, runs on CPU in approximately 5ms per query, weighs about 90MB, and achieves roughly 90% of the retrieval quality of OpenAI's text-embedding-3-small on standard benchmarks.

## Dense-Only Mode

In dense_only mode, the HybridRetriever skips BM25 sparse retrieval and Reciprocal Rank Fusion entirely. It queries ChromaDB using only the query embedding and returns the top_n results by cosine similarity. This is faster but misses exact keyword matches that BM25 would catch.

## Pipeline Flow

1. Dense retriever queries ChromaDB with the embedded query, returns top_k results with cosine distances
2. Sparse BM25 retriever tokenizes the query, scores the corpus, returns top_k results
3. RRF fuses both ranked lists into a single ranked list
4. Cross-encoder reranker re-scores the fused top candidates and returns top_n final results
