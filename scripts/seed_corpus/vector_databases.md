# Vector Databases

## What is a Vector Database?

A vector database stores high-dimensional embedding vectors alongside document text and metadata, and provides efficient approximate nearest-neighbor (ANN) search. It is the core storage layer for dense retrieval in RAG pipelines.

## ChromaDB

ChromaDB is an open-source embeddable vector database for LLM applications. It stores vectors alongside document text and metadata, uses HNSW for approximate nearest-neighbor search with cosine distance, persists to disk via PersistentClient, and requires no separate server for single-machine deployments — making it ideal for local development and portfolio projects.

### HNSW Indexing

ChromaDB uses Hierarchical Navigable Small World (HNSW) graphs for approximate nearest-neighbor search. HNSW enables logarithmic-time search, meaning query latency grows very slowly as the index size increases. ChromaDB defaults to cosine distance (distance = 1 minus similarity).

### No Separate Server Needed

ChromaDB's PersistentClient runs in-process — it reads and writes local files rather than making network calls to a server. Both the api and streamlit services share the same index by mounting the same ./data volume. A separate ChromaDB server container would add network latency and operational complexity with no benefit on a single machine.

## Deduplication with Cosine Similarity

Before adding a new chunk, the system queries ChromaDB for the nearest existing chunk using cosine similarity. If the similarity exceeds a threshold (default 0.95), the chunk is skipped as a near-duplicate. This handles re-ingested documents and near-duplicate content without requiring exact string comparison.

## ChromaDB vs. Pinecone

Pinecone is a managed SaaS vector database with zero operational overhead but costs money at scale. ChromaDB can run embedded in-process with no server, making it free and easy for local development and portfolios. For production, Pinecone eliminates infrastructure management while ChromaDB requires self-hosting or using its managed cloud offering.

## Other Vector Databases

- **Weaviate**: open-source, supports multi-modal vectors, has a GraphQL API
- **Qdrant**: Rust-based, high performance, supports payload filtering
- **Milvus**: distributed, scales to billions of vectors, requires Kubernetes
- **pgvector**: PostgreSQL extension, stores vectors in existing relational tables — easiest migration path for teams already using Postgres

For portfolio and local development, ChromaDB's zero-infrastructure requirement makes it the standard choice.
