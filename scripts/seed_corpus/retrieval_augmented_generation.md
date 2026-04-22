# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that improves LLM outputs by grounding them in retrieved external knowledge. Instead of relying solely on model weights, a RAG system retrieves relevant document chunks from a knowledge base and passes them as context to the LLM at inference time.

RAG constrains the model to content in retrieved passages rather than relying on patterns encoded in weights during training. The model is instructed to cite every claim with a numbered source, making it easier to detect when the model deviates from the provided context. This reduces hallucination compared to a plain LLM.

RAG allows knowledge to be updated by indexing new documents without retraining or fine-tuning the LLM. Fine-tuning requires expensive GPU compute and risks catastrophic forgetting of prior knowledge. RAG separates the knowledge store from the model, making updates cheap and immediate.

## The Six Stages of a RAG Pipeline

1. **Ingestion** — load, chunk, embed, and index documents
2. **Retrieval** — hybrid search using dense and sparse methods
3. **Reranking** — cross-encoder re-scores top candidates
4. **Prompt construction** — numbered context block injected into system prompt
5. **Generation** — LLM produces answer with inline citations
6. **Verification and scoring** — LLM-as-judge checks citations and computes confidence

## Bi-encoder vs. Cross-encoder

A bi-encoder encodes the query and document independently into separate vectors; similarity is computed via dot product or cosine similarity after encoding. A cross-encoder jointly encodes the query-document pair together, allowing full attention across both — it is more accurate but slower because documents cannot be pre-encoded.

The pipeline uses cross-encoder/ms-marco-MiniLM-L-6-v2. It was trained on the MS MARCO passage ranking dataset and is fast (CPU-viable) and strong for passage reranking.

## Chunking Strategies

The pipeline supports three chunking strategies:

- **Fixed-size**: splits on character count with configurable overlap
- **Recursive-header**: splits on Markdown headers first, then by character count
- **Semantic**: embeds each sentence and inserts chunk boundaries where cosine similarity between adjacent sentences drops below a threshold

Chunk overlap (e.g. 64 characters) preserves context for sentences that span chunk boundaries. Without overlap, a sentence split exactly at a boundary would lose its first or last words, degrading both embedding quality and the readability of the chunk when displayed as a citation source.

## Document Formats

The pipeline handles .txt, .md, .html, .htm, and .pdf files. HTML files are parsed with BeautifulSoup (removing navigation, scripts, and style tags). PDFs are parsed page-by-page with pypdf. All formats are normalised to plain text before chunking.

## Deduplication

Before adding a chunk, the system embeds it and queries ChromaDB for the nearest existing chunk. If the cosine similarity exceeds the dedup threshold (default 0.95), the chunk is skipped. This handles re-ingested documents and near-identical content without exact string comparison, keeping the index clean.

## Retrieval Confidence Threshold

When the average clamped relevance of retrieved chunks falls below the retrieval_confidence_threshold (default 0.3), the pipeline returns a structured "I don't know" response without calling the LLM at all. This saves tokens, prevents hallucination, and returns actionable guidance to the caller about checking source documents.

## FastAPI Implementation

The lifespan context manager (replacing the deprecated @app.on_event('startup')) initialises the RAGPipeline once when the server starts and holds it alive for the server's lifetime. This avoids the 2–5 second model and ChromaDB load on the first request. The lifespan pattern also works correctly with pytest's AsyncClient for integration tests.

The API schemas are public contracts that must remain stable for callers. Internal dataclasses carry types like SearchResult and VerifiedCitation that reference internal implementation details. Separating them means the internal pipeline can evolve without breaking the API, and the API can be versioned independently of the pipeline logic.

## API Endpoints

The `/documents` endpoint lists all documents currently indexed in the vector store. It returns the total document count, total chunk count, and per-document information including doc_id, source path, and chunk count. This allows the Streamlit UI and API consumers to inspect what has been indexed without querying ChromaDB directly.

The `/ingest` endpoint accepts an UploadFile, validates the file extension, saves the content to a NamedTemporaryFile on disk (because IngestionPipeline expects a file path), runs the full ingestion pipeline using the shared DocumentStore and EmbeddingModel from the live RAGPipeline, then deletes the temp file. The shared store ensures new documents are immediately queryable.

## The skip_verification Flag

When skip_verification=True, the pipeline skips the LLM-as-judge citation verification step and uses neutral placeholder confidence values (0.5 for citation coverage and completeness). This reduces latency and LLM token usage, making it suitable for high-throughput evaluation runs where generation metrics are assessed separately.

## Hybrid vs. Dense-Only Comparison

When the comparison toggle is enabled, the Streamlit UI runs the same question against the API twice — once with retrieval_mode='hybrid' (BM25 + dense + RRF) and once with retrieval_mode='dense_only' (embeddings only). Results are shown side-by-side, allowing users to see how retrieval mode affects the retrieved sources and the generated answer.

## Streamlit UI: Citation Confidence Display

The UI uses a traffic-light badge system. A composite score of 0.7+ displays HIGH in green, 0.4–0.7 displays MEDIUM in orange, and below 0.4 displays LOW in red. It also shows four individual metric values (composite, retrieval, citation coverage, completeness) using st.metric cards.

## Docker Compose

The Docker Compose setup has three services: api (FastAPI on port 8000 with a health check), streamlit (Streamlit on port 8501, waits for api to be healthy), and seed (runs scripts/seed.py under the 'seed' profile, so it only runs when explicitly triggered with --profile seed). All services mount ./data as a volume to share the ChromaDB and BM25 indexes.

Downloading HuggingFace models at build time bakes them into the image layer. This means containers start instantly without waiting for a network download on the first request. Without this, the first query after a cold container start would time out while the 90MB embedding model and cross-encoder reranker downloaded from the HuggingFace Hub.

## Low Temperature for Generation

Low temperature (e.g. 0.1) prioritises factual consistency over creativity, reducing hallucination variance. For grounded Q&A the model should closely follow the retrieved context rather than generating diverse paraphrases, so high temperature is counterproductive.
