from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM — any OpenAI-compatible provider (Groq, OpenAI, Ollama, Together AI…)
    # Set LLM_BASE_URL + LLM_API_KEY + LLM_MODEL in .env to switch providers.
    llm_api_key: str = Field(..., description="API key for the LLM provider")
    llm_base_url: str = Field(
        "https://api.groq.com/openai/v1",
        description="OpenAI-compatible base URL for the LLM provider",
    )
    llm_model: str = Field("llama-3.3-70b-versatile", description="Model used for generation")
    llm_judge_model: str = Field(
        "llama-3.3-70b-versatile",
        description="Model used as LLM-as-judge (can differ from generation model)",
    )

    # Embeddings — runs locally, swap model name in .env to change
    embedding_model: str = Field("all-MiniLM-L6-v2")
    embedding_dimension: int = Field(384)

    # ChromaDB
    chroma_db_path: str = Field("data/chroma_db")
    chroma_collection_name: str = Field("documents")

    # BM25
    bm25_index_path: str = Field("data/bm25_index.pkl")

    # Chunking
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(64)
    # For semantic chunking: similarity below this value = topic boundary
    semantic_similarity_threshold: float = Field(0.5)

    # Dedup: skip chunk if nearest neighbor similarity >= this
    dedup_similarity_threshold: float = Field(0.95)

    # Retrieval
    retrieval_top_k: int = Field(20)
    rerank_top_n: int = Field(5)
    rrf_k: int = Field(60)
    dense_weight: float = Field(0.7)
    sparse_weight: float = Field(0.3)
    reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker: bool = Field(True)

    raw_docs_path: str = Field("data/raw")

    # Generation
    # Low temperature (0.1) prioritises factual consistency over creativity.
    # max_context_chars ~12 000 chars ≈ 3 000 tokens, leaving headroom in the
    # 8 192-token llama3 context window for the system prompt + generated answer.
    generation_temperature: float = Field(0.1)
    generation_max_tokens: int = Field(1024)
    max_context_chars: int = Field(12_000)

    # Retrieval confidence threshold: below this → structured "I don't know"
    # response without calling the LLM (saves tokens, avoids hallucination).
    retrieval_confidence_threshold: float = Field(0.3)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
