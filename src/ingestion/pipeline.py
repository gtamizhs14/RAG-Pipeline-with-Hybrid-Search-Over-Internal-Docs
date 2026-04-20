"""
Ingestion pipeline: loads documents, chunks them, embeds, deduplicates, stores.

Wires the four components together so callers don't have to manage the sequence
themselves. Each component is injected so tests can substitute mocks without
touching this file.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.ingestion.chunker import ChunkingStrategy, DocumentChunker
from src.ingestion.embedder import EmbeddingModel
from src.ingestion.loader import DocumentLoader
from src.ingestion.store import DocumentStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    documents_loaded: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    chunks_added: int = 0
    chunks_skipped_duplicate: int = 0
    strategy_used: str = ""
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:

    def __init__(
        self,
        loader: DocumentLoader = None,
        chunker: DocumentChunker = None,
        embedder: EmbeddingModel = None,
        store: DocumentStore = None,
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            semantic_threshold=settings.semantic_similarity_threshold,
        )
        self.embedder = embedder or EmbeddingModel()
        self.store = store or DocumentStore()

    def run(
        self,
        source: Path,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
    ) -> IngestionStats:
        stats = IngestionStats(strategy_used=strategy.value)

        # Step 1: load
        source_path = Path(source)
        if source_path.is_dir():
            documents = self.loader.load_directory(source_path)
        elif source_path.is_file():
            doc = self.loader.load_file(source_path)
            documents = [doc] if doc else []
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        stats.documents_loaded = len(documents)
        if not documents:
            logger.warning("No documents loaded.")
            return stats

        # Step 2: chunk
        all_chunks = []
        # Only pass embed_fn for semantic strategy — avoids loading the model otherwise
        embed_fn = self.embedder.embed if strategy == ChunkingStrategy.SEMANTIC else None

        for doc in documents:
            try:
                chunks = self.chunker.chunk(
                    doc_content=doc.content,
                    doc_source=doc.source,
                    doc_id=doc.doc_id,
                    strategy=strategy,
                    embed_fn=embed_fn,
                )
                all_chunks.extend(chunks)
                logger.debug(f"{doc.title}: {len(chunks)} chunks")
            except Exception as e:
                msg = f"Failed to chunk {doc.source}: {e}"
                logger.error(msg)
                stats.documents_failed += 1
                stats.errors.append(msg)

        stats.chunks_created = len(all_chunks)
        if not all_chunks:
            logger.warning("No chunks produced.")
            return stats

        # Step 3: embed (single batched call — much faster than per-chunk)
        logger.info(f"Embedding {len(all_chunks)} chunks...")
        embeddings = self.embedder.embed([c.content for c in all_chunks])

        # Step 4: store with dedup
        logger.info("Writing to store...")
        result = self.store.add_chunks(all_chunks, embeddings)

        stats.chunks_added = result["added"]
        stats.chunks_skipped_duplicate = result["skipped_duplicate"]
        logger.info(
            f"Done: {stats.chunks_added} added, {stats.chunks_skipped_duplicate} skipped, "
            f"{result['total_in_store']} total in store"
        )
        return stats
