"""
Dual index: ChromaDB (dense vector) + BM25 (sparse keyword).

Both indexes are kept in sync on every write. Dense search handles semantic
similarity ("car accident" matches "vehicle collision"). Sparse search handles
exact term matching ("LangGraph v0.2.1", "errno ECONNREFUSED"). Technical docs
need both.

BM25 is stored as a pickle because rank_bm25 doesn't have a native persistence
format. It rebuilds from the corpus on every new batch — rebuilding is cheap
(milliseconds for <100k chunks) and keeps IDF scores correct.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Shared tokenizer used at index time and query time. Must be identical in both places."""
    return text.lower().split()


class DocumentStore:

    def __init__(self):
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            Path(settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=settings.chroma_db_path)
            self._collection = client.get_or_create_collection(
                name=settings.chroma_collection_name,
                # Cosine space: distance = 1 - similarity. Embeddings are L2-normalized
                # so dot product already equals cosine similarity, but being explicit here
                # avoids confusion if someone reads the ChromaDB config later.
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Collection '{settings.chroma_collection_name}' "
                f"({self._collection.count()} existing chunks)"
            )
        return self._collection

    def _load_bm25(self) -> dict:
        p = Path(settings.bm25_index_path)
        if p.exists():
            with open(p, "rb") as f:
                data = pickle.load(f)
            logger.info(f"BM25 loaded ({len(data['doc_ids'])} entries)")
            return data
        return {"corpus": [], "doc_ids": [], "bm25": None}

    def _save_bm25(self, data: dict) -> None:
        p = Path(settings.bm25_index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(data, f)

    def _rebuild_bm25(self, corpus: list[list[str]]):
        from rank_bm25 import BM25Okapi
        return BM25Okapi(corpus)

    def is_duplicate(self, embedding: np.ndarray) -> bool:
        """
        True if the nearest existing chunk has cosine similarity >= dedup threshold.

        ChromaDB cosine distance = 1 - similarity, so similarity = 1 - distance.
        """
        if self.collection.count() == 0:
            return False

        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1,
            include=["distances"],
        )

        if not results["distances"] or not results["distances"][0]:
            return False

        similarity = 1.0 - results["distances"][0][0]
        return similarity >= settings.dedup_similarity_threshold

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> dict:
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}")

        bm25_data = self._load_bm25()
        added = 0
        skipped = 0

        ids, vecs, docs, metas, new_corpus, new_doc_ids = [], [], [], [], [], []

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{chunk.doc_id}_{chunk.chunk_index}_{chunk.strategy}"

            if self.is_duplicate(emb):
                skipped += 1
                continue

            ids.append(chunk_id)
            vecs.append(emb.tolist())
            docs.append(chunk.content)
            metas.append({
                "source": chunk.source,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "section_heading": chunk.section_heading,
                "strategy": chunk.strategy,
                "char_count": chunk.char_count,
            })
            new_corpus.append(tokenize(chunk.content))
            new_doc_ids.append(chunk_id)
            added += 1

        if ids:
            self.collection.add(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
            logger.info(f"Added {len(ids)} chunks to ChromaDB")

            bm25_data["corpus"].extend(new_corpus)
            bm25_data["doc_ids"].extend(new_doc_ids)
            bm25_data["bm25"] = self._rebuild_bm25(bm25_data["corpus"])
            self._save_bm25(bm25_data)
            logger.info(f"BM25 rebuilt ({len(bm25_data['doc_ids'])} total entries)")

        return {
            "added": added,
            "skipped_duplicate": skipped,
            "total_in_store": self.collection.count(),
        }

    def get_stats(self) -> dict:
        bm25_data = self._load_bm25()
        return {
            "chroma_count": self.collection.count(),
            "bm25_count": len(bm25_data["doc_ids"]),
        }

    def list_documents(self) -> list[dict]:
        """Return one record per unique source document (grouped by doc_id)."""
        if self.collection.count() == 0:
            return []

        result = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in result["metadatas"]:
            doc_id = meta.get("doc_id", "")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", ""),
                    "chunk_count": 1,
                }
            else:
                seen[doc_id]["chunk_count"] += 1
        return list(seen.values())
