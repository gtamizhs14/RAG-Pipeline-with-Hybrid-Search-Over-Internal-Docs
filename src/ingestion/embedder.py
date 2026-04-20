"""
Local embedding model wrapper using Sentence Transformers.

Running embeddings locally means zero API cost, no network latency, and documents
never leave the machine. all-MiniLM-L6-v2 gets ~90% of OpenAI's quality at 0% cost.
Model name and dimension are read from .env so swapping to a different model is one
line change.
"""

import logging
from functools import cached_property

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model

    @cached_property
    def _model(self):
        # Lazy load: ~2s startup cost, paid once, not on every import
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        logger.info(f"Model ready (dim={model.get_sentence_embedding_dimension()})")
        return model

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Returns shape (len(texts), embedding_dim)."""
        if not texts:
            return np.empty((0, settings.embedding_dimension))

        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            # L2-normalize so cosine similarity = dot product (simpler dedup math)
            normalize_embeddings=True,
        )

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper. Returns shape (embedding_dim,)."""
        return self.embed([text])[0]
