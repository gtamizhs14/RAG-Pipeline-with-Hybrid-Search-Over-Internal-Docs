"""
Three chunking strategies for splitting documents into retrieval-ready pieces.

No single strategy wins across all document types:
  - fixed_size:        good baseline for homogeneous prose
  - recursive_header:  best for structured docs with markdown headings
  - semantic:          best for mixed-topic content without clear headings

The eval phase benchmarks all three against the golden Q&A set so you have
actual data to justify your chunking choice per document type.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE_HEADER = "recursive_header"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    content: str
    source: str
    doc_id: str
    chunk_index: int
    section_heading: str
    strategy: str
    char_count: int


class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        semantic_threshold: float = 0.5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold

    def chunk(
        self,
        doc_content: str,
        doc_source: str,
        doc_id: str,
        strategy: ChunkingStrategy,
        embed_fn=None,  # required only for SEMANTIC strategy
    ) -> list[Chunk]:
        if strategy == ChunkingStrategy.FIXED_SIZE:
            texts = self._fixed_size(doc_content)
            headings = ["N/A"] * len(texts)

        elif strategy == ChunkingStrategy.RECURSIVE_HEADER:
            pairs = self._recursive_header(doc_content)
            texts = [p[1] for p in pairs]
            headings = [p[0] for p in pairs]

        elif strategy == ChunkingStrategy.SEMANTIC:
            if embed_fn is None:
                raise ValueError("embed_fn is required for semantic chunking")
            texts = self._semantic(doc_content, embed_fn)
            headings = ["N/A"] * len(texts)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return [
            Chunk(
                content=text,
                source=doc_source,
                doc_id=doc_id,
                chunk_index=i,
                section_heading=headings[i],
                strategy=strategy.value,
                char_count=len(text),
            )
            for i, text in enumerate(texts)
            if text.strip()
        ]

    def _fixed_size(self, text: str) -> list[str]:
        # Overlap prevents a sentence split at a chunk boundary from being invisible
        # to both chunks. At 64/512, overhead is ~12%.
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        while start < len(text):
            chunk = text[start : start + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    def _recursive_header(self, text: str) -> list[tuple[str, str]]:
        # Split on markdown headings (# ## ###), then sub-split long sections
        # with fixed_size. This preserves section provenance in metadata.
        header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

        sections: list[tuple[str, str]] = []
        last_pos = 0
        current_heading = "Introduction"

        for match in header_re.finditer(text):
            body = text[last_pos : match.start()].strip()
            if body:
                sections.append((current_heading, body))
            current_heading = match.group(2).strip()
            last_pos = match.end()

        tail = text[last_pos:].strip()
        if tail:
            sections.append((current_heading, tail))

        result: list[tuple[str, str]] = []
        for heading, body in sections:
            if len(body) <= self.chunk_size:
                result.append((heading, body))
            else:
                for sub in self._fixed_size(body):
                    result.append((heading, sub))

        return result

    def _semantic(self, text: str, embed_fn) -> list[str]:
        # Find topic boundaries by measuring cosine similarity between consecutive
        # sentences. Where similarity drops below threshold, start a new chunk.
        # Produces topically coherent chunks at the cost of ~2x ingestion time.
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return self._fixed_size(text)

        embeddings = embed_fn(sentences)  # shape: (n, dim)

        similarities = [
            float(self._cosine(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < self.semantic_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))

        chunks = []
        for start, end in zip(boundaries, boundaries[1:]):
            group = " ".join(sentences[start:end]).strip()
            if not group:
                continue
            if len(group) > self.chunk_size * 2:
                chunks.extend(self._fixed_size(group))
            else:
                chunks.append(group)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Regex splitter avoids the NLTK punkt_tab download that fails in Docker
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0
