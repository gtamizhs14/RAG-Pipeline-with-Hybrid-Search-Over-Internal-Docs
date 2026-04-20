"""
Smoke test: verifies the retrieval module imports cleanly and the class
hierarchy is wired correctly. Does NOT require a populated index.
Run: python test_retrieval_imports.py
"""
import os
# Provide a dummy key so pydantic-settings validation passes without a real .env
os.environ.setdefault("GROQ_API_KEY", "dummy-for-smoke-test")

from src.retrieval import HybridRetriever, RetrievalTrace, SearchResult
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

print("All retrieval imports OK")

# Verify RRF with synthetic data (no index needed)
r1 = SearchResult(chunk_id="a", content="foo", source="s", doc_id="d", score=0.9)
r2 = SearchResult(chunk_id="b", content="bar", source="s", doc_id="d", score=0.8)
r3 = SearchResult(chunk_id="c", content="baz", source="s", doc_id="d", score=0.7)

dense = [r1, r2]       # chunk 'a' rank 1, 'b' rank 2
sparse = [r2, r3]      # chunk 'b' rank 1, 'c' rank 2

fused = reciprocal_rank_fusion(dense, sparse, k=60, dense_weight=0.7, sparse_weight=0.3, top_n=3)

assert len(fused) == 3, f"Expected 3, got {len(fused)}"
# 'b' appears in both lists so should score higher than 'a' (dense only) and 'c' (sparse only)
assert fused[0].chunk_id == "b", f"Expected 'b' first (appears in both lists), got '{fused[0].chunk_id}'"
print(f"RRF order: {[r.chunk_id for r in fused]}  scores: {[round(r.score, 5) for r in fused]}")
print("RRF fusion logic verified")

print("\nAll checks passed. HybridRetriever is ready to use.")
print("(Cross-encoder model will lazy-load on first .rerank() call)")
