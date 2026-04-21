"""
Smoke test: verifies the generation module imports cleanly and the core logic
works without a live Groq API key or a populated index.
Run: python test_generation_imports.py
"""
import os
os.environ.setdefault("GROQ_API_KEY", "dummy-for-smoke-test")

from src.generation import RAGPipeline, RAGResponse, CitedSource
from src.generation.prompt import PromptBuilder, SYSTEM_PROMPT
from src.generation.citations import CitationParser
from src.generation.models import CitedSource, RAGResponse
from src.retrieval.models import SearchResult

print("All generation imports OK")

# ── PromptBuilder ────────────────────────────────────────────────────────────
builder = PromptBuilder(max_context_chars=500)

r1 = SearchResult(chunk_id="a", content="Alpha content here.", source="doc1.pdf", doc_id="d1", score=0.9)
r2 = SearchResult(chunk_id="b", content="Beta content here.", source="doc2.pdf", doc_id="d2", score=0.8)
r3 = SearchResult(chunk_id="c", content="Gamma content here.", source="doc3.pdf", doc_id="d3", score=0.7)

user_msg, used = builder.build("What is alpha?", [r1, r2, r3])

assert "[1]" in user_msg, "Chunk 1 label missing from prompt"
assert "[2]" in user_msg, "Chunk 2 label missing from prompt"
assert "Alpha content here." in user_msg, "Chunk 1 content missing"
assert "What is alpha?" in user_msg, "Question missing from prompt"
assert len(used) <= 3, "used_results should be a subset of input"
print(f"PromptBuilder OK: {len(used)} chunks included in prompt")

# ── CitationParser ───────────────────────────────────────────────────────────
parser = CitationParser()

answer_text = "Alpha is described in [1]. Beta is mentioned in [2]. Unknown is in [9]."
# Pass a small used_results list (2 real chunks, so [9] is phantom)
used_two = [r1, r2]
cited = parser.parse(answer_text, used_two)

assert len(cited) == 2, f"Expected 2 valid citations, got {len(cited)}"
assert cited[0].citation_number == 1
assert cited[0].chunk_id == "a"
assert cited[1].citation_number == 2
assert cited[1].chunk_id == "b"
print(f"CitationParser OK: {[c.citation_number for c in cited]} — phantom [9] correctly dropped")

# ── Deduplication ─────────────────────────────────────────────────────────────
answer_repeat = "See [1] and also [1] again and [2]."
cited_dedup = parser.parse(answer_repeat, used_two)
assert len(cited_dedup) == 2, f"Expected 2 after dedup, got {len(cited_dedup)}"
print("CitationParser deduplication OK")

# ── SYSTEM_PROMPT sanity ─────────────────────────────────────────────────────
assert "cite" in SYSTEM_PROMPT.lower(), "SYSTEM_PROMPT should mention citation"
assert "[number]" in SYSTEM_PROMPT or "[n]" in SYSTEM_PROMPT.lower() or "number" in SYSTEM_PROMPT.lower()
print("SYSTEM_PROMPT sanity OK")

print("\nAll checks passed. RAGPipeline is ready to use.")
print("(Groq API and ChromaDB will be exercised when answer() is called with a real key and index.)")
