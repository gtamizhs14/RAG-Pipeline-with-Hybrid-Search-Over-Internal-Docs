"""
Smoke test: verifies the generation module imports cleanly and the core logic
works without a live Groq API key or a populated index.
Run: python test_generation_imports.py
"""
import os
os.environ.setdefault("GROQ_API_KEY", "dummy-for-smoke-test")

from src.generation import RAGPipeline, RAGResponse, CitedSource, VerifiedCitation, AnswerConfidence
from src.generation.prompt import PromptBuilder, SYSTEM_PROMPT
from src.generation.citations import CitationParser
from src.generation.models import CitedSource, VerifiedCitation, AnswerConfidence, RAGResponse
from src.generation.verifier import CitationVerifier, _extract_claim_for
from src.generation.scorer import AnswerConfidenceScorer
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

# ── VerifiedCitation dataclass ───────────────────────────────────────────────
vc = VerifiedCitation(
    citation_number=1,
    chunk_id="a",
    source="doc1.pdf",
    doc_id="d1",
    content="Alpha content here.",
    score=0.9,
    verified=True,
    verification_reason="Chunk directly states alpha.",
)
assert vc.verified is True
assert vc.verification_reason == "Chunk directly states alpha."
print("VerifiedCitation dataclass OK")

# ── AnswerConfidence dataclass ───────────────────────────────────────────────
conf = AnswerConfidence(
    retrieval_confidence=0.85,
    citation_coverage=1.0,
    completeness_score=0.8,
    composite_score=round(0.35 * 0.85 + 0.40 * 1.0 + 0.25 * 0.8, 4),
)
assert 0.0 <= conf.composite_score <= 1.0
print(f"AnswerConfidence dataclass OK: composite={conf.composite_score}")

# ── _extract_claim_for (verifier helper) ──────────────────────────────────────
claim = _extract_claim_for(1, "Alpha is fast [1]. Beta is slow [2].")
assert "[1]" in claim
assert "Alpha is fast" in claim
assert "Beta" not in claim, "Should only return sentence(s) containing [1]"
print(f"_extract_claim_for OK: {claim!r}")

# ── AnswerConfidenceScorer._retrieval_confidence (static, no LLM needed) ─────
sources = [r1, r2, r3]  # scores: 0.9, 0.8, 0.7
rc = AnswerConfidenceScorer._retrieval_confidence(sources)
assert abs(rc - (0.9 + 0.8 + 0.7) / 3) < 1e-6
print(f"AnswerConfidenceScorer._retrieval_confidence OK: {rc:.4f}")

# ── AnswerConfidenceScorer._citation_coverage (static, no LLM needed) ────────
vc_true  = VerifiedCitation("1","a","s","d","c",0.9, verified=True,  verification_reason="ok")
vc_false = VerifiedCitation("2","b","s","d","c",0.8, verified=False, verification_reason="no")
vc_none  = VerifiedCitation("3","c","s","d","c",0.7, verified=None,  verification_reason="err")
cov = AnswerConfidenceScorer._citation_coverage([vc_true, vc_false, vc_none])
assert abs(cov - (1.0 + 0.0 + 0.5) / 3) < 1e-6
print(f"AnswerConfidenceScorer._citation_coverage OK: {cov:.4f}")

print("\nAll checks passed. RAGPipeline is ready to use.")
print("(Groq API and ChromaDB will be exercised when answer() is called with a real key and index.)")
