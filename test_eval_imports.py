"""
Smoke test: verifies eval module imports and metric math without a live index or LLM.
Run: python test_eval_imports.py
"""
import os
os.environ.setdefault("GROQ_API_KEY", "dummy-for-smoke-test")

from src.eval import EvalSample, EvalReport, EvalResult, EvalRunner, load_eval_dataset
from src.eval.retrieval_metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k, compute_all
from src.eval.models import RetrievalEval, GenerationEval

print("All eval imports OK")

# ── Precision@K ──────────────────────────────────────────────────────────────
retrieved = ["d1", "d2", "d3", "d4", "d5"]
relevant  = ["d1", "d3"]

p3 = precision_at_k(retrieved, relevant, k=3)
assert abs(p3 - 2/3) < 1e-9, f"P@3 expected 0.667, got {p3}"
p5 = precision_at_k(retrieved, relevant, k=5)
assert abs(p5 - 2/5) < 1e-9, f"P@5 expected 0.4, got {p5}"
print(f"precision_at_k OK: P@3={p3:.4f}, P@5={p5:.4f}")

# ── Recall@K ─────────────────────────────────────────────────────────────────
r3 = recall_at_k(retrieved, relevant, k=3)
assert abs(r3 - 1.0) < 1e-9, f"R@3 expected 1.0, got {r3}"  # both in top-3
r1 = recall_at_k(retrieved, relevant, k=1)
assert abs(r1 - 0.5) < 1e-9, f"R@1 expected 0.5, got {r1}"
print(f"recall_at_k OK: R@1={r1:.4f}, R@3={r3:.4f}")

# ── MRR ──────────────────────────────────────────────────────────────────────
m = mrr(retrieved, relevant)
assert abs(m - 1.0) < 1e-9, f"MRR expected 1.0 (d1 at rank 1), got {m}"
m_miss = mrr(["d9", "d8"], relevant)
assert m_miss == 0.0, f"MRR expected 0.0 on no hit, got {m_miss}"
print(f"mrr OK: {m:.4f} (perfect), {m_miss:.4f} (miss)")

# ── NDCG@K ───────────────────────────────────────────────────────────────────
import math
ndcg = ndcg_at_k(retrieved, relevant, k=5)
assert 0.0 <= ndcg <= 1.0, f"NDCG@5 out of range: {ndcg}"
ndcg_perfect = ndcg_at_k(["d1", "d3", "d9"], relevant, k=2)
assert abs(ndcg_perfect - 1.0) < 1e-9, f"Perfect NDCG@2 expected 1.0, got {ndcg_perfect}"
print(f"ndcg_at_k OK: NDCG@5={ndcg:.4f}, perfect NDCG@2={ndcg_perfect:.4f}")

# ── compute_all ───────────────────────────────────────────────────────────────
all_metrics = compute_all(retrieved, relevant, k=5)
assert set(all_metrics.keys()) == {"precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"}
print(f"compute_all OK: {all_metrics}")

# ── EvalSample dataclass ─────────────────────────────────────────────────────
sample = EvalSample(
    id="q1",
    question="What is RAG?",
    ground_truth_answer="RAG combines retrieval and generation.",
    relevant_doc_ids=["doc_a"],
)
assert sample.id == "q1"
print("EvalSample dataclass OK")

# ── load_eval_dataset ─────────────────────────────────────────────────────────
samples = load_eval_dataset("eval_dataset.json")
assert len(samples) == 3
assert samples[0].id == "q1"
print(f"load_eval_dataset OK: {len(samples)} samples loaded")

print("\nAll eval checks passed.")
print("(EvalRunner.run() will be exercised when called with a live index and Groq key.)")
