"""
Chunking strategy comparison: runs the retrieval eval across all three
chunking strategies and prints a side-by-side report.

For each strategy:
  1. Ingest the seed corpus into a strategy-specific temp directory
  2. Run retrieval-only eval (no LLM calls) on all 55 questions
  3. Record Precision@K, Recall@K, MRR, NDCG@K, Hit Rate, chunk count

Each strategy gets its own isolated store directory so there are no
Windows file-lock conflicts between runs (ChromaDB holds HNSW locks
in-process; sharing a directory across runs causes PermissionError).

Usage:
    python scripts/compare_chunking.py
    python scripts/compare_chunking.py --output results/chunking_comparison.json
    python scripts/compare_chunking.py --k 3
"""

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.eval.retrieval_metrics import compute_all
from src.eval.runner import load_eval_dataset
from src.ingestion.chunker import ChunkingStrategy
from src.ingestion.embedder import EmbeddingModel
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.store import DocumentStore
from src.retrieval.hybrid import HybridRetriever

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

SEED_DIR = Path(__file__).parent / "seed_corpus"
RAW_DIR = Path("data/raw")
EVAL_DATASET = Path("eval_dataset.json")
COMPARE_BASE = Path("data/_compare")  # temp dirs live here; deleted on exit

STRATEGIES = [
    ChunkingStrategy.FIXED_SIZE,
    ChunkingStrategy.RECURSIVE_HEADER,
    ChunkingStrategy.SEMANTIC,
]


@dataclass
class StrategyResult:
    strategy: str
    chunks_indexed: int
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    mean_ndcg: float
    hit_rate: float
    mean_latency_ms: float
    k: int


def copy_seed_to_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for src in SEED_DIR.glob("*"):
        shutil.copy2(src, RAW_DIR / src.name)


def make_store(strategy: ChunkingStrategy) -> tuple[DocumentStore, Path, Path]:
    """Create a fresh DocumentStore in a strategy-specific temp directory."""
    base = COMPARE_BASE / strategy.value
    chroma_dir = base / "chroma_db"
    bm25_file = base / "bm25.pkl"
    base.mkdir(parents=True, exist_ok=True)

    # Temporarily redirect settings paths so DocumentStore uses our temp dirs.
    # This is safe because compare_chunking.py is a single-threaded script.
    settings.chroma_db_path = str(chroma_dir)
    settings.bm25_index_path = str(bm25_file)
    settings.chroma_collection_name = f"documents_{strategy.value}"

    return DocumentStore(), chroma_dir, bm25_file


def run_strategy(
    strategy: ChunkingStrategy,
    samples,
    k: int,
    embedder: EmbeddingModel,
) -> StrategyResult:
    print(f"  Strategy: {strategy.value}")

    store, _, _ = make_store(strategy)

    # Ingest
    print(f"    Ingesting...", end=" ", flush=True)
    t0 = time.perf_counter()
    pipeline = IngestionPipeline(embedder=embedder, store=store)
    stats = pipeline.run(RAW_DIR, strategy=strategy)
    print(f"{stats.chunks_added} chunks in {time.perf_counter() - t0:.1f}s")

    # Retriever uses the same store (already loaded in memory)
    retriever = HybridRetriever(store=store, embedder=embedder)

    # Eval
    print(f"    Evaluating {len(samples)} questions...", end=" ", flush=True)
    precisions, recalls, mrrs, ndcgs, hits, latencies = [], [], [], [], [], []

    for sample in samples:
        t1 = time.perf_counter()
        retrieved = retriever.retrieve(sample.question, top_n=k)
        latency_ms = (time.perf_counter() - t1) * 1_000

        ids = [r.doc_id for r in retrieved]
        m = compute_all(ids, sample.relevant_doc_ids, k)
        precisions.append(m["precision_at_k"])
        recalls.append(m["recall_at_k"])
        mrrs.append(m["mrr"])
        ndcgs.append(m["ndcg_at_k"])
        hits.append(1.0 if m["recall_at_k"] > 0 else 0.0)
        latencies.append(latency_ms)

    n = len(samples)
    result = StrategyResult(
        strategy=strategy.value,
        chunks_indexed=stats.chunks_added,
        mean_precision=round(sum(precisions) / n, 4),
        mean_recall=round(sum(recalls) / n, 4),
        mean_mrr=round(sum(mrrs) / n, 4),
        mean_ndcg=round(sum(ndcgs) / n, 4),
        hit_rate=round(sum(hits) / n, 4),
        mean_latency_ms=round(sum(latencies) / n, 1),
        k=k,
    )
    print(f"done  MRR={result.mean_mrr:.3f}  Hit={result.hit_rate:.3f}")
    return result


def print_report(results: list[StrategyResult]) -> None:
    k = results[0].k
    sep = "-" * 74
    col = 20

    print(f"\n{sep}")
    print(f"  Chunking Strategy Comparison  (k={k}, n={results[0].chunks_indexed + 0} ... )")
    print(f"  Dataset: {EVAL_DATASET}  |  Corpus: {SEED_DIR.name}/")
    print(sep)

    header = f"  {'Metric':<22}"
    for r in results:
        header += f"  {r.strategy:>{col}}"
    print(header)
    print(f"  {'-'*22}" + f"  {'-'*col}" * len(results))

    rows = [
        ("Chunks indexed",    "chunks_indexed",  "d"),
        (f"Precision@{k}",    "mean_precision",  ".4f"),
        (f"Recall@{k}",       "mean_recall",     ".4f"),
        ("MRR",               "mean_mrr",        ".4f"),
        (f"NDCG@{k}",         "mean_ndcg",       ".4f"),
        ("Hit Rate",          "hit_rate",        ".4f"),
        ("Latency/query (ms)","mean_latency_ms", ".1f"),
    ]
    for label, attr, fmt in rows:
        vals = [getattr(r, attr) for r in results]
        # Mark best for quality metrics (not latency or chunk count)
        best = max(vals) if attr not in ("mean_latency_ms", "chunks_indexed") else None
        row = f"  {label:<22}"
        for r in results:
            v = getattr(r, attr)
            cell = f"{v:{fmt}}"
            marker = " *" if (best is not None and v == best) else "  "
            row += f"  {cell + marker:>{col}}"
        print(row)

    print(sep)
    print("  * = best value for that metric\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies on the retrieval eval dataset."
    )
    parser.add_argument("--k", type=int, default=settings.rerank_top_n,
                        help="Top-k for retrieval metrics")
    parser.add_argument("--output", default=None,
                        help="Save JSON report to this path")
    args = parser.parse_args()

    if not EVAL_DATASET.exists():
        print(f"ERROR: {EVAL_DATASET} not found. Run from the project root.")
        sys.exit(1)
    if not SEED_DIR.exists():
        print(f"ERROR: {SEED_DIR} not found.")
        sys.exit(1)

    samples = load_eval_dataset(EVAL_DATASET)
    print(f"\nLoaded {len(samples)} eval samples  k={args.k}")
    print(f"Seed corpus: {SEED_DIR}\n")

    # Clean up any leftover temp dirs from a previous run (locks are gone now)
    if COMPARE_BASE.exists():
        shutil.rmtree(COMPARE_BASE, ignore_errors=True)

    copy_seed_to_raw()
    embedder = EmbeddingModel()  # shared across strategies — loaded once

    results = []
    for strategy in STRATEGIES:
        results.append(run_strategy(strategy, samples, args.k, embedder))

    # Restore production settings paths and rebuild default index.
    # (Temp _compare dirs are left on disk; Windows holds the HNSW locks until
    # process exit. They'll be cleaned on the next run via ignore_errors above.)
    settings.chroma_db_path = "data/chroma_db"
    settings.bm25_index_path = "data/bm25_index.pkl"
    settings.chroma_collection_name = "documents"
    print("Restoring default index (fixed_size)...")
    store = DocumentStore()
    IngestionPipeline(embedder=embedder, store=store).run(RAW_DIR, ChunkingStrategy.FIXED_SIZE)
    print("Done.\n")

    print_report(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                [
                    {
                        "strategy": r.strategy,
                        "k": r.k,
                        "chunks_indexed": r.chunks_indexed,
                        "precision_at_k": r.mean_precision,
                        "recall_at_k": r.mean_recall,
                        "mrr": r.mean_mrr,
                        "ndcg_at_k": r.mean_ndcg,
                        "hit_rate": r.hit_rate,
                        "mean_latency_ms": r.mean_latency_ms,
                    }
                    for r in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
