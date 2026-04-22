"""
CLI entry point for the evaluation framework.

Usage:
    python run_eval.py                              # full eval (retrieval + generation)
    python run_eval.py --retrieval-only             # skip LLM generation scoring
    python run_eval.py --dataset path/to/eval.json  # custom dataset path
    python run_eval.py --k 3                        # evaluate at P@3, R@3, NDCG@3
    python run_eval.py --output results/eval_out.json

The report is printed to stdout as a formatted summary and optionally saved to
JSON (--output flag). JSON output includes per-sample breakdowns for debugging.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _print_report(report) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  RAG Pipeline Evaluation Report")
    print(sep)
    print(f"  Samples : {report.num_samples}")
    print(f"  K       : {report.k}")
    print(sep)
    print("  RETRIEVAL METRICS")
    print(f"    Precision@{report.k}  : {report.mean_precision_at_k:.4f}")
    print(f"    Recall@{report.k}     : {report.mean_recall_at_k:.4f}")
    print(f"    MRR          : {report.mean_mrr:.4f}")
    print(f"    NDCG@{report.k}       : {report.mean_ndcg_at_k:.4f}")
    print(f"    Hit Rate     : {report.hit_rate:.4f}")
    print(sep)
    if report.mean_faithfulness is not None:
        print("  GENERATION METRICS")
        print(f"    Faithfulness   : {report.mean_faithfulness:.4f}")
        print(f"    Answer Rel.    : {report.mean_answer_relevance:.4f}")
        print(f"    Gen Composite  : {report.mean_generation_composite:.4f}")
        print(sep)
    print("  PIPELINE METRICS")
    print(f"    Confidence (composite) : {report.mean_confidence_composite:.4f}")
    print(f"    Mean Latency (ms)      : {report.mean_latency_ms:.1f}")
    print(sep)

    print("\n  PER-SAMPLE BREAKDOWN")
    print(f"  {'ID':<20} {'P@K':>6} {'R@K':>6} {'MRR':>6} {'NDCG':>6} {'Faith':>6} {'Rel':>6} {'ms':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for r in report.results:
        ret = r.retrieval
        gen = r.generation
        faith = f"{gen.faithfulness:.3f}" if gen else "  —  "
        rel   = f"{gen.answer_relevance:.3f}" if gen else "  —  "
        print(
            f"  {r.sample.id:<20} "
            f"{ret.precision_at_k:>6.3f} "
            f"{ret.recall_at_k:>6.3f} "
            f"{ret.mrr:>6.3f} "
            f"{ret.ndcg_at_k:>6.3f} "
            f"{faith:>6} "
            f"{rel:>6} "
            f"{r.latency_ms:>7.0f}"
        )
    print()


def _report_to_dict(report) -> dict:
    """Convert EvalReport to a JSON-serialisable dict."""
    d = {
        "num_samples": report.num_samples,
        "k": report.k,
        "retrieval": {
            "mean_precision_at_k": report.mean_precision_at_k,
            "mean_recall_at_k": report.mean_recall_at_k,
            "mean_mrr": report.mean_mrr,
            "mean_ndcg_at_k": report.mean_ndcg_at_k,
            "hit_rate": report.hit_rate,
        },
        "generation": {
            "mean_faithfulness": report.mean_faithfulness,
            "mean_answer_relevance": report.mean_answer_relevance,
            "mean_generation_composite": report.mean_generation_composite,
        },
        "pipeline": {
            "mean_confidence_composite": report.mean_confidence_composite,
            "mean_latency_ms": report.mean_latency_ms,
        },
        "samples": [],
    }
    for r in report.results:
        sample_dict = {
            "id": r.sample.id,
            "question": r.sample.question,
            "answer": r.answer,
            "latency_ms": r.latency_ms,
            "confidence_composite": r.confidence_composite,
            "retrieval": {
                "precision_at_k": r.retrieval.precision_at_k,
                "recall_at_k": r.retrieval.recall_at_k,
                "mrr": r.retrieval.mrr,
                "ndcg_at_k": r.retrieval.ndcg_at_k,
                "hit": r.retrieval.hit,
                "retrieved_doc_ids": r.retrieval.retrieved_doc_ids,
                "relevant_doc_ids": r.retrieval.relevant_doc_ids,
            },
        }
        if r.generation:
            sample_dict["generation"] = {
                "faithfulness": r.generation.faithfulness,
                "faithfulness_reason": r.generation.faithfulness_reason,
                "answer_relevance": r.generation.answer_relevance,
                "answer_relevance_reason": r.generation.answer_relevance_reason,
                "composite_score": r.generation.composite_score,
            }
        d["samples"].append(sample_dict)
    return d


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on a test dataset.")
    parser.add_argument(
        "--dataset",
        default="eval_dataset.json",
        help="Path to the eval dataset JSON file (default: eval_dataset.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Top-k for retrieval metrics (default: RERANK_TOP_N from config)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation quality scoring (faster, no extra LLM calls)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the full JSON report",
    )
    args = parser.parse_args()

    # Import here so the module-level logging is already configured.
    from src.eval.runner import EvalRunner, load_eval_dataset

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    samples = load_eval_dataset(dataset_path)
    runner = EvalRunner(
        k=args.k,
        eval_generation=not args.retrieval_only,
    )

    logger.info(
        f"Starting eval: {len(samples)} samples, k={runner.k}, "
        f"generation_eval={not args.retrieval_only}"
    )
    report = runner.run(samples)
    _print_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(_report_to_dict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Full report saved to {out_path}")


if __name__ == "__main__":
    main()
