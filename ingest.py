#!/usr/bin/env python3
"""
CLI for ingesting documents into the RAG pipeline.

Usage:
    python ingest.py
    python ingest.py --strategy recursive_header
    python ingest.py --strategy semantic
    python ingest.py --source path/to/specific/doc.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.config import settings
from src.ingestion.chunker import ChunkingStrategy
from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(settings.raw_docs_path),
        help=f"File or directory to ingest (default: {settings.raw_docs_path})",
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkingStrategy],
        default=ChunkingStrategy.FIXED_SIZE.value,
    )
    args = parser.parse_args()

    strategy = ChunkingStrategy(args.strategy)
    pipeline = IngestionPipeline()

    print(f"\nIngesting from: {args.source}")
    print(f"Strategy: {strategy.value}")
    print("-" * 40)

    stats = pipeline.run(source=args.source, strategy=strategy)

    print(f"\nDone.")
    print(f"  Documents loaded:   {stats.documents_loaded}")
    print(f"  Documents failed:   {stats.documents_failed}")
    print(f"  Chunks created:     {stats.chunks_created}")
    print(f"  Chunks added:       {stats.chunks_added}")
    print(f"  Duplicates skipped: {stats.chunks_skipped_duplicate}")

    if stats.errors:
        print("\nErrors:")
        for err in stats.errors:
            print(f"  {err}")

    sys.exit(1 if stats.documents_loaded == 0 and stats.documents_failed > 0 else 0)


if __name__ == "__main__":
    main()
