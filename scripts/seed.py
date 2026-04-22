"""
Seed script: copies documents from scripts/seed_corpus/ into data/raw/ and ingests them.

Run once before the first demo to populate the vector store:
    python scripts/seed.py

WHY seed_corpus/ lives in scripts/ not data/raw/:
  data/raw/ is gitignored so users can drop confidential documents there safely.
  scripts/seed_corpus/ is committed to the repo — it's demo content that anyone
  cloning can read, understand, and immediately query without needing their own docs.

WHY copy to data/raw/ before ingesting:
  IngestionPipeline watches data/raw/ as the canonical document source.
  Keeping seed docs in scripts/seed_corpus/ and copying on demand means
  the user's own documents in data/raw/ are never overwritten by accident.
  The seed can be re-run safely — duplicate chunks are filtered by the store.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline  # noqa: E402

SEED_DIR = Path(__file__).parent / "seed_corpus"
RAW_DIR = Path("data/raw")


def copy_documents() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = list(SEED_DIR.glob("*"))
    for src in files:
        shutil.copy2(src, RAW_DIR / src.name)
    print(f"Copied {len(files)} documents from {SEED_DIR}/ to {RAW_DIR}/")


def ingest_documents() -> None:
    print("Starting ingestion pipeline...")
    pipeline = IngestionPipeline()
    stats = pipeline.run(RAW_DIR)

    print(f"\nIngestion complete:")
    print(f"  Documents loaded : {stats.documents_loaded}")
    print(f"  Chunks created   : {stats.chunks_created}")
    print(f"  Chunks added     : {stats.chunks_added}")
    print(f"  Chunks skipped   : {stats.chunks_skipped_duplicate} (deduplication)")
    if stats.errors:
        print(f"  Errors           : {len(stats.errors)}")
        for err in stats.errors:
            print(f"    - {err}")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline — Seed Script")
    print("=" * 60)
    copy_documents()
    ingest_documents()
    print("\nDone. Start the API server and Streamlit UI to query the documents.")
    print("  uvicorn src.api.main:app --reload --port 8000")
    print("  streamlit run streamlit_app.py")
