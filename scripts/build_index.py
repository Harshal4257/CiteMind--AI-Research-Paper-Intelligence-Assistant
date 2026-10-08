"""Build the persistent FAISS index from PDFs in data/papers/."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunk_documents
from src.config import Settings
from src.embeddings import create_embeddings
from src.ingestion import load_research_papers
from src.logging_config import configure_logging
from src.preprocessing import clean_documents
from src.vectorstore import build_vector_store, save_vector_store


def main() -> int:
    settings = Settings.from_env(PROJECT_ROOT)
    configure_logging(settings.log_level)
    try:
        loaded = load_research_papers(settings.papers_dir)
        cleaned = clean_documents(loaded.documents)
        chunks = chunk_documents(cleaned, settings)
        embeddings = create_embeddings(settings)
        vector_store = build_vector_store(chunks, embeddings)
        save_vector_store(vector_store, settings.vector_store_dir)
    except Exception as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 1

    pdf_count = len({document.metadata["filename"] for document in loaded.documents})
    print(
        f"Index built successfully: {pdf_count} PDFs, {len(loaded.documents)} pages, "
        f"{len(chunks)} chunks.\nSaved to: {settings.vector_store_dir}"
    )
    if loaded.errors:
        print("Unreadable PDFs:")
        print("\n".join(f"- {error}" for error in loaded.errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

