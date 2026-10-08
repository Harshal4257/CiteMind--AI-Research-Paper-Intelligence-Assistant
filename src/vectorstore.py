"""Build, persist, and reload the local FAISS vector index."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.utils import IndexLoadError, IndexNotFoundError

logger = logging.getLogger(__name__)

_FAISS_FILES = ("index.faiss", "index.pkl")


def index_exists(index_dir: Path) -> bool:
    """Return whether both files needed for a FAISS index are present."""
    return all((index_dir / filename).is_file() for filename in _FAISS_FILES)


def build_vector_store(documents: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed chunks and create an in-memory FAISS index."""
    if not documents:
        raise ValueError("Cannot build a vector store from zero document chunks.")
    logger.info("Building FAISS index from %s chunks", len(documents))
    return FAISS.from_documents(documents, embeddings)


def save_vector_store(vector_store: FAISS, index_dir: Path) -> None:
    """Persist a FAISS index locally."""
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    logger.info("Saved FAISS index to %s", index_dir)


def load_vector_store(index_dir: Path, embeddings: Embeddings) -> FAISS:
    """Load a locally created FAISS index; never load untrusted index files."""
    if not index_exists(index_dir):
        raise IndexNotFoundError(
            f"No complete vector index at {index_dir}. Run: python scripts/build_index.py"
        )
    try:
        # LangChain requires this acknowledgement because FAISS metadata uses pickle.
        # The application only loads the index it created in its own storage folder.
        vector_store = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
    except Exception as exc:
        raise IndexLoadError(
            f"Could not load the vector index at {index_dir}. Rebuild it with scripts/build_index.py."
        ) from exc
    logger.info("Loaded FAISS index from %s", index_dir)
    return vector_store

