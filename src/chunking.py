"""Notebook-compatible recursive chunking with stable provenance metadata."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings

logger = logging.getLogger(__name__)


def chunk_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split documents and add globally unique, readable chunk identifiers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata
        metadata["chunk_id"] = (
            f"{metadata['paper_id']}::p{metadata['page_number']}::c{index:05d}"
        )
    logger.info("Created %s chunks", len(chunks))
    return chunks

