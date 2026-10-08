"""Embedding-model initialization."""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings

logger = logging.getLogger(__name__)


def create_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Create the notebook's normalized local Hugging Face embedding model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize embeddings model '{settings.embedding_model_name}'."
        ) from exc
    logger.info("Initialized embeddings model: %s", settings.embedding_model_name)
    return embeddings

