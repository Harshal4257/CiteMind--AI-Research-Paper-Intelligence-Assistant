"""Shared exceptions and small helpers for the RAG backend."""


class RAGError(Exception):
    """Base exception for expected application errors."""


class DocumentLoadError(RAGError):
    """Raised when no usable PDF content can be loaded."""


class IndexNotFoundError(RAGError):
    """Raised when the persisted FAISS index is unavailable."""


class IndexLoadError(RAGError):
    """Raised when a persisted FAISS index cannot be loaded."""


class QueryValidationError(RAGError):
    """Raised when a user query fails an input guardrail."""

