"""Practical input validation used by the RAG application."""

from __future__ import annotations

from src.utils import QueryValidationError


def validate_query(query: object, max_query_chars: int) -> str:
    """Normalize a query and reject empty or unexpectedly large inputs."""
    normalized = query.strip() if isinstance(query, str) else ""
    if not normalized:
        raise QueryValidationError("Question cannot be empty.")
    if len(normalized) > max_query_chars:
        raise QueryValidationError(
            f"Question exceeds the {max_query_chars}-character limit."
        )
    return normalized

