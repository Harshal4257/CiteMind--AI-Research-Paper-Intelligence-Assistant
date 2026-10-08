"""Lightweight, technical-content-preserving document cleanup."""

from __future__ import annotations

import re

from langchain_core.documents import Document


def clean_page_text(text: str) -> str:
    """Normalize extraction whitespace without removing technical content."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    """Return non-empty cleaned copies, retaining every metadata field."""
    cleaned: list[Document] = []
    for document in documents:
        text = clean_page_text(document.page_content)
        if text:
            cleaned.append(Document(page_content=text, metadata=document.metadata.copy()))
    return cleaned

