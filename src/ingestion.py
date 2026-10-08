"""Load research-paper PDFs while preserving source metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.utils import DocumentLoadError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    """Documents loaded successfully plus non-fatal per-file errors."""

    documents: list[Document]
    errors: list[str]


def load_research_papers(papers_dir: Path) -> IngestionResult:
    """Load every PDF in *papers_dir*, preserving notebook-compatible metadata."""
    if not papers_dir.exists():
        raise DocumentLoadError(f"Papers folder does not exist: {papers_dir}")

    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if not pdf_paths:
        raise DocumentLoadError(
            f"No PDFs found in {papers_dir}. Add one or more research-paper PDFs and rebuild the index."
        )

    all_pages: list[Document] = []
    errors: list[str] = []
    for pdf_path in pdf_paths:
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
            for page in pages:
                page.metadata.update(
                    {
                        "filename": pdf_path.name,
                        "page_number": int(page.metadata.get("page", 0)) + 1,
                        "source_path": str(pdf_path.resolve()),
                        "paper_id": pdf_path.stem,
                    }
                )
            all_pages.extend(pages)
            logger.info("Loaded %s page(s) from %s", len(pages), pdf_path.name)
        except Exception as exc:  # A bad PDF should not prevent other papers loading.
            message = f"Could not read {pdf_path.name}: {exc}"
            errors.append(message)
            logger.warning(message)

    if not all_pages:
        details = " | ".join(errors) or "No pages were returned by the PDF loader."
        raise DocumentLoadError(f"No readable pages were loaded. {details}")

    logger.info(
        "Loaded %s page(s) from %s PDF(s)",
        len(all_pages),
        len({document.metadata["filename"] for document in all_pages}),
    )
    return IngestionResult(documents=all_pages, errors=errors)

