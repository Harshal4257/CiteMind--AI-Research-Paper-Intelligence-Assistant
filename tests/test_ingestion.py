from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion import load_research_papers
from src.utils import DocumentLoadError


def test_load_research_papers_preserves_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    class FakeLoader:
        def __init__(self, _: str) -> None:
            pass

        def load(self) -> list[Document]:
            return [Document(page_content="paper text", metadata={"page": 2})]

    monkeypatch.setattr("src.ingestion.PyPDFLoader", FakeLoader)
    result = load_research_papers(tmp_path)

    metadata = result.documents[0].metadata
    assert metadata["filename"] == "paper.pdf"
    assert metadata["page_number"] == 3
    assert metadata["paper_id"] == "paper"
    assert metadata["source_path"] == str(pdf_path.resolve())


def test_load_research_papers_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="No PDFs"):
        load_research_papers(tmp_path)

