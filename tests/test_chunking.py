from langchain_core.documents import Document

from src.chunking import chunk_documents


def test_chunks_preserve_metadata_and_assign_ids(settings) -> None:
    document = Document(
        page_content="self attention " * 30,
        metadata={"paper_id": "transformer", "page_number": 6, "filename": "paper.pdf"},
    )
    chunks = chunk_documents([document], settings)

    assert len(chunks) > 1
    assert all(chunk.metadata["filename"] == "paper.pdf" for chunk in chunks)
    assert chunks[0].metadata["chunk_id"] == "transformer::p6::c00000"
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)

