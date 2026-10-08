from langchain_core.documents import Document

from src.retriever import retrieve


class FakeVectorStore:
    def similarity_search_with_relevance_scores(self, question: str, k: int):
        assert question == "What is self-attention?"
        assert k == 2
        return [
            (Document(page_content="relevant", metadata={"filename": "paper.pdf", "page_number": 2, "chunk_id": "p::1"}), 0.8),
            (Document(page_content="weak", metadata={"filename": "paper.pdf", "page_number": 3, "chunk_id": "p::2"}), 0.1),
        ]


def test_retrieval_returns_scored_metadata_aware_chunks(settings) -> None:
    results = retrieve(FakeVectorStore(), "What is self-attention?", settings)

    assert len(results) == 1
    assert results[0].relevance_score == 0.8
    assert results[0].document.metadata["relevance_score"] == 0.8
    assert results[0].document.metadata["chunk_id"] == "p::1"


