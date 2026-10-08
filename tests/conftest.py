from pathlib import Path

import pytest

from src.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        papers_dir=tmp_path / "data" / "papers",
        vector_store_dir=tmp_path / "storage" / "vectorstore",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=20,
        retrieval_top_k=2,
        initial_retrieval_k=4,
        min_relevance_score=0.15,
        llm_provider="huggingface",
        llm_model="Qwen/Qwen2.5-3B-Instruct",
        temperature=0.0,
        max_new_tokens=32,
        reranking_enabled=False,
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_query_chars=100,
        log_level="INFO",
    )

