"""Central configuration for the notebook-equivalent RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_optional_float(value: str | None, default: float | None) -> float | None:
    if value is None or not value.strip():
        return default
    if value.strip().lower() == "none":
        return None
    return float(value)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings, loaded once from environment variables."""

    project_root: Path
    papers_dir: Path
    vector_store_dir: Path
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    initial_retrieval_k: int
    min_relevance_score: float | None
    llm_provider: str
    llm_model: str
    temperature: float
    max_new_tokens: int
    reranking_enabled: bool
    reranker_model_name: str
    max_query_chars: int
    log_level: str
    local_device: int = -1

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        load_dotenv(root / ".env")

        papers_dir = Path(os.getenv("PAPERS_DIR", root / "data" / "papers"))
        vector_store_dir = Path(
            os.getenv("VECTOR_STORE_DIR", root / "storage" / "vectorstore")
        )
        return cls(
            project_root=root,
            papers_dir=papers_dir,
            vector_store_dir=vector_store_dir,
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
            initial_retrieval_k=int(os.getenv("INITIAL_RETRIEVAL_K", "10")),
            min_relevance_score=_as_optional_float(
                os.getenv("MIN_RELEVANCE_SCORE"), 0.15
            ),
            llm_provider=os.getenv("LLM_PROVIDER", "local_huggingface"),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_new_tokens=int(
                os.getenv("MAX_OUTPUT_TOKENS", os.getenv("MAX_NEW_TOKENS", "256"))
            ),
            reranking_enabled=_as_bool(os.getenv("RERANKING_ENABLED"), False),
            reranker_model_name=os.getenv(
                "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            max_query_chars=int(os.getenv("MAX_QUERY_CHARS", "2000")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            local_device=int(os.getenv("LOCAL_DEVICE", "-1")),
        )

