"""Main orchestration for grounded research-paper question answering."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

from src.config import Settings
from src.embeddings import create_embeddings
from src.guardrails import validate_query
from src.prompts import (
    RAG_PROMPT,
    SAFE_FALLBACK,
    SYSTEM_INSTRUCTIONS,
    answer_depth_instruction,
)
from src.reranker import CrossEncoderReranker
from src.retriever import RetrievedChunk, retrieve
from src.utils import QueryValidationError
from src.vectorstore import load_vector_store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RAGResult:
    """Structured response returned for every request, successful or otherwise."""

    question: str
    answer: str = SAFE_FALLBACK
    sources: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_seconds: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def build_llm(settings: Settings) -> Any:
    """Create a local free model by default, or an optional hosted HF endpoint."""
    provider = settings.llm_provider.lower()
    if provider == "local_huggingface":
        try:
            device = settings.local_device
            model_kwargs: dict[str, Any] = {}
            if device >= 0:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "LOCAL_DEVICE is set to a GPU, but CUDA-enabled PyTorch cannot access one."
                    )
                model_kwargs["dtype"] = torch.float16
            return HuggingFacePipeline.from_model_id(
                model_id=settings.llm_model,
                task="text-generation",
                device=device,
                model_kwargs=model_kwargs,
                pipeline_kwargs={
                    "max_new_tokens": settings.max_new_tokens,
                    "do_sample": False,
                    "return_full_text": False,
                    "repetition_penalty": 1.08,
                    "no_repeat_ngram_size": 3,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize local Hugging Face model '{settings.llm_model}'."
            ) from exc

    if provider in {"huggingface", "huggingface_endpoint"}:
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise EnvironmentError(
                "HUGGINGFACEHUB_API_TOKEN is missing. Add a Hugging Face fine-grained token "
                "with Inference Providers permission to .env."
            )
        endpoint_llm = HuggingFaceEndpoint(
            repo_id=settings.llm_model,
            huggingfacehub_api_token=token,
            task="text-generation",
            temperature=settings.temperature,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
        )
        return ChatHuggingFace(llm=endpoint_llm)

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use 'local_huggingface' or 'huggingface_endpoint'."
    )


def invoke_llm(
    llm: Any, question: str, context: str, answer_depth: str = "Standard"
) -> Any:
    """Invoke either a local text-generation model or hosted chat endpoint."""
    answer_style = answer_depth_instruction(answer_depth)
    if isinstance(llm, ChatHuggingFace):
        return llm.invoke(
            RAG_PROMPT.format_messages(
                question=question, context=context, answer_style=answer_style
            )
        )
    if isinstance(llm, HuggingFacePipeline):
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Response style:\n{answer_style}\n\nQuestion:\n{question}"
                    f"\n\nRetrieved research-paper context:\n{context}"
                ),
            },
        ]
        tokenizer = getattr(llm.pipeline, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return llm.invoke(prompt_text)
    prompt_text = RAG_PROMPT.format_prompt(
        question=question, context=context, answer_style=answer_style
    ).to_string()
    return llm.invoke(prompt_text)

def format_context(chunks: list[RetrievedChunk]) -> str:
    """Label evidence passages for traceable, grounded model input."""
    sections: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        metadata = chunk.document.metadata
        sections.append(
            f"[Evidence {number} | {metadata['filename']} | page {metadata['page_number']} "
            f"| {metadata['chunk_id']}]\n{chunk.document.page_content}"
        )
    return "\n\n---\n\n".join(sections)


def unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Create stable, deduplicated file/page citations from retrieved metadata."""
    return list(
        dict.fromkeys(
            f"{chunk.document.metadata['filename']} \N{EM DASH} page {chunk.document.metadata['page_number']}"
            for chunk in chunks
        )
    )


class ResearchPaperRAG:
    """Small orchestration class preserving the notebook's RAG request flow."""

    def __init__(
        self,
        settings: Settings,
        vector_store: Any,
        llm: Any,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ResearchPaperRAG":
        """Load the persisted index and the configured model once for an app session."""
        settings = settings or Settings.from_env()
        embeddings = create_embeddings(settings)
        vector_store = load_vector_store(settings.vector_store_dir, embeddings)
        llm = build_llm(settings)

        reranker = None
        if settings.reranking_enabled:
            try:
                reranker = CrossEncoderReranker.create(settings.reranker_model_name)
                logger.info("Optional reranker enabled: %s", settings.reranker_model_name)
            except Exception as exc:
                logger.warning("Reranker failed to load; falling back to vector retrieval: %s", exc)
        return cls(settings=settings, vector_store=vector_store, llm=llm, reranker=reranker)

    def ask(self, question: object, answer_depth: str = "Standard") -> RAGResult:
        """Answer a question from retrieved paper evidence and report stage latency."""
        total_start = time.perf_counter()
        normalized_question = question.strip() if isinstance(question, str) else ""
        result = RAGResult(question=normalized_question)
        try:
            normalized_question = validate_query(question, self.settings.max_query_chars)
            result.question = normalized_question
        except QueryValidationError as exc:
            result.error = str(exc)
            result.latency_seconds = _latencies(total_start)
            return result

        try:
            retrieval_start = time.perf_counter()
            chunks = retrieve(self.vector_store, normalized_question, self.settings)
            result.latency_seconds["retrieval"] = _elapsed(retrieval_start)

            rerank_start = time.perf_counter()
            if self.reranker is not None:
                chunks = self.reranker.rerank(
                    normalized_question, chunks, self.settings.retrieval_top_k
                )
                result.latency_seconds["reranking"] = _elapsed(rerank_start)
            else:
                result.latency_seconds["reranking"] = 0.0
            result.retrieved_chunks = chunks

            if not chunks:
                result.error = "No sufficiently relevant evidence was retrieved."
                result.latency_seconds["generation"] = 0.0
                result.latency_seconds["total"] = _elapsed(total_start)
                return result

            generation_start = time.perf_counter()
            response = invoke_llm(
                self.llm, normalized_question, format_context(chunks), answer_depth
            )
            content = getattr(response, "content", response)
            result.answer = content.strip() if isinstance(content, str) else str(content).strip()
            result.sources = unique_sources(chunks)
            result.latency_seconds["generation"] = _elapsed(generation_start)
        except Exception as exc:
            logger.exception("RAG request failed")
            result.error = f"RAG request failed: {type(exc).__name__}: {exc}"
            result.latency_seconds.setdefault("generation", 0.0)

        result.latency_seconds.setdefault("retrieval", 0.0)
        result.latency_seconds.setdefault("reranking", 0.0)
        result.latency_seconds["total"] = _elapsed(total_start)
        logger.info("RAG request completed in %.3fs", result.latency_seconds["total"])
        return result


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def _latencies(total_start: float) -> dict[str, float]:
    return {"retrieval": 0.0, "reranking": 0.0, "generation": 0.0, "total": _elapsed(total_start)}




