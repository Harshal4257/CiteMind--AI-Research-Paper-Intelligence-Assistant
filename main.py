"""Command-line interface for the AI/ML Research Paper RAG Assistant."""

from __future__ import annotations

import argparse

from src.config import Settings
from src.logging_config import configure_logging
from src.rag_pipeline import RAGResult, ResearchPaperRAG


def print_result(result: RAGResult, debug: bool) -> None:
    """Render a structured RAG result for a terminal user."""
    print(f"\nAnswer:\n{result.answer}")
    print("\nSources:")
    print("\n".join(f"- {source}" for source in result.sources) or "- No sources.")
    print("\nLatency (seconds):")
    for stage in ("retrieval", "reranking", "generation", "total"):
        print(f"{stage.title()}: {result.latency_seconds.get(stage, 0.0):.3f}")
    if result.error:
        print(f"\nStatus: {result.error}")
    if debug and result.retrieved_chunks:
        print("\nRetrieved chunks:")
        for chunk in result.retrieved_chunks:
            metadata = chunk.document.metadata
            print(
                f"\n{metadata['chunk_id']} | relevance={chunk.relevance_score:.3f}"
            )
            print(chunk.document.page_content[:700])


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions across indexed AI/ML papers.")
    parser.add_argument("--debug", action="store_true", help="Show retrieved chunks and scores.")
    args = parser.parse_args()

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    try:
        rag = ResearchPaperRAG.from_settings(settings)
    except Exception as exc:
        print(f"Startup failed: {exc}")
        return 1

    print("AI/ML Research Paper RAG Assistant")
    print("Type a question or 'exit'.")
    while True:
        try:
            question = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if question.strip().lower() == "exit":
            print("Goodbye.")
            return 0
        print_result(rag.ask(question), debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main())

