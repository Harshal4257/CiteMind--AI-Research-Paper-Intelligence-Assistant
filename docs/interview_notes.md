# Interview notes: AI/ML Research Paper RAG Assistant

These answers describe this repository's implemented design, not generic RAG theory.

## 1. What did you build?

A local multi-document assistant for AI/ML papers. A user asks in Streamlit or the terminal, the system retrieves FAISS evidence from PDF chunks, generates a constrained answer with local Qwen, and shows paper/page sources plus timing.

## 2. What user problem does it solve?

It shortens the time needed to find and compare details across long research papers while keeping the supporting page visible.

## 3. Why RAG instead of fine-tuning?

The paper collection changes more often than a model should be retrained. Re-indexing PDFs is cheaper, preserves source attribution, and lets the model use explicit evidence at answer time.

## 4. What happens during ingestion?

`PyPDFLoader` loads each PDF page, lightweight cleanup removes extraction noise, and metadata records filename, path, paper ID, page number, and deterministic chunk ID.

## 5. How did you split documents?

`RecursiveCharacterTextSplitter` uses chunks of 1,000 characters with 150-character overlap. The overlap protects ideas that cross a boundary; the chunk size keeps prompt context manageable.

## 6. Which embedding model did you choose and why?

`sentence-transformers/all-MiniLM-L6-v2`. It is compact enough for local experimentation and is a practical semantic-search baseline for a five-paper corpus.

## 7. Why FAISS?

FAISS provides fast local vector similarity search and easy persistence. It keeps the prototype free of a hosted vector database and runs from the generated `storage/vectorstore` files.

## 8. How do you retain source provenance?

Every chunk retains the originating filename, page number, and chunk ID. The UI deduplicates paper/page pairs for citations and exposes the raw retrieved chunks in an expander.

## 9. How does retrieval work?

The query is embedded with the same MiniLM model, FAISS returns candidates, scores are filtered by `MIN_RELEVANCE_SCORE`, and the best chunks are sent to the prompt. `RETRIEVAL_TOP_K` defaults to 4 after an initial candidate count of 10.

## 10. Is reranking included?

Yes, as an optional `cross-encoder/ms-marco-MiniLM-L-6-v2` stage. It is disabled by default to keep startup and CPU usage small; when unavailable the pipeline falls back to vector retrieval.

## 11. Which generation model is used?

The current configuration is `Qwen/Qwen2.5-1.5B-Instruct` through a local Hugging Face pipeline in FP16 on the NVIDIA GTX 1650 (`LOCAL_DEVICE=0`). Hosted Hugging Face endpoint support is retained as an option, but it needs valid provider credits.

## 12. How do you reduce hallucinations?

The prompt tells the model to use only the supplied evidence, a relevance threshold can reject weak retrieval, context is limited to retrieved chunks, page citations are shown, and no relevant evidence returns an explicit safe fallback.

## 13. How do you handle prompt injection in papers?

Retrieved paper text is treated as untrusted evidence, not as application instructions. The system prompt establishes that boundary; this is risk reduction, not a complete security guarantee.

## 14. What validation and error handling exist?

Queries are normalized and checked for empty and oversized input. The app checks for PDFs and a complete FAISS index before inference. PDF, index, model, and provider failures are caught, logged, and mapped to safe user-facing messages.

## 15. Why did you use Streamlit instead of a separate frontend/API?

It gives a polished demonstrable interface quickly and can call the existing `ResearchPaperRAG` backend without duplicating contracts. For a single-user prototype, FastAPI would add operational complexity without user value.

## 16. How do you manage state and model loading in the UI?

`st.session_state` retains display history. `st.cache_resource` creates the embedding model, FAISS store, and generator once per Streamlit process, avoiding a reload for every question. Prior chat messages are intentionally not injected into retrieval.

## 17. What observability did you add?

Backend results carry retrieval, reranking, generation, and total seconds. The service logs a short request ID, character count, retrieved-chunk count, total time, and error state without logging question content. Debug mode displays per-stage metrics.

## 18. What did actual validation show?

Five papers produced 76 pages and 355 chunks. The test suite has 18 passing tests. A verified local GPU response used 0.570 s retrieval, 16.855 s generation, and 17.424 s total, with 3.25 GB peak GPU memory. Generation remains the bottleneck, but GPU inference materially improved it over the prior CPU run.

## 19. How is it packaged and deployed?

The Docker image installs dependencies, exposes Streamlit on 8501, and checks `/_stcore/health`. `render.yaml` uses a paid `2c-4g` web service and a `/app/storage` persistent disk for papers, FAISS index, and model cache. No deployment is claimed without account credentials.

## 20. What would you improve for production?

Use hybrid sparse+dense retrieval, metadata filters, curated evaluation data, a better reranker, GPU or managed inference, asynchronous ingestion, authentication, feedback/audit storage, rate limiting, and monitoring/alerts. For scale, move documents and vectors to managed storage and expose the backend through an API.
