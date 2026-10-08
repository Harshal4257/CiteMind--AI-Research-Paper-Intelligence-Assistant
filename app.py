"""Streamlit portfolio interface for the AI/ML Research Paper RAG Assistant."""

from __future__ import annotations

import logging

import streamlit as st

from src.app_service import RAGApplicationService, get_health_status
from src.config import Settings
from src.logging_config import configure_logging
from src.rag_pipeline import RAGResult
from src.ui_helpers import (
    clean_paper_name,
    evidence_rows,
    evidence_strength,
    latency_rows,
    source_rows,
    user_facing_error,
)


st.set_page_config(
    page_title="AI/ML Research Paper Assistant",
    page_icon="\N{BOOKS}",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_product_style() -> None:
    """Keep the UI refined with a small, dependency-free Streamlit theme."""
    st.markdown(
        """
        <style>
          .stApp {background: #ffffff; color: #172033;}
          .block-container {max-width: 1210px; padding-top: 2rem; padding-bottom: 4rem;}
          [data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e7eaf0;}
          [data-testid="stSidebar"] * {color: #172033;}
          .product-hero {background: #ffffff; border: 1px solid #e4e7ed; border-radius: 16px; padding: 1.8rem 2rem; margin: .4rem 0 1.15rem; box-shadow: 0 8px 24px rgba(15,23,42,.06);}
          .eyebrow {color: #6b7280; font-size: .73rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase;}
          .product-hero h1 {color: #172033; font-size: 2.45rem; letter-spacing: -.05em; margin: .35rem 0 .55rem;}
          .product-hero p {color: #566074; font-size: 1.02rem; line-height: 1.6; margin: 0; max-width: 700px;}
          .chip-row {display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.25rem;}
          .chip {background: #f7f8fa; border: 1px solid #e2e5ea; color:#4b5565; border-radius:999px; font-size:.78rem; padding:.36rem .72rem;}
          .empty-state {border: 1px dashed #d6dbe4; border-radius: 16px; padding: 2.35rem 1.4rem; text-align: center; background: #fbfcfd; margin: .4rem 0 1rem;}
          .empty-state h3 {color: #172033; margin: 0 0 .4rem;}
          .empty-state p {color: #687386; margin: 0;}
          .subtle {color: #687386; font-size: .87rem;}
          .status-ready {color: #15803d; font-weight: 700;}
          .status-warn {color: #b45309; font-weight: 700;}
          div[data-testid="stChatMessage"] {background: #ffffff; border: 1px solid #e5e8ee; border-radius: 14px; padding: .7rem 1rem; margin-bottom: .95rem;}
          div[data-testid="stChatInput"] {padding-top: .75rem;}
          [data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e5e8ee; border-radius: 12px; padding: .45rem .7rem;}
          [data-testid="stExpander"] {border: 1px solid #e5e8ee; border-radius: 12px; overflow: hidden;}
          [data-testid="stDownloadButton"] button {border-radius: 9px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading the research assistant...")
def get_service() -> RAGApplicationService:
    """Load models and the persisted index once per Streamlit server process."""
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    return RAGApplicationService.create(settings)


def render_result(result: RAGResult, debug: bool) -> None:
    """Render a scannable answer with sources and transparent retrieval details."""
    message = user_facing_error(result)
    if message:
        if result.error == "No sufficiently relevant evidence was retrieved.":
            with st.container(border=True):
                st.markdown("#### Insufficient evidence in the indexed papers")
                st.write(message)
                st.caption(
                    "Try rephrasing the question, add a relevant paper, or ask about another indexed topic."
                )
            return
        st.error(message)
        if not result.retrieved_chunks:
            return

    st.markdown(result.answer)
    strength, strength_note = evidence_strength(result)
    st.caption(f"Evidence strength: **{strength}** - {strength_note}")

    export_column, _ = st.columns([1, 5])
    with export_column:
        st.download_button(
            "Export answer",
            data=result.answer,
            file_name="research-paper-answer.md",
            mime="text/markdown",
            use_container_width=True,
        )

    sources = source_rows(result)
    if sources:
        st.markdown("#### Sources used")
        for index, source in enumerate(sources, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}. {clean_paper_name(str(source['paper']))}**")
                st.caption(
                    f"Page {source['page']}  |  Retrieval relevance {source['relevance_score']:.3f}"
                )

    with st.expander("Why this answer?", expanded=False):
        st.caption(
            "This panel shows the retrieved passages used as evidence. Evidence strength is derived "
            "from retrieval scores, not from LLM self-confidence."
        )
        for index, evidence in enumerate(evidence_rows(result), start=1):
            with st.container(border=True):
                st.markdown(f"**Evidence {index}: {evidence['paper']}**")
                st.caption(
                    f"Page {evidence['page']}  |  relevance {evidence['relevance_score']:.3f}  |  "
                    f"{evidence['chunk_id']}"
                )
                st.write(evidence["excerpt"])

    with st.expander("Performance details", expanded=False):
        metrics = latency_rows(result)
        columns = st.columns(4)
        for column, metric in zip(columns, metrics, strict=True):
            column.metric(metric["stage"], f"{metric['seconds']:.2f} s")

    if debug:
        with st.expander("Debug details", expanded=False):
            st.dataframe(latency_rows(result), hide_index=True, use_container_width=True)
        if result.error:
            st.caption(f"Backend status: {result.error}")


def main() -> None:
    apply_product_style()
    settings = Settings.from_env()
    health = get_health_status(settings)

    with st.sidebar:
        st.markdown("## Research Library")
        st.caption("A private workspace for exploring AI/ML papers.")
        st.divider()
        if health.ready:
            st.markdown("<span class='status-ready'>● Vector index ready</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-warn'>● Index or papers unavailable</span>", unsafe_allow_html=True)
        st.metric("Indexed papers", len(health.papers))
        st.caption(f"**Embeddings**  \n{settings.embedding_model_name}")
        st.caption(f"**LLM**  \n{health.llm_model}")
        st.caption(f"**Reranker**  \n{'Enabled' if settings.reranking_enabled else 'Disabled'}")
        answer_depth = st.radio(
            "Answer depth",
            options=("Concise", "Standard", "Detailed"),
            index=1,
            horizontal=True,
            help="Changes the model's response instructions; it never slices a generated answer.",
        )
        debug = st.toggle("Debug mode", value=False)
        if health.papers:
            with st.expander("Indexed papers", expanded=False):
                for paper in health.papers:
                    st.write(f"- {clean_paper_name(paper)}")
        st.divider()
        st.caption("Add PDFs to `data/papers/`, then rebuild the index from the terminal.")

    st.markdown(
        f"""
        <div class="product-hero">
          <div class="eyebrow">AI/ML research workspace</div>
          <h1>Research Paper Assistant</h1>
          <p>Explore your paper library with source-backed answers, transparent retrieval, and a local GPU-powered model.</p>
          <div class="chip-row">
            <span class="chip">{len(health.papers)} indexed papers</span>
            <span class="chip">Page-level sources</span>
            <span class="chip">Local GPU inference</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtle'>Each question is independently retrieved. Chat history is never injected into the evidence context.</div>",
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_result(message["result"], debug)

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
              <h3>Your research workspace is ready</h3>
              <p>Ask a question below to search the indexed papers and inspect the supporting evidence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    question = st.chat_input("Ask about concepts, methods, datasets, or paper comparisons...")
    if question is None:
        return
    if not question.strip():
        st.warning("Please enter a question before submitting.")
        return
    if len(question) > settings.max_query_chars:
        st.warning(f"Question exceeds the {settings.max_query_chars}-character limit.")
        return
    if not health.ready:
        st.error("The vector index is unavailable. Run `python scripts/build_index.py` first.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.status("Preparing an evidence-grounded response...", expanded=True) as status:
                st.write("Searching the indexed papers and generating a complete answer.")
                result = get_service().ask(question, answer_depth=answer_depth)
                status.update(label="Grounded response ready", state="complete", expanded=False)
            render_result(result, debug)
        except Exception:
            logging.getLogger(__name__).exception("Streamlit request failed")
            st.error("The research assistant is unavailable right now. Please try again shortly.")
            return
    st.session_state.messages.append({"role": "assistant", "result": result})


if __name__ == "__main__":
    main()

