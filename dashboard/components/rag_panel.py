"""
dashboard/components/rag_panel.py — Phase 10

Renders the RAG Knowledge Retrieval panel:
  - Text input for natural-language query
  - Quick-pick suggested queries
  - Retrieve button (fires only on button press, not on input change)
  - Results displayed as expandable sections
  - Clear error messages if Ollama is unavailable

This panel performs RETRIEVAL ONLY. No LLM generation occurs.
All knowledge comes from the existing ChromaDB collection built in Phase 9.
"""

from __future__ import annotations

import streamlit as st

from dashboard.services import retrieve_knowledge, get_retriever_chunk_count


SUGGESTED_QUERIES = [
    "What are common bearing outer race fault symptoms?",
    "How to interpret high RMS vibration in bearings?",
    "What maintenance actions are recommended for degrading bearings?",
    "How is the health score interpreted for equipment condition?",
    "What does high kurtosis indicate in bearing vibration analysis?",
]


def render_rag_panel() -> None:
    """Render the RAG knowledge retrieval panel."""

    st.markdown("#### 🔍 Knowledge Base Query")

    # ── Collection status ─────────────────────────────────────────────────────
    chunk_count = get_retriever_chunk_count()
    if chunk_count > 0:
        st.caption(f"📚 Knowledge base: **{chunk_count} chunks** indexed (nomic-embed-text via Ollama)")
    else:
        st.warning(
            "Knowledge base not loaded. "
            "Run `python scripts/build_knowledge_base.py` to build it."
        )

    # ── Query input ───────────────────────────────────────────────────────────
    col_input, col_suggest = st.columns([3, 2])

    with col_input:
        query = st.text_input(
            "Enter your query",
            placeholder="e.g. outer race fault symptoms",
            label_visibility="collapsed",
            key="rag_query_input",
        )

    with col_suggest:
        selected_suggestion = st.selectbox(
            "Or pick a suggestion",
            options=[""] + SUGGESTED_QUERIES,
            label_visibility="collapsed",
            key="rag_suggestion",
        )

    # Use suggestion if selected and query is empty
    active_query = query.strip() if query.strip() else selected_suggestion.strip()

    col_btn, col_k = st.columns([2, 1])
    with col_btn:
        search_clicked = st.button("🔍 Retrieve Knowledge", use_container_width=True)
    with col_k:
        top_k = st.number_input("Results (top_k)", min_value=1, max_value=10, value=5)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    if search_clicked:
        if not active_query:
            st.warning("Please enter a query or select a suggestion.")
            return

        with st.spinner(f"Querying knowledge base for: '{active_query}'…"):
            result = retrieve_knowledge(active_query, top_k=int(top_k))

        if "error" in result:
            st.error(f"**Retrieval failed:** {result['error']}")
            if result.get("details"):
                with st.expander("Details / Fix"):
                    for detail in result["details"]:
                        st.code(detail)
            return

        results = result.get("results", [])
        if not results:
            st.info("No results found. Try a different query.")
            return

        st.success(f"Found **{len(results)}** relevant passages")
        st.markdown("---")

        for i, r in enumerate(results, 1):
            source  = r.get("source",      "unknown")
            section = r.get("section",     "")
            dist    = r.get("distance",    0.0)
            title   = r.get("title",       "")
            content = r.get("content",     "")
            relevance = max(0.0, 1.0 - dist / 2.0) * 100  # cosine dist → % relevance

            header = (
                f"**{i}.** `{source}` — {section}  "
                f"| Relevance: **{relevance:.0f}%**"
            )
            with st.expander(header, expanded=(i == 1)):
                if title:
                    st.markdown(f"*Document:* {title}")
                st.markdown(f"*Section:* {section}")
                st.markdown(f"*Source:* `{source}` · *Distance:* `{dist:.4f}`")
                st.markdown("---")
                st.markdown(content)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.caption(
        "ℹ️ **Source:** knowledge_base/ Markdown documents only. "
        "No LLM generation. No fabricated content. "
        "Retrieval is vector-similarity search using nomic-embed-text (Ollama)."
    )
