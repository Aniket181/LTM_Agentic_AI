"""
dashboard/views/v5_maintenance_insights.py
Renders the maintenance insights and knowledge retrieval view.
"""
from typing import Any
import streamlit as st

from dashboard.services import retrieve_knowledge, get_retriever_chunk_count

SUGGESTED_QUERIES = [
    "What are common bearing outer race fault symptoms?",
    "How should high RMS vibration be interpreted?",
    "What maintenance actions are recommended for degrading bearings?",
    "What does high kurtosis indicate in bearing vibration?",
    "How is the health score interpreted?"
]

def render(result: dict[str, Any], test_id: str) -> None:
    st.markdown("## 📚 Maintenance Insights")
    st.caption("Investigate current conditions using the AI-powered Maintenance Knowledge Base.")
    
    health = result.get("health", {})
    fault = result.get("fault", {})
    anomaly = result.get("anomaly", {})
    
    status = health.get("final_health_status", "Unknown")
    fault_pred = fault.get("predicted_fault", "Unknown")
    anomaly_label = anomaly.get("anomaly_label", "Unknown")
    
    # Context Banner
    st.markdown(f"""
    <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.2rem; background-color: #0f172a; margin-bottom: 2rem;">
        <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 0.5rem;">Current Condition Context</div>
        <div style="display: flex; gap: 2rem;">
            <div><span style="color: #64748b;">Health:</span> <strong style="color: {'#ef4444' if status == 'Critical' else '#f59e0b' if status == 'Degrading' else '#10b981'};">{status}</strong></div>
            <div><span style="color: #64748b;">Fault:</span> <strong style="color: {'#ef4444' if fault_pred == 'Faulty' else '#10b981'};">{fault_pred}</strong></div>
            <div><span style="color: #64748b;">Anomaly:</span> <strong style="color: {'#ef4444' if anomaly_label == 'Anomaly' else '#10b981'};">{anomaly_label}</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Search Maintenance Knowledge")
    
    chunk_count = get_retriever_chunk_count()
    if chunk_count > 0:
        st.caption(f"Knowledge base active ({chunk_count} document chunks indexed).")
    else:
        st.warning("Knowledge base not loaded. Run `python scripts/build_knowledge_base.py` to build it.")

    col_input, col_btn = st.columns([4, 1])
    
    with col_input:
        query_input = st.text_input(
            "Search query",
            placeholder="e.g. outer race fault symptoms...",
            label_visibility="collapsed",
            key="rag_query"
        )
        
    with col_btn:
        search_clicked = st.button("Search", type="primary")
        
    def set_query(q_str):
        st.session_state["rag_query"] = q_str

    st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem; margin-bottom: 0.2rem;'>Suggested questions:</div>", unsafe_allow_html=True)
    
    # Quick pick suggestions
    for q in SUGGESTED_QUERIES:
        st.button(q, key=f"btn_q_{q}", on_click=set_query, args=(q,))

    st.markdown("---")
    
    # Retrieve and show results
    query_to_run = st.session_state.get("rag_query", "").strip()
    
    if search_clicked or (query_to_run and not query_input == ""):
        if not query_to_run:
            query_to_run = query_input.strip()
            
        if not query_to_run:
            st.info("Please enter a search query.")
            return

        with st.spinner("Retrieving knowledge..."):
            retrieval_res = retrieve_knowledge(query_to_run)

        if "error" in retrieval_res:
            st.error(f"**Retrieval Error:** {retrieval_res['error']}")
            st.info("Is Ollama running with the nomic-embed-text model?")
            return

        chunks = retrieval_res.get("results", [])
        st.markdown(f"**Top {len(chunks)} Relevant Results for:** *\"{query_to_run}\"*")

        if not chunks:
            st.warning("No relevant information found in the knowledge base.")
            return

        for idx, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'Unknown')
            section = chunk.get('section', 'Unknown')
            distance = chunk.get('distance', 1.0)
            
            # Simple heuristic mapping for distance -> relevance % (lower distance = higher relevance)
            relevance = max(0, min(100, int((1.0 - (distance / 2.0)) * 100)))

            with st.expander(f"{idx}. {source} - {section} (Relevance: {relevance}%)", expanded=(idx==1)):
                st.markdown(f"**Document:** `{source}`")
                st.markdown(f"**Section:** `{section}`")
                st.markdown("**Content:**")
                st.info(chunk.get("content", ""))
