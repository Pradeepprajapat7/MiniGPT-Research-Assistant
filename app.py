"""
MiniGPT Research Assistant - Streamlit Web UI (Day 6)
Interactive RAG interface with real-time chat, source citations, and summarization.
"""

from pathlib import Path
import streamlit as st
import torch

from src.rag.rag_pipeline import RAGPipeline

# Page Configuration
st.set_page_config(
    page_title="MiniGPT Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading MiniGPT model & FAISS Vector Store...")
def load_pipeline():
    """Load the RAG pipeline once and cache it in memory."""
    return RAGPipeline()


# Load the cached pipeline
pipeline = load_pipeline()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ RAG Dashboard")
    st.caption("MiniGPT Research Assistant")

    # System Status
    st.subheader("System Status")
    device_name = str(pipeline.device).upper()
    total_chunks = pipeline.vector_store.index.ntotal if pipeline.vector_store.index else 0
    st.success(f"Compute Device: **{device_name}**")
    st.info(f"Indexed Chunks: **{total_chunks}**")

    # Indexed Documents List
    st.subheader("📁 Indexed Documents")
    unique_docs = sorted(list({
        Path(m.get("source", "Unknown")).name
        for m in pipeline.vector_store.metadata
    }))
    if unique_docs:
        for doc in unique_docs:
            st.markdown(f"- 📄 `{doc}`")
    else:
        st.write("No documents found.")

    st.divider()

    # Search Settings
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=4,
        value=2,
        help="Number of most similar passages to supply as context to the model.",
    )

    st.divider()

    # Document Summarizer Tool
    st.subheader("📝 Quick Summarizer")
    summary_target = st.selectbox(
        "Select Document to Summarize",
        options=["All Documents"] + unique_docs,
    )

    if st.button("Generate Summary", use_container_width=True):
        doc_arg = None if summary_target == "All Documents" else summary_target
        with st.spinner(f"Generating summary for {summary_target}..."):
            summary_res = pipeline.summarize_document(
                document_name=doc_arg,
                max_chunks=3,
                max_new_tokens=150,
            )
            st.session_state["latest_summary"] = summary_res

    # Reset Chat History
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if "latest_summary" in st.session_state:
            del st.session_state["latest_summary"]
        st.rerun()

# ----------------- MAIN CONTENT AREA -----------------
st.title("🤖 MiniGPT Research Assistant")
st.markdown(
    "Ask questions about your research papers. Every answer is grounded in retrieved PDF passages with **page citations**."
)

# Render Summary banner if generated
if "latest_summary" in st.session_state:
    summary_data = st.session_state["latest_summary"]
    with st.chat_message("assistant", avatar="📝"):
        st.markdown(f"### 📋 Executive Summary: *{summary_data['document']}*")
        st.write(summary_data["summary"])
        with st.expander(f"Sources used ({summary_data['chunks_used']} chunks)"):
            for s in summary_data["sources"]:
                st.write(f"- **{s['source']}** (Page {s['page']})")

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your research assistant. Ask me anything about your loaded documents!",
            "sources": [],
        }
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display citations if any
        if message.get("sources"):
            with st.expander(f"📚 Citations & Sources ({len(message['sources'])} matches)"):
                for idx, src in enumerate(message["sources"], start=1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**[{idx}] {src['source']}** — Page `{src['page']}`")
                    with col2:
                        st.caption(f"Score: `{src['similarity_score']}`")
                    st.caption(f"> \"{src['preview']}\"")
                    st.divider()

# User Input Box
if prompt := st.chat_input("Ask a question about the papers..."):
    # 1. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response via RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            result = pipeline.answer_question(prompt, top_k=top_k)
            answer_text = result["answer"]
            sources = result["sources"]

            st.markdown(answer_text)

            # Render expandable citations
            if sources:
                with st.expander(f"📚 Citations & Sources ({len(sources)} matches)"):
                    for idx, src in enumerate(sources, start=1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**[{idx}] {src['source']}** — Page `{src['page']}`")
                        with col2:
                            st.caption(f"Score: `{src['similarity_score']}`")
                        st.caption(f"> \"{src['preview']}\"")
                        st.divider()

    # 3. Save assistant message with sources to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
    })