"""
app.py

Streamlit interface for the RAG document Q&A assistant.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Session state / pipeline setup
# --------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


def get_pipeline() -> RAGPipeline:
    if st.session_state.pipeline is None:
        st.session_state.pipeline = RAGPipeline(
            embedding_model_name=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            chunk_size=int(os.environ.get("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 50)),
            top_k=int(os.environ.get("TOP_K", 5)),
        )
    return st.session_state.pipeline


# --------------------------------------------------------------------------
# Sidebar: upload + index
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")

    api_key = st.text_input(
        "Groq API key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Free, no credit card required",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    with st.expander("🔑 How to get a free Groq API key"):
        st.markdown("""
        **1.** Go to **[console.groq.com](https://console.groq.com)** and click **Sign Up**
        
        **2.** Sign in with **Google**, **GitHub**, or **email** (takes 10 seconds)
        
        **3.** Click **API Keys** in the left sidebar
        
        **4.** Click **Create API Key** → give it a name → click **Create**
        
        **5.** Copy the key (starts with `gsk_...`) and paste it above
        
        > Free tier — no credit card required.
        """)

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)

    if st.button("Build / update index", type="primary", disabled=not uploaded_files):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please provide a Groq API key first.")
        else:
            pipeline = get_pipeline()
            pipeline.top_k = top_k
            with st.spinner("Chunking and embedding documents..."):
                tmp_dir = Path(tempfile.mkdtemp())
                saved_paths = []
                for uploaded_file in uploaded_files:
                    dest = tmp_dir / uploaded_file.name
                    dest.write_bytes(uploaded_file.getbuffer())
                    saved_paths.append(dest)

                n_chunks = pipeline.ingest(saved_paths)
                st.session_state.indexed_files.extend(
                    [f.name for f in uploaded_files]
                )
            st.success(f"Indexed {n_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for name in st.session_state.indexed_files:
            st.caption(f"✓ {name}")

    if st.button("Clear index"):
        st.session_state.pipeline = None
        st.session_state.indexed_files = []
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------
# Main: chat interface
# --------------------------------------------------------------------------
st.title("RAG Document Q&A Assistant")
st.caption("Upload documents in the sidebar, then ask questions grounded in their content.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(f"**{s['source']}** · relevance {s['score']:.2f}")
                    st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

query = st.chat_input("Ask a question about your documents...")

if query:
    if st.session_state.pipeline is None or len(st.session_state.pipeline.vector_store) == 0:
        st.warning("Upload and index at least one document first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_pipeline().answer(query, k=top_k)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.markdown(f"**{s['source']}** · relevance {s['score']:.2f}")
                            st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )
