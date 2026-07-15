"""
rag_pipeline.py

Wires document ingestion -> embedding -> FAISS retrieval -> LLM generation
into a single RAGPipeline class. Generation uses Groq's free-tier API
(OpenAI-compatible chat completions, no cost, no credit card required).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .document_loader import Chunk, load_and_chunk
from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question \
using ONLY the information in the provided context excerpts. \
If the context does not contain enough information to answer, say so plainly \
instead of guessing. When you use a piece of context, mention which source it \
came from (by filename)."""


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        api_key: Optional[str] = None,
    ):
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.vector_store = FAISSVectorStore(dimension=self.embedding_model.dimension)
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        from groq import Groq

        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    # ----------------------------------------------------------------
    def ingest(self, paths: List[str | Path]) -> int:
        """Load, chunk, embed, and index one or more documents. Returns #chunks added."""
        all_chunks: List[Chunk] = []
        for path in paths:
            all_chunks.extend(
                load_and_chunk(path, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            )

        if not all_chunks:
            return 0

        embeddings = self.embedding_model.embed([c.text for c in all_chunks])
        self.vector_store.add(embeddings, all_chunks)
        return len(all_chunks)

    # ----------------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None):
        """Return [(score, Chunk), ...] for the top-k most relevant chunks."""
        query_embedding = self.embedding_model.embed_query(query)
        return self.vector_store.search(query_embedding, k=k or self.top_k)

    # ----------------------------------------------------------------
    def answer(self, query: str, k: Optional[int] = None) -> dict:
        """Retrieve relevant chunks and generate an answer grounded in them."""
        results = self.retrieve(query, k=k)

        if not results:
            return {
                "answer": "I don't have any indexed documents to search yet.",
                "sources": [],
            }

        context_blocks = []
        for i, (score, chunk) in enumerate(results, start=1):
            context_blocks.append(
                f"[{i}] (source: {chunk.source}, relevance: {score:.2f})\n{chunk.text}"
            )
        context = "\n\n".join(context_blocks)

        user_message = f"Context excerpts:\n\n{context}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=self.llm_model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer_text = response.choices[0].message.content

        return {
            "answer": answer_text,
            "sources": [
                {"source": chunk.source, "score": score, "text": chunk.text}
                for score, chunk in results
            ],
        }

    # ----------------------------------------------------------------
    def save_index(self, dir_path: str | Path) -> None:
        self.vector_store.save(dir_path)

    def load_index(self, dir_path: str | Path) -> None:
        self.vector_store = FAISSVectorStore.load(dir_path)
