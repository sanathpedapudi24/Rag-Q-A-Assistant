"""
vector_store.py

A thin FAISS wrapper that stores chunk embeddings alongside their text and
metadata, and supports saving/loading the index to disk.

Uses an IndexFlatIP (inner product) index. Because EmbeddingModel L2-normalizes
its output, inner product is equivalent to cosine similarity.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .document_loader import Chunk


class FAISSVectorStore:
    def __init__(self, dimension: int):
        import faiss

        self.dimension = dimension
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []  # parallel array: index i <-> self.index vector i

    # ----------------------------------------------------------------
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must be the same length")
        if len(embeddings) == 0:
            return
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    # ----------------------------------------------------------------
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Chunk]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, k)

        results: List[Tuple[float, Chunk]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self.chunks[idx]))
        return results

    # ----------------------------------------------------------------
    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(dir_path / "index.faiss"))
        with open(dir_path / "chunks.pkl", "wb") as f:
            pickle.dump([asdict(c) for c in self.chunks], f)
        with open(dir_path / "config.json", "w") as f:
            json.dump({"dimension": self.dimension}, f)

    @classmethod
    def load(cls, dir_path: str | Path) -> "FAISSVectorStore":
        import faiss

        dir_path = Path(dir_path)
        with open(dir_path / "config.json") as f:
            config = json.load(f)

        store = cls(dimension=config["dimension"])
        store.index = faiss.read_index(str(dir_path / "index.faiss"))
        with open(dir_path / "chunks.pkl", "rb") as f:
            raw_chunks = pickle.load(f)
        store.chunks = [Chunk(**c) for c in raw_chunks]
        return store

    def __len__(self) -> int:
        return self.index.ntotal
