"""
embeddings.py

Thin wrapper around sentence-transformers so the rest of the codebase
doesn't need to know which embedding model/library is in use.
"""

from __future__ import annotations

from typing import List

import numpy as np


class EmbeddingModel:
    """Wraps a sentence-transformers model and exposes a simple .embed() API."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of strings -> (n_texts, dimension) float32 array, L2-normalized."""
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so inner-product search == cosine similarity
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string -> (dimension,) float32 array."""
        return self.embed([text])[0]
