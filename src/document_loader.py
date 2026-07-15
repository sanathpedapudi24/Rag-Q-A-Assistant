"""
document_loader.py

Loads text out of PDF / DOCX / TXT files and splits it into overlapping
chunks suitable for embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    """A single chunk of text plus metadata about where it came from."""

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# File readers
# --------------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    import docx  # python-docx

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


_LOADERS = {
    ".txt": _read_txt,
    ".md": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def load_document(path: str | Path) -> str:
    """Extract raw text from a supported document type."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {list(_LOADERS)}"
        )
    return _LOADERS[suffix](path)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (avoids pulling in nltk/spacy)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital/quote.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into ~chunk_size-word chunks, sentence-aware, with a sliding
    overlap between consecutive chunks so retrieval doesn't lose context at
    chunk boundaries.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > chunk_size and current:
            chunks.append(" ".join(current))
            # carry the tail of the previous chunk forward for overlap
            overlap_words = " ".join(current).split()[-overlap:] if overlap else []
            current = [" ".join(overlap_words)] if overlap_words else []
            current_len = len(overlap_words)
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def load_and_chunk(
    path: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Load a document and return it as a list of Chunk objects."""
    path = Path(path)
    text = load_document(path)
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    return [
        Chunk(
            chunk_id=f"{path.name}::{i}",
            text=chunk,
            source=path.name,
            chunk_index=i,
        )
        for i, chunk in enumerate(raw_chunks)
    ]
