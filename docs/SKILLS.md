# Skills Demonstrated

What this project shows you can do, mapped to the technologies used and to
the resume bullets it supports. Useful as interview prep — for each row you
should be able to explain *why* that choice was made, not just that it was
used.

## Core skills

| Skill | Where it shows up | Why it was the right choice here |
|---|---|---|
| **RAG system design** | `src/rag_pipeline.py` | Grounding LLM answers in retrieved context reduces hallucination vs. asking a model to answer from parametric memory alone, and lets you cite sources. |
| **Vector search / FAISS** | `src/vector_store.py` | `IndexFlatIP` gives exact nearest-neighbor search — appropriate at this scale (thousands of chunks); the docs note how to swap to `IndexIVFFlat`/`HNSW` for approximate search at larger scale. |
| **Embeddings** | `src/embeddings.py` | `sentence-transformers` (`all-MiniLM-L6-v2`) runs locally, free, and is a standard, fast, well-benchmarked sentence embedding model — a deliberate choice to keep the embedding step free and offline-capable. |
| **Text chunking strategy** | `src/document_loader.py` | Sentence-aware chunking with overlap, not naive fixed-width splitting — avoids severing a sentence mid-thought at a chunk boundary, which directly affects retrieval quality. |
| **LLM API integration** | `src/rag_pipeline.py` | Structured prompting (system prompt constrains the model to answer only from provided context), swappable backend (originally Anthropic, now Groq — same interface pattern, different provider). |
| **Retrieval evaluation methodology** | `evaluation/evaluate.py`, `evaluation/eval_qa.json` | Recall@k and MRR are standard information-retrieval metrics — this is the difference between "I think it works" and "I measured that it works," which is the detail that makes a resume bullet credible. |
| **Web application development** | `app.py` | Streamlit UI with session state management, file upload, and a chat interface — demonstrates you can wrap a backend pipeline in something a non-technical user could actually use. |
| **Python packaging / project structure** | whole repo | Clean separation of concerns (`src/` library code vs. `app.py` UI vs. `evaluation/` scripts), `requirements.txt`, `.env`-based configuration — the shape of a project someone else could clone and run. |

## Concepts worth being able to explain out loud

- **Why cosine similarity for embeddings, and how `IndexFlatIP` computes it**
  (L2-normalize vectors, then inner product = cosine similarity — done in
  `EmbeddingModel.embed()`)
- **Why chunk overlap matters** — without it, a fact split across a chunk
  boundary can become unretrievable because neither chunk contains the full
  context
- **What Recall@k and MRR actually measure**, and why they're evaluating
  *retrieval*, not the final generated answer — those are different failure
  modes (bad retrieval = right info never reaches the model; bad generation
  = model had the right info and still answered poorly)
- **The tradeoff of `IndexFlatIP` vs. approximate indexes** (`IndexIVFFlat`,
  `HNSW`) — exact search is fine up to tens of thousands of vectors;
  approximate search trades a small accuracy hit for much better scaling
- **Why grounding reduces hallucination but doesn't eliminate it** — the
  system prompt tells the model to answer only from context, but a
  sufficiently vague or adversarial question can still produce a
  plausible-sounding but ungrounded answer, which is why the sources are
  always shown alongside the answer

## Stack summary (for a resume line)

`Python · FAISS · sentence-transformers · Streamlit · Groq (Llama 3.3) ·
RAG · vector search · information retrieval evaluation (Recall@k, MRR)`
