# Project Structure

```
rag-qa-assistant/
├── AGENTS.md                    Instructions for AI coding agents (read this if you're one)
├── README.md                    Setup, usage, and architecture overview
├── requirements.txt              Python dependencies
├── .env.example                  Template for required environment variables
├── app.py                        Streamlit web UI (entry point for the app)
│
├── src/                          Core library code
│   ├── __init__.py
│   ├── document_loader.py        Load PDF/DOCX/TXT/MD, split into overlapping chunks
│   ├── embeddings.py             Wraps sentence-transformers: text -> vectors
│   ├── vector_store.py           FAISS index wrapper: add / search / save / load
│   └── rag_pipeline.py           Ties it together: ingest -> retrieve -> generate
│
├── evaluation/                   Retrieval-quality measurement
│   ├── eval_qa.json              10 labeled question -> source-document pairs
│   └── evaluate.py               Computes Recall@1/3/5 and MRR; optional answer-accuracy check
│
├── data/sample/                  Example documents to try the app on
│   ├── renewable_energy.txt
│   └── space_exploration.txt
│
└── docs/                         Supplementary documentation
    ├── PROJECT_STRUCTURE.md      This file
    └── SKILLS.md                 Technologies/skills demonstrated, mapped to resume bullets
```

## File-by-file reference

### `app.py`
Streamlit entry point. Run with `streamlit run app.py`. Holds UI state in
`st.session_state` (the pipeline instance, chat history, list of indexed
filenames). The sidebar handles API key entry, file upload, and index
building; the main panel is a chat interface that calls
`RAGPipeline.answer()` per question and renders the answer plus an
expandable "Sources" section.

### `src/document_loader.py`
- `load_document(path)` — dispatches to a PDF/DOCX/TXT reader based on file
  extension, returns raw text.
- `chunk_text(text, chunk_size, overlap)` — sentence-aware chunker. Splits
  into sentences first (regex-based, no external NLP library), then packs
  sentences into ~`chunk_size`-word windows, carrying the last `overlap`
  words of each chunk into the start of the next one so retrieval doesn't
  lose context at a chunk boundary.
- `load_and_chunk(path, ...)` — convenience wrapper returning a list of
  `Chunk` dataclass instances (`chunk_id`, `text`, `source`, `chunk_index`,
  `metadata`).

### `src/embeddings.py`
- `EmbeddingModel` — thin wrapper around
  `sentence_transformers.SentenceTransformer`. `.embed(texts)` returns an
  L2-normalized `float32` numpy array; normalization is what lets the
  vector store use a cheap inner-product search as a stand-in for cosine
  similarity. `.embed_query(text)` is a single-string convenience method.

### `src/vector_store.py`
- `FAISSVectorStore` — owns a `faiss.IndexFlatIP` plus a parallel Python
  list of `Chunk` objects (index `i` in the FAISS index corresponds to
  `self.chunks[i]`). `.add()`, `.search()`, `.save()`/`.load()` (index to
  `index.faiss`, chunk metadata to `chunks.pkl` via pickle, dimension to
  `config.json`).

### `src/rag_pipeline.py`
- `RAGPipeline` — the composition root. Constructed with an embedding model
  name, an LLM model name (Groq), chunking parameters, and top-k.
  - `.ingest(paths)` — load + chunk + embed + add to the vector store.
  - `.retrieve(query, k)` — embed the query, return `[(score, Chunk), ...]`.
  - `.answer(query, k)` — retrieve, build a context-stuffed prompt, call
    Groq's chat completions API, return `{"answer": str, "sources": [...]}`.
  - `.save_index()` / `.load_index()` — persist/restore the FAISS index so
    you don't have to re-embed documents every run.

### `evaluation/evaluate.py`
CLI script. Ingests every document in `--docs`, runs every question in
`--eval` through `RAGPipeline.retrieve()`, and reports:
- **Recall@k** — fraction of questions where the correct source document
  appeared in the top-k retrieved chunks
- **MRR** (Mean Reciprocal Rank) — average of `1/rank` of the first correct
  hit, rewarding retrieving the right chunk *higher*, not just present

Pass `--check-answers` to additionally call the LLM for each question and
keyword-match the generated answer against `expected_answer_contains` —
a rough proxy for end-to-end (not just retrieval) quality.

### `evaluation/eval_qa.json`
Each entry: `question`, `relevant_source` (filename), and
`expected_answer_contains` (list of keywords). To evaluate against your own
documents, add files to a folder and write matching entries here.

## Data flow at a glance

```
 Upload/point at documents
        │
        ▼
 load_and_chunk()  ──►  list[Chunk]
        │
        ▼
 EmbeddingModel.embed()  ──►  float32[n_chunks, dim]
        │
        ▼
 FAISSVectorStore.add()  ──►  index now searchable
        │
   (user asks a question)
        │
        ▼
 EmbeddingModel.embed_query()  ──►  float32[dim]
        │
        ▼
 FAISSVectorStore.search()  ──►  top-k [(score, Chunk), ...]
        │
        ▼
 Groq chat.completions.create(system + context + question)
        │
        ▼
 {"answer": "...", "sources": [...]}  ──►  rendered in Streamlit / printed by evaluate.py
```
