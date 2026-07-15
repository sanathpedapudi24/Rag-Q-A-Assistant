# AGENTS.md

Instructions for AI coding agents (Claude Code, Cursor, Copilot, etc.) working
in this repository. Read this before making changes.

## What this project is

A Retrieval-Augmented Generation (RAG) document Q&A assistant. Documents are
chunked, embedded locally with `sentence-transformers`, indexed in `FAISS`,
and retrieved chunks are passed to an LLM (Groq's free API) to generate a
grounded, cited answer. There's a Streamlit UI (`app.py`) and a retrieval
evaluation harness (`evaluation/`).

## Setup / run commands

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add GROQ_API_KEY
```

- Run the app: `streamlit run app.py` (or `python -m streamlit run app.py` if
  the `streamlit` command isn't on PATH — common on Windows)
- Run the eval harness: `python -m evaluation.evaluate --docs data/sample --eval evaluation/eval_qa.json`
- Syntax-check everything: `python -m py_compile src/*.py app.py evaluation/evaluate.py`

## Architecture / data flow

```
document_loader.py  (load + chunk text)
        │
embeddings.py        (chunks -> vectors, sentence-transformers, local, free)
        │
vector_store.py       (FAISS IndexFlatIP: add / search / save / load)
        │
rag_pipeline.py       (retrieve top-k -> prompt LLM via Groq -> answer + sources)
        │
app.py                 (Streamlit: upload, build index, chat)
```

`RAGPipeline` in `src/rag_pipeline.py` is the composition root — it owns an
`EmbeddingModel`, a `FAISSVectorStore`, and a Groq client, and exposes
`ingest()`, `retrieve()`, and `answer()`. Both `app.py` and
`evaluation/evaluate.py` build one of these and call those three methods;
they don't touch `EmbeddingModel` or `FAISSVectorStore` directly.

## Conventions and constraints

- **Generation backend is Groq, not Anthropic/OpenAI.** The `groq` package
  is used with its OpenAI-compatible `chat.completions.create` interface.
  Do not reintroduce an `anthropic` or `openai` dependency unless the user
  explicitly asks to switch providers — this project intentionally runs on
  Groq's free tier (no credit card, no cost).
- **Embeddings are always local** (`sentence-transformers`, default model
  `all-MiniLM-L6-v2`). Never route embedding calls through a paid API.
- **Vector search uses inner product, not L2.** `EmbeddingModel.embed()`
  L2-normalizes output, so `FAISSVectorStore` uses `IndexFlatIP` — inner
  product on unit vectors equals cosine similarity. If you change the
  embedding model or normalization, keep this invariant or search quality
  will silently degrade.
- **Config lives in `.env`**, read via `python-dotenv` in `app.py`. Defaults
  are duplicated as function-argument defaults in `RAGPipeline.__init__` —
  keep both in sync if you change a default (`GROQ_MODEL`, `EMBEDDING_MODEL`,
  `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`).
- **`Chunk` is a dataclass** (`src/document_loader.py`) used across the
  whole pipeline and persisted via `FAISSVectorStore.save()`
  (`pickle` of `asdict(chunk)`). If you add/rename a field, `load()` must
  still be able to reconstruct it — check `vector_store.py`'s save/load pair
  together.
- **`evaluation/eval_qa.json` format:** each entry has `question`,
  `relevant_source` (filename that should be retrieved), and optional
  `expected_answer_contains` (keywords for the `--check-answers` end-to-end
  check). Keep new eval questions in this shape.

## Known environment gotchas (don't "fix" these — they're expected)

- `sentence-transformers` downloads model weights from huggingface.co on
  first run. This requires real internet access; it will fail in fully
  offline/sandboxed environments. That's expected, not a bug in the code.
- On Windows, `pip`-installed console scripts (like `streamlit.exe`) can
  land in a directory that's not on `PATH`. Prefer `python -m streamlit run
  app.py` over the bare `streamlit` command in docs/instructions.
- `pip install --break-system-packages` may be needed in externally-managed
  Python environments (e.g. some Linux distros) — not needed on the
  Windows/venv setup this project targets by default.

## Before committing a change

1. `python -m py_compile src/*.py app.py evaluation/evaluate.py` — must pass
2. If you touched `document_loader.py`, `embeddings.py`, or `vector_store.py`,
   sanity-check chunking/retrieval logic still works (see
   `docs/PROJECT_STRUCTURE.md` for what each module owns)
3. If you added a dependency, add it to `requirements.txt` with a version
   floor, and mention it in `README.md`'s setup section if it needs an API
   key or external account
4. Don't commit `.env` (it's meant to be copied from `.env.example` locally,
   not tracked)
