# RAG Document Q&A Assistant

A Retrieval-Augmented Generation pipeline that answers questions over your own
documents (PDF, DOCX, TXT, MD). Documents are chunked, embedded locally with
`sentence-transformers`, indexed in `FAISS`, and retrieved chunks are passed
to an LLM (via Groq's free API) to generate a grounded, cited answer. Includes
a Streamlit UI and a retrieval-accuracy evaluation harness.

## Live Demo

**[Try it now](https://rag-qna-assistant.streamlit.app/)** — no setup required. Upload your docs and start asking questions.

## Architecture

```
 Documents (pdf/docx/txt)
        │
        ▼
 document_loader.py  ── extract text, chunk into overlapping windows
        │
        ▼
 embeddings.py        ── sentence-transformers -> vector per chunk
        │
        ▼
 vector_store.py       ── FAISS IndexFlatIP (cosine similarity search)
        │
        ▼
 rag_pipeline.py       ── retrieve top-k chunks -> prompt LLM (Groq) -> answer
        │
        ▼
 app.py (Streamlit)    ── upload docs, chat interface, shows cited sources
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY (free, no credit card — see below)
```

### Getting a free Groq API key (required)

The app uses [Groq](https://console.groq.com) to run the LLM — it's **100% free**, no credit card needed.

**Step-by-step:**

1. Go to **[console.groq.com](https://console.groq.com)** and click **Sign Up**
2. Sign in with **Google**, **GitHub**, or **email** (takes 10 seconds)
3. Once logged in, click **API Keys** in the left sidebar
4. Click **Create API Key** → give it a name (e.g. `rag-app`) → click **Create**
5. **Copy** the key (starts with `gsk_...`) and paste it into one of:
   - **Local**: edit `.env` and set `GROQ_API_KEY=gsk_...`
   - **Live demo**: go to the app → paste it in the sidebar text box

> The free tier gives you plenty of requests for development and demos. You'll never be asked for a credit card.

## Run the app

```bash
streamlit run app.py
```

Upload one or more documents in the sidebar, click **Build / update index**,
then ask questions in the chat box. Each answer shows the source chunks it
was grounded in.

## Run from the command line

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()  # reads GROQ_API_KEY from the environment
pipeline.ingest(["data/sample/renewable_energy.txt", "data/sample/space_exploration.txt"])

result = pipeline.answer("What percentage of global electricity comes from renewables?")
print(result["answer"])
for s in result["sources"]:
    print(f"  - {s['source']} (score {s['score']:.2f})")
```

## Evaluating retrieval accuracy

`evaluation/eval_qa.json` contains a labeled set of questions, each tagged with
the source document that should be retrieved. `evaluation/evaluate.py` ingests
the sample documents, runs every question through the pipeline, and reports
**Recall@1 / Recall@3 / Recall@5** and **Mean Reciprocal Rank (MRR)**.

```bash
python -m evaluation.evaluate --docs data/sample --eval evaluation/eval_qa.json

# also check end-to-end answer accuracy via the LLM (uses your free Groq API key)
python -m evaluation.evaluate --docs data/sample --eval evaluation/eval_qa.json --check-answers
```

To evaluate against your own documents, drop them in a folder, write a
matching `eval_qa.json` (question + `relevant_source` filename + optional
`expected_answer_contains` keywords), and point `--docs` / `--eval` at them.

## Configuration

All defaults live in `.env` / `.env.example`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | required (free — see setup above) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | generation model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | 500 | approx. words per chunk |
| `CHUNK_OVERLAP` | 50 | words of overlap between chunks |
| `TOP_K` | 5 | chunks retrieved per query |

## Project layout

```
rag-qa-assistant/
├── app.py                     # Streamlit UI
├── AGENTS.md                  # instructions for AI coding agents working in this repo
├── src/
│   ├── document_loader.py     # PDF/DOCX/TXT loading + chunking
│   ├── embeddings.py          # sentence-transformers wrapper
│   ├── vector_store.py        # FAISS index (save/load supported)
│   └── rag_pipeline.py        # ingest + retrieve + generate
├── evaluation/
│   ├── eval_qa.json           # labeled Q&A set
│   └── evaluate.py            # Recall@k / MRR harness
├── data/sample/                # example documents to try it out on
├── docs/
│   ├── PROJECT_STRUCTURE.md   # detailed file-by-file reference
│   └── SKILLS.md              # tech stack mapped to skills/interview prep
├── requirements.txt
└── .env.example
```

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for a detailed
walkthrough of what each file does and how data flows through the pipeline,
and [`docs/SKILLS.md`](docs/SKILLS.md) for how this project maps to
resume/interview talking points.

## Possible extensions

- Swap `IndexFlatIP` for `IndexIVFFlat` / `IndexHNSWFlat` for larger corpora
- Add hybrid search (BM25 + embeddings) for better keyword-heavy queries
- Persist the FAISS index (`RAGPipeline.save_index` / `load_index`) so the
  app doesn't need to re-embed documents on every restart
- Add re-ranking of retrieved chunks with a cross-encoder before generation
