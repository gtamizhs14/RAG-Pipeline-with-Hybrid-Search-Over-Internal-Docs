# RAG Pipeline — Hybrid Search over Internal Documents

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions about your own documents with grounded, cited, verified answers.

**Tech stack:** Sentence Transformers · ChromaDB · BM25 · Reciprocal Rank Fusion · Cross-encoder reranker · Groq (any OpenAI-compatible LLM) · FastAPI · Streamlit

---

## What it does

Drop in any documents (PDF, Markdown, HTML, plain text) and ask questions about them. The pipeline:

1. **Retrieves** relevant passages using hybrid search (dense embeddings + BM25 keyword matching, fused with RRF)
2. **Reranks** candidates with a cross-encoder for higher precision
3. **Generates** a grounded answer with inline `[1]`, `[2]` citations
4. **Verifies** each citation with an LLM-as-judge
5. **Scores** answer confidence (retrieval quality + citation coverage + completeness)

---

## Quickstart

### 1. Clone and install

```bash
git clone <repo-url>
cd rag-pipeline
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your LLM API key. The default provider is **Groq** (free at [console.groq.com](https://console.groq.com)):

```env
LLM_API_KEY=gsk_your_key_here
```

To use **OpenAI** instead, change three lines — no code changes needed:

```env
LLM_API_KEY=sk-your_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

For **Ollama** (fully local, no API key):

```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3
```

### 3. Add your documents

The folder `data/raw/` is where your documents go. It already exists in the repo — just drop files in:

```
rag-pipeline/
└── data/
    └── raw/          ← PUT YOUR FILES HERE
        ├── my_report.pdf
        ├── internal_wiki.md
        └── product_manual.html
```

Supported formats: `.txt` `.md` `.pdf` `.html`

Then run the ingestion pipeline to index them:

```bash
python ingest.py
```

> **Just want to try it first?** Use the included sample corpus in
> [`scripts/seed_corpus/`](scripts/seed_corpus/) — 5 documents covering
> One Piece characters, Devil Fruits, Haki, story arcs, and the world/factions.
> Copy them into the index with:
> ```bash
> python scripts/seed.py
> ```
> Then ask questions like:
> - *"What is Gear 5 and how did Luffy unlock it?"*
> - *"What are the three types of Haki?"*
> - *"How does Blackbeard have two Devil Fruit powers?"*
> - *"What happened at Marineford?"*

### 4. Start the servers

**Terminal 1 — API:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — UI:**
```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### 5. Ask questions

Type a question about your documents in the text box and click **Ask**. Questions should be specific to the content you indexed — the pipeline only knows what's in your documents.

Example questions if you used the seed corpus:
- *"What are the three types of Haki?"*
- *"How does Blackbeard have two Devil Fruit powers?"*
- *"What happened at Marineford?"*
- *"Who are the Four Emperors after Wano?"*

**Sidebar options:**
- **Top-N**: how many source chunks to retrieve (default 5)
- **Use cross-encoder reranker**: improves precision, adds ~1s latency
- **Skip citation verification**: faster responses, disables the verified/unverified badges
- **Compare hybrid vs. dense-only**: runs your question twice and shows results side by side so you can see how much BM25 helps
- **Show indexed documents**: lists every document currently in the index

---

## Using Docker

```bash
cp .env.example .env   # add your LLM_API_KEY

# Seed sample documents (first time only)
docker compose --profile seed run seed

# Start API + Streamlit
docker compose up
```

- API: http://localhost:8000
- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## Project structure

```
rag-pipeline/
├── src/
│   ├── ingestion/       # Document loading, chunking, embedding, dedup
│   ├── retrieval/       # Dense, sparse, RRF fusion, cross-encoder reranker
│   ├── generation/      # LLM client, citation parser, verifier, confidence scorer
│   ├── eval/            # Retrieval metrics, LLM-as-judge generation metrics
│   └── api/             # FastAPI routes, Pydantic schemas
├── scripts/
│   └── seed.py          # Generates and ingests sample documents
├── streamlit_app.py     # Streamlit UI
├── ingest.py            # CLI ingestion entry point
├── run_eval.py          # Evaluation runner CLI
├── eval_dataset.json    # 55-question evaluation dataset
├── .env.example         # Environment variable template
└── docker-compose.yml   # API + Streamlit + seed services
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Pipeline status |
| `POST` | `/api/v1/query` | Ask a question |
| `GET` | `/api/v1/documents` | List indexed documents |
| `POST` | `/api/v1/ingest` | Upload a new document |

Full interactive docs at **http://localhost:8000/docs**.

---

## Running evaluation

```bash
# Full eval (retrieval + generation metrics, uses LLM-as-judge)
python run_eval.py

# Retrieval metrics only (no LLM calls — fast)
python run_eval.py --retrieval-only

# Custom dataset or output path
python run_eval.py --dataset path/to/eval.json --output results/out.json
```

Metrics reported: Precision@K, Recall@K, MRR, NDCG@K, Hit Rate, Faithfulness, Answer Relevance, Composite Confidence.

---

## Swapping LLM providers

All LLM calls go through `LLMClient` in `src/generation/llm_client.py`, which uses the OpenAI SDK pointed at a configurable `base_url`. Any OpenAI-compatible API works with zero code changes — just update `.env`.

| Provider | `LLM_BASE_URL` |
|----------|----------------|
| Groq | `https://api.groq.com/openai/v1` |
| OpenAI | `https://api.openai.com/v1` |
| Ollama | `http://localhost:11434/v1` |
| Together AI | `https://api.together.xyz/v1` |

---

## Environment variables

See `.env.example` for the full list with descriptions. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | API key for your LLM provider |
| `LLM_BASE_URL` | Groq endpoint | OpenAI-compatible base URL |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Generation model |
| `LLM_JUDGE_MODEL` | `llama-3.3-70b-versatile` | Citation verification model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (no API key needed) |
| `CHROMA_DB_PATH` | `data/chroma_db` | ChromaDB persistence path |
| `RETRIEVAL_TOP_K` | `20` | Candidates per retriever before fusion |
| `RERANK_TOP_N` | `5` | Final chunks after reranking |
