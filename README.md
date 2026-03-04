# 📄 Document AI Q&A — RAG System

A **production-ready Retrieval-Augmented Generation (RAG)** system that answers natural language questions about uploaded PDF documents, strictly grounded in document content. No hallucinations.

Built with **LlamaIndex**, **LlamaParse**, **Gemini**, and **Neon PostgreSQL (pgvector)**.

---

## 🏗 Architecture

```
PDF Upload → LlamaParse (Cloud Markdown extraction)
           → Metadata tagging (document_name injected per chunk)
           → SentenceSplitter (512 tokens, 64 overlap)
           → HuggingFace BGE-small (384-dim, LOCAL — no API calls) → Neon pgvector DB
─────────────────────────────────────────────────────────────────────────────────────
User Question → HuggingFace BGE-small (LOCAL embed) → Cosine Search
             → Top-5 Chunks → Strict Grounded Prompt
             → Gemini 2.0 Flash (temp=0.1) → Answer + Page Citations
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed component diagram.

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/app/apikey) API key (for the LLM only)
- A [Neon.tech](https://neon.tech) PostgreSQL database (free tier works)
- A [LlamaCloud](https://cloud.llamaindex.ai) API key (for LlamaParse)

> **Note:** No embedding API key needed — embeddings run locally via HuggingFace.
> The model (`BAAI/bge-small-en-v1.5`, ~133 MB) is downloaded automatically on first run.

### 1. Install dependencies

```bash
cd ai-doc-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in GOOGLE_API_KEY, NEON_DATABASE_URL, LLAMA_CLOUD_API_KEY
```

| Variable | Source |
|---|---|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) — LLM only |
| `NEON_DATABASE_URL` | [Neon.tech](https://neon.tech) — free PostgreSQL + pgvector |
| `LLAMA_CLOUD_API_KEY` | [LlamaCloud](https://cloud.llamaindex.ai) — for LlamaParse |

### 3. Run

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501)

### 4. Use the App

1. Upload any PDF document in the sidebar 
   (e.g., [Annual-Report-FY-2023-24.pdf](Annual-Report-FY-2023-24.pdf))
2. Click **"Process Document"** — LlamaParse extracts tables + text as Markdown
3. Ask questions — e.g. *"What was the total revenue in FY2024?"*
4. Upload additional documents — they are added to the same index with metadata tagging

---

## 📚 Multi-Document Support

All documents share the same pgvector table (`document_chunks_llama`). Each chunk
is tagged with a `document_name` in its metadata during ingestion, so:

- Source chunks display which document they came from
- The LLM is instructed to mention the source document when relevant
- Duplicate uploads (same file hash) are rejected automatically

---

## 🐳 Docker

```bash
docker build -t doc-rag .
docker run -p 8501:8501 --env-file .env doc-rag
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

All tests run fully offline (external APIs mocked).

---

## 🛡 Hallucination Mitigation

| Technique | Setting |
|---|---|
| Low temperature | `0.1` |
| Strict system prompt | Answer ONLY from context |
| Refusal instruction | Say "not available" if absent |
| Citation enforcement | Always cite page numbers |
| Table fidelity | LlamaParse preserves Markdown tables |
| Limited context | Top-5 chunks only |
| Document source | Prompt mentions document_name metadata |

---

## 📁 Project Structure

```
ai-doc-rag/
├── app/
│   ├── main.py                  # Streamlit UI (chat + upload + multi-doc)
│   ├── config.py                # Config, env vars, URL helpers
│   ├── llm/
│   │   └── __init__.py          # Centralized LlamaIndex Settings
│   ├── ingestion/
│   │   └── pipeline.py          # LlamaParse → tag → chunk → embed → Neon
│   ├── retrieval/
│   │   └── query_engine.py      # Load index → QueryEngine + strict prompt
│   └── utils/
│       └── logger.py            # Structured logging
├── tests/
│   ├── test_ingestion.py        # Ingestion pipeline + metadata tests
│   └── test_retrieval.py        # Query engine + prompt tests
├── Dockerfile
├── requirements.txt
├── .env.example
├── ARCHITECTURE.md
└── README.md
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| PDF Parsing | LlamaParse (cloud) via `llama-index-readers-llama-parse` |
| Chunking | LlamaIndex SentenceSplitter (512 tokens / 64 overlap) |
| Embeddings | **HuggingFace `BAAI/bge-small-en-v1.5` (384-dim, LOCAL — no API key)** via `llama-index-embeddings-huggingface` |
| Vector DB | Neon PostgreSQL + pgvector via `llama-index-vector-stores-postgres` |
| LLM | GoogleGenAI `gemini-2.0-flash` (temp=0.1) via `llama-index-llms-google-genai` |
| Framework | LlamaIndex Core |
| UI | Streamlit |
| Deployment | Docker |
