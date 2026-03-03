# 🍊 Swiggy Annual Report — AI Document Intelligence (RAG)

A **production-ready Retrieval-Augmented Generation (RAG)** system that answers natural language questions about the Swiggy Annual Report, strictly grounded in document content. No hallucinations.

---

## 📄 Document Source

**Swiggy Annual Report FY 2023–24**
Source: [https://investors.swiggy.com/annual-reports](https://investors.swiggy.com/annual-reports)
File used: `Annual-Report-FY-2023-24.pdf`

---

## 🏗 Architecture

```
PDF Upload → pdf2image → Tesseract OCR → Text Cleaner
→ Chunker (500 chars, 100 overlap)
→ Gemini Embeddings (768-dim) → Neon pgvector DB
─────────────────────────────────────────────────
User Question → Gemini Embeddings → Cosine Search
→ Top-5 Chunks → Strict Grounded Prompt
→ Gemini 1.5 Flash (temp=0.1) → Answer + Page Citations
```

---

## 🚀 Quick Start (Local)

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

### 1. Install dependencies

```bash
cd ai-doc-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in GOOGLE_API_KEY and NEON_DATABASE_URL
```

| Variable | Source |
|---|---|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `NEON_DATABASE_URL` | [Neon.tech](https://neon.tech) — free PostgreSQL + pgvector |

### 3. Run

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501)

### 4. Use the App

1. Upload `Annual-Report-FY-2023-24.pdf` in the sidebar
2. Click **"Process Document"** — OCR + embedding (~10–20 min for full PDF)
3. Ask questions — e.g. *"What was Swiggy's total revenue in FY2024?"*

---

## 🐳 Docker

```bash
docker build -t swiggy-rag .
docker run -p 8501:8501 --env-file .env swiggy-rag
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

All 24 tests run fully offline (APIs mocked).

---

## 🛡 Hallucination Mitigation

| Technique | Setting |
|---|---|
| Low temperature | `0.1` |
| Restricted top-p | `0.8` |
| Strict system prompt | Answer ONLY from context |
| Refusal instruction | Say "not available" if absent |
| Citation enforcement | Always cite page numbers |
| Limited context | Top-5 chunks only |

---

## 📁 Project Structure

```
ai-doc-rag/
├── app/
│   ├── main.py              # Streamlit UI
│   ├── config.py            # Config + env vars
│   ├── ingestion/
│   │   ├── pdf_loader.py    # PDF → images
│   │   ├── ocr_engine.py    # Tesseract OCR
│   │   ├── cleaner.py       # Text cleaning
│   │   ├── chunker.py       # Chunking with metadata
│   │   └── embedder.py      # Gemini Embeddings
│   ├── retrieval/
│   │   ├── vector_store.py  # Neon + pgvector
│   │   ├── similarity_search.py
│   │   └── prompt_builder.py
│   ├── llm/
│   │   └── gemini_client.py
│   └── utils/logger.py
├── tests/
│   ├── test_ingestion.py
│   └── test_retrieval.py
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
| OCR | Tesseract + pdf2image |
| Embeddings | Gemini `embedding-001` (768-dim) |
| Vector DB | Neon PostgreSQL + pgvector |
| LLM | Gemini 1.5 Flash |
| UI | Streamlit |
| Deployment | Docker |
