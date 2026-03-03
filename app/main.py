"""
main.py - Streamlit entry point for the Swiggy Annual Report RAG system.

UI Layout:
- Sidebar: PDF upload + "Process Document" button
- Main area: Q&A chat interface with answer + expandable context
"""

# ── Ensure the project root (ai-doc-rag/) is on sys.path ─────────────────────
# This is required because Streamlit runs this file directly (not as a module),
# so relative imports like `from app.config import ...` need the root in path.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ai-doc-rag/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import os
import hashlib
import tempfile

import streamlit as st

from app.config import validate_config
from app.utils.logger import get_logger
from app.ingestion.pdf_loader import load_pdf_as_images
from app.ingestion.ocr_engine import run_ocr
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import init_schema, clear_document, upsert_chunks
from app.retrieval.similarity_search import search
from app.retrieval.prompt_builder import build_prompt
from app.llm.gemini_client import generate_answer

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swiggy RAG · Document Intelligence",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,165,0,0.2);
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #ff6b00, #ff9500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

/* Answer card */
.answer-card {
    background: rgba(255, 107, 0, 0.08);
    border: 1px solid rgba(255, 107, 0, 0.3);
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-top: 1rem;
    color: #f0f4ff;
    line-height: 1.7;
}

/* Context card */
.context-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.875rem;
    color: #94a3b8;
}

/* Page badge */
.page-badge {
    display: inline-block;
    background: rgba(255, 107, 0, 0.25);
    color: #ff9500;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Chat input */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,107,0,0.4) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff6b00, #ff9500) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Divider */
hr { border-color: rgba(255,107,0,0.2) !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _document_id_from_file(name: str) -> str:
    """Generate a stable document_id from the filename."""
    return hashlib.md5(name.encode()).hexdigest()[:12]


def _run_ingestion_pipeline(pdf_path: str, document_id: str) -> int:
    """
    Full ingestion: PDF → OCR → clean → chunk → embed → store.
    Returns number of chunks stored.
    """
    prog = st.progress(0, text="📄 Loading PDF pages...")

    pages_images = load_pdf_as_images(pdf_path)
    prog.progress(15, text=f"🔍 Running OCR on {len(pages_images)} pages...")

    raw_pages = run_ocr(pages_images)
    prog.progress(40, text="🧹 Cleaning text...")

    clean = clean_pages(raw_pages)
    prog.progress(55, text="✂️ Chunking text...")

    chunks = chunk_pages(clean, document_id=document_id)
    prog.progress(65, text=f"🧠 Embedding {len(chunks)} chunks via Gemini...")

    embedded = embed_chunks(chunks)
    prog.progress(85, text="💾 Storing embeddings in Neon...")

    clear_document(document_id)
    count = upsert_chunks(embedded)

    prog.progress(100, text=f"✅ Done! {count} chunks stored.")
    return count


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🍊 Swiggy RAG")
    st.markdown("**Document Intelligence System**")
    st.divider()

    st.markdown("### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload the Swiggy Annual Report PDF",
        type=["pdf"],
        help="Upload a PDF to process and index.",
    )

    process_btn = st.button(
        "⚙️ Process Document",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )

    st.divider()
    st.markdown("""
    **How it works:**
    1. Upload the PDF
    2. Click "Process Document" (takes a few minutes)
    3. Ask questions in the chat below

    ---
    **Stack:**  
    🔎 Tesseract OCR  
    🧠 Gemini Embeddings  
    🗄️ Neon + pgvector  
    🤖 Gemini 1.5 Flash  
    """)


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown('<p class="hero-header">🍊 Swiggy Annual Report · AI Q&A</p>', unsafe_allow_html=True)
st.markdown("Ask any question about Swiggy's annual report. Answers are **strictly grounded** in the document — no hallucinations.")
st.divider()

# ── Validate env on first load ────────────────────────────────────────────────
try:
    validate_config()
except EnvironmentError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.stop()

# ── Initialise DB schema (idempotent) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_db():
    init_schema()

_init_db()

# ── Handle ingestion ──────────────────────────────────────────────────────────
if process_btn and uploaded_file is not None:
    doc_id = _document_id_from_file(uploaded_file.name)
    st.session_state["document_id"] = doc_id
    st.session_state["document_name"] = uploaded_file.name

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing document — this may take several minutes for large PDFs..."):
        try:
            count = _run_ingestion_pipeline(tmp_path, doc_id)
            st.success(f"✅ **{uploaded_file.name}** processed successfully! **{count}** chunks indexed.")
        except Exception as exc:
            st.error(f"❌ Ingestion failed: {exc}")
            logger.error(f"Ingestion error: {exc}", exc_info=True)
        finally:
            os.unlink(tmp_path)

# ── Q&A Interface ─────────────────────────────────────────────────────────────
doc_id = st.session_state.get("document_id")
doc_name = st.session_state.get("document_name", "No document loaded")

if doc_id:
    st.markdown(f"**📄 Active Document:** `{doc_name}`")
else:
    st.info("⬅️ Upload and process a PDF from the sidebar to start asking questions.")

st.markdown("### 💬 Ask a Question")
query = st.text_input(
    label="Your question",
    placeholder="e.g. What was Swiggy's total revenue in FY2024?",
    label_visibility="collapsed",
)

ask_btn = st.button("🔍 Ask", disabled=(not query or not doc_id), use_container_width=False)

if ask_btn and query and doc_id:
    with st.spinner("Searching document and generating answer..."):
        try:
            # Retrieve relevant chunks
            chunks = search(query=query, document_id=doc_id)

            # Build grounded prompt
            prompt = build_prompt(query=query, context_chunks=chunks)

            # Generate answer
            answer = generate_answer(prompt)

            # ── Display answer ──
            st.markdown("#### 📝 Answer")
            st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

            # ── Display supporting context ──
            if chunks:
                with st.expander("📚 Supporting Context (from document)", expanded=False):
                    for i, chunk in enumerate(chunks, start=1):
                        st.markdown(
                            f'<div class="context-card">'
                            f'<span class="page-badge">Page {chunk["page_number"]}</span><br>'
                            f'{chunk["content"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.warning("No relevant context found in the document for this query.")

        except Exception as exc:
            st.error(f"❌ Error generating answer: {exc}")
            logger.error(f"Q&A error: {exc}", exc_info=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.8rem;'>"
    "Swiggy Annual Report RAG · Powered by Gemini + Neon pgvector · "
    "Answers grounded strictly in document context"
    "</p>",
    unsafe_allow_html=True,
)
