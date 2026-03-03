"""
main.py — Streamlit UI for the Document RAG System (LlamaIndex edition).

Handles:
  - PDF upload → triggers LlamaParse ingestion pipeline
  - Multi-document support (same vector table, metadata-tagged)
  - Chat interface → calls QueryEngine for grounded answers
"""

# ── Ensure the project root (ai-doc-rag/) is on sys.path ─────────────────────
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ai-doc-rag/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
from app.config import validate_config
from app.ingestion.pipeline import run_ingestion_pipeline, get_ingested_doc_names
from app.retrieval.query_engine import get_query_engine
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document AI Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0f0f0f; color: #e8e8e8; }
    .main-header {
        background: linear-gradient(135deg, #4A90D9 0%, #67B8F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .context-box {
        background: #1e1e1e;
        border-left: 3px solid #4A90D9;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #aaa;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        background: #4A90D9;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Validate config on startup ─────────────────────────────────────────────

try:
    validate_config()
except EnvironmentError as e:
    st.error(f"⚠️ Configuration Error:\n\n{e}")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "documents" not in st.session_state:
    # Restore previously-ingested doc names from the DB on first load
    try:
        st.session_state.documents = get_ingested_doc_names()
    except Exception:
        st.session_state.documents = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # bump to reset file_uploader widget

# Auto-connect to existing index if documents already exist in DB
if st.session_state.documents and not st.session_state.index_ready:
    try:
        st.session_state.query_engine = get_query_engine()
        st.session_state.index_ready = True
    except Exception:
        pass  # will prompt user to upload

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📄 Document RAG")
    st.caption("Powered by LlamaIndex + LlamaParse + Gemini")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="LlamaParse will extract tables and text as Markdown.",
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file:
        if st.button("⚙️ Process Document", use_container_width=True):
            # Save uploaded PDF to a temp file
            tmp_path = Path("/tmp") / uploaded_file.name
            tmp_path.write_bytes(uploaded_file.getvalue())
            doc_name = tmp_path.stem

            with st.spinner("🔍 Parsing PDF with LlamaParse..."):
                try:
                    progress = st.progress(0, text="Starting LlamaParse...")
                    progress.progress(10, text="Parsing PDF to Markdown...")

                    index = run_ingestion_pipeline(tmp_path, doc_name=doc_name)
                    progress.progress(70, text="Storing embeddings in Neon...")

                    st.session_state.query_engine = get_query_engine()
                    st.session_state.index_ready = True
                    if doc_name not in st.session_state.documents:
                        st.session_state.documents.append(doc_name)
                    progress.progress(100, text="Done!")
                    st.success(f"✅ '{doc_name}' processed! Ask your questions below.")
                    logger.info(f"Document '{uploaded_file.name}' ingested successfully.")

                except ValueError as ve:
                    # Duplicate document
                    st.warning(f"⚠️ {ve}")
                    # Still allow querying the existing index
                    try:
                        st.session_state.query_engine = get_query_engine()
                        st.session_state.index_ready = True
                        st.info("Using previously ingested index.")
                    except Exception:
                        st.error("Could not connect to existing index.")
                    logger.warning(f"Duplicate upload: {ve}")

                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
                    logger.error(f"Ingestion error: {e}")

    # ── Ingested documents list ───────────────────────────────────────────────
    if st.session_state.index_ready:
        st.success("✅ Index ready")
    if st.session_state.documents:
        st.caption(f"📚 Ingested: {', '.join(st.session_state.documents)}")

    st.divider()

    # ── Clear / Reset buttons ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.index_ready = False
            st.session_state.query_engine = None
            st.session_state.documents = []
            st.session_state.uploader_key += 1  # reset file uploader
            st.rerun()

    st.divider()
    st.caption("llama-index-core · LlamaParse · Gemini · Neon pgvector")

# ── Main content ──────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">Document AI Q&A</p>', unsafe_allow_html=True)
st.caption("Ask questions grounded strictly in the uploaded documents. Tables are preserved via LlamaParse.")

if not st.session_state.index_ready:
    st.info("👈 Upload the PDF and click **Process Document** to begin.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 View source chunks"):
                for i, node in enumerate(msg["sources"], 1):
                    score = getattr(node, "score", None)
                    page = node.metadata.get("page_label", "?")
                    doc = node.metadata.get("document_name", "")
                    label_parts = [f"Excerpt {i}", f"Page {page}"]
                    if doc:
                        label_parts.append(doc)
                    if score is not None:
                        label_parts.append(f"Score {score:.3f}")
                    st.markdown(
                        f'<div class="context-box">'
                        f'<span class="badge">{" · ".join(label_parts)}'
                        f"</span><br>{node.text[:400]}...</div>",
                        unsafe_allow_html=True,
                    )

# ── Chat input ────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about the uploaded document(s)..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            try:
                engine = st.session_state.query_engine
                if engine is None:
                    raise RuntimeError("Query engine not initialised. Please process a document first.")

                response = engine.query(question)

                answer = str(response)
                sources = getattr(response, "source_nodes", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📄 View {len(sources)} source chunks"):
                        for i, node in enumerate(sources, 1):
                            score = getattr(node, "score", None)
                            page = node.metadata.get("page_label", "?")
                            doc = node.metadata.get("document_name", "")
                            label_parts = [f"Excerpt {i}", f"Page {page}"]
                            if doc:
                                label_parts.append(doc)
                            if score is not None:
                                label_parts.append(f"Score {score:.3f}")
                            st.markdown(
                                f'<div class="context-box">'
                                f'<span class="badge">{" · ".join(label_parts)}'
                                f"</span><br>{node.text[:400]}...</div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                err = f"Query failed: {e}"
                st.error(err)
                logger.error(err)
