"""
pipeline.py — Ingestion pipeline using LlamaParse + LlamaIndex.

Flow:
  1. LlamaParse parses the PDF to Markdown (preserves financial tables)
  2. Centralized Settings for Gemini LLM + Embeddings (via app.llm)
  3. Inject ``document_name`` metadata on every document/chunk
  4. PGVectorStore connects to Neon PostgreSQL (sync + async)
  5. VectorStoreIndex.from_documents() chunks, embeds, and upserts
  6. Document hash persisted in DB to prevent duplicate ingestion (survives restarts)
"""

from pathlib import Path
from typing import List

import psycopg2
from llama_index.readers.llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.postgres import PGVectorStore

from app.config import (
    NEON_DATABASE_URL,
    LLAMA_CLOUD_API_KEY,
    EMBEDDING_DIMENSION,
    VECTOR_TABLE_NAME,
    HASH_TABLE_NAME,
    get_async_connection_string,
    compute_file_hash,
)
from app.llm import configure_llama_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Step 1: Parse PDF with LlamaParse ─────────────────────────────────────────

def parse_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Parse a PDF using LlamaParse and return LlamaIndex Document objects.

    LlamaParse extracts content as Markdown, preserving tables as
    structured text (e.g., ``| Revenue | ₹12,000 |``), which is far
    more accurate than OCR for financial reports.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of LlamaIndex Document objects (one per page/section).

    Raises:
        RuntimeError: If LlamaParse fails to process the file.
    """
    logger.info(f"Parsing PDF with LlamaParse: {pdf_path}")
    try:
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",       # Preserve tables as Markdown
            verbose=False,
            language="en",
        )
        documents = parser.load_data(str(pdf_path))
    except Exception as exc:
        logger.error(f"LlamaParse failed: {exc}")
        raise RuntimeError(f"PDF parsing failed: {exc}") from exc

    logger.info(f"LlamaParse returned {len(documents)} document sections.")
    return documents


# ── Step 2: Inject document-level metadata ────────────────────────────────────

def _tag_documents(documents: List[Document], doc_name: str) -> None:
    """
    Add ``document_name`` to the metadata of every Document in-place.

    This allows later filtering / display — e.g. showing which document
    a retrieved chunk belongs to, or scoping retrieval to one file when
    multiple PDFs have been ingested into the same table.
    """
    for doc in documents:
        doc.metadata = doc.metadata or {}
        doc.metadata["document_name"] = doc_name


# ── Step 3: DB-persisted deduplication ────────────────────────────────────────

def _ensure_hash_table() -> None:
    """Create the hash tracking table if it doesn't exist."""
    with psycopg2.connect(NEON_DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {HASH_TABLE_NAME} (
                    file_hash  TEXT PRIMARY KEY,
                    doc_name   TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()


def _hash_exists_in_db(file_hash: str) -> bool:
    """Check whether a file hash already exists in the DB."""
    _ensure_hash_table()
    with psycopg2.connect(NEON_DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {HASH_TABLE_NAME} WHERE file_hash = %s LIMIT 1",
                (file_hash,),
            )
            return cur.fetchone() is not None


def _store_hash_in_db(file_hash: str, doc_name: str) -> None:
    """Persist a file hash + doc name in the DB."""
    with psycopg2.connect(NEON_DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {HASH_TABLE_NAME} (file_hash, doc_name) "
                f"VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (file_hash, doc_name),
            )
        conn.commit()


def get_ingested_doc_names() -> List[str]:
    """
    Return the list of document names already ingested (from the DB).

    Useful for populating the UI sidebar after a restart.
    """
    _ensure_hash_table()
    with psycopg2.connect(NEON_DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT doc_name FROM {HASH_TABLE_NAME} ORDER BY doc_name"
            )
            return [row[0] for row in cur.fetchall()]


# ── Step 4: Vector store factory ──────────────────────────────────────────────

def _get_vector_store() -> PGVectorStore:
    """
    Create a PGVectorStore pointing to the Neon database.

    Supplies **both** the sync (psycopg2) and async (asyncpg) connection
    strings so that LlamaIndex can use either execution mode.
    """
    return PGVectorStore.from_params(
        connection_string=NEON_DATABASE_URL,
        async_connection_string=get_async_connection_string(),
        table_name=VECTOR_TABLE_NAME,
        embed_dim=EMBEDDING_DIMENSION,
    )


# ── Step 5: Full ingestion pipeline ──────────────────────────────────────────

def run_ingestion_pipeline(
    pdf_path: str | Path,
    doc_name: str | None = None,
) -> VectorStoreIndex:
    """
    Full ingestion pipeline: configure → parse → tag → embed → store.

    1. Configure LlamaIndex globally (Gemini LLM + Embeddings) — idempotent.
    2. Compute a SHA-256 hash of the PDF and skip if already ingested.
    3. Parse PDF to Markdown Documents via LlamaParse.
    4. Tag every document with ``document_name`` metadata.
    5. Connect to the PGVectorStore on Neon (sync + async).
    6. Build a VectorStoreIndex from the documents
       (LlamaIndex handles chunking, embedding, and upsert automatically).

    Args:
        pdf_path: Path to the PDF file to ingest.
        doc_name: Human-readable name stored in chunk metadata.
                  Defaults to the PDF filename.

    Returns:
        A VectorStoreIndex backed by the Neon PGVectorStore.

    Raises:
        RuntimeError: On parse or indexing failure.
        ValueError: If the document has already been ingested.
    """
    pdf_path = Path(pdf_path)
    if doc_name is None:
        doc_name = pdf_path.stem  # e.g. "Annual-Report-FY-2023-24"

    configure_llama_settings()

    # ── Deduplication check (DB-persisted) ──
    file_hash = compute_file_hash(pdf_path)
    if _hash_exists_in_db(file_hash):
        logger.warning("Document already ingested (hash found in DB). Skipping.")
        raise ValueError(
            "This document has already been processed. "
            "Upload a different file or clear the index first."
        )

    # ── Parse ──
    documents = parse_pdf(pdf_path)
    if not documents:
        logger.warning("LlamaParse returned 0 documents — nothing to index.")
        raise RuntimeError("PDF produced no extractable content.")

    # ── Tag metadata ──
    _tag_documents(documents, doc_name)

    # ── Index ──
    try:
        vector_store = _get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info(
            f"Building VectorStoreIndex from {len(documents)} sections "
            f"(doc='{doc_name}')..."
        )
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
    except Exception as exc:
        logger.error(f"Indexing failed: {exc}")
        raise RuntimeError(f"Failed to build vector index: {exc}") from exc

    _store_hash_in_db(file_hash, doc_name)
    logger.info(f"Ingestion complete for '{doc_name}'. Index stored in Neon.")
    return index