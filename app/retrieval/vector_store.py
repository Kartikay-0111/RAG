"""
vector_store.py - Manages the Neon (PostgreSQL + pgvector) vector database.
"""

import psycopg2
from psycopg2.extras import execute_values
from typing import List, Tuple

from app.config import NEON_DATABASE_URL, EMBEDDING_DIMENSION
from app.ingestion.chunker import TextChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_connection():
    """Return a psycopg2 connection to Neon."""
    return psycopg2.connect(NEON_DATABASE_URL)


def init_schema() -> None:
    """
    Create the pgvector extension and document_chunks table if they don't exist.
    If the table exists but with a different vector dimension, it drops and recreates it.
    """
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                # 1. Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # 2. Check existing dimension if table exists
                cur.execute("""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    WHERE attrelid = 'document_chunks'::regclass 
                      AND attname = 'embedding';
                """)
                row = cur.fetchone()
                
                if row:
                    current_dim = row[0]
                    if current_dim != EMBEDDING_DIMENSION:
                        logger.warning(
                            f"Dimension mismatch: DB has {current_dim}, config has {EMBEDDING_DIMENSION}. "
                            "Recreating table..."
                        )
                        cur.execute("DROP TABLE document_chunks CASCADE;")
                    else:
                        logger.info(f"Database schema verified (dimension: {current_dim}).")
                        return

                # 3. Create/Recreate the chunks table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id            SERIAL PRIMARY KEY,
                        document_id   TEXT        NOT NULL,
                        page_number   INT         NOT NULL,
                        chunk_index   INT         NOT NULL,
                        content       TEXT        NOT NULL,
                        embedding     vector({EMBEDDING_DIMENSION}),
                        created_at    TIMESTAMP   DEFAULT NOW()
                    );
                """)

                # 4. IVFFlat index for fast cosine similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                    ON document_chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
        logger.info("Database schema initialised successfully.")
    except psycopg2.Error as e:
        # Handle case where table doesn't exist yet for the SELECT check
        if "does not exist" in str(e):
             conn.rollback() 
             _force_create_schema()
        else:
            raise e
    finally:
        conn.close()


def _force_create_schema() -> None:
    """Helper to create schema when table doesn't exist yet."""
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id            SERIAL PRIMARY KEY,
                        document_id   TEXT        NOT NULL,
                        page_number   INT         NOT NULL,
                        chunk_index   INT         NOT NULL,
                        content       TEXT        NOT NULL,
                        embedding     vector({EMBEDDING_DIMENSION}),
                        created_at    TIMESTAMP   DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                    ON document_chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
        logger.info("Database schema created.")
    finally:
        conn.close()


def clear_document(document_id: str) -> int:
    """Delete all chunks belonging to a document_id."""
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s;",
                    (document_id,),
                )
                deleted = cur.rowcount
        logger.info(f"Cleared {deleted} existing chunks for document '{document_id}'.")
        return deleted
    finally:
        conn.close()


def upsert_chunks(
    embedded_chunks: List[Tuple[TextChunk, List[float]]],
) -> int:
    """Insert (chunk, embedding) pairs into Neon."""
    if not embedded_chunks:
        logger.warning("No chunks to upsert.")
        return 0

    rows = [
        (
            chunk.document_id,
            chunk.page_number,
            chunk.chunk_index,
            chunk.content,
            embedding,
        )
        for chunk, embedding in embedded_chunks
    ]

    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO document_chunks
                        (document_id, page_number, chunk_index, content, embedding)
                    VALUES %s;
                    """,
                    rows,
                    template="(%s, %s, %s, %s, %s::vector)",
                )
        logger.info(f"Inserted {len(rows)} chunks into Neon.")
        return len(rows)
    finally:
        conn.close()
