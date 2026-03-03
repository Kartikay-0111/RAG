"""
similarity_search.py - Performs semantic similarity search against Neon pgvector.
"""

import psycopg2
from typing import List, Dict, Any

from app.config import NEON_DATABASE_URL, TOP_K_CHUNKS
from app.ingestion.embedder import embed_text
from app.utils.logger import get_logger

logger = get_logger(__name__)


def search(
    query: str,
    document_id: str | None = None,
    top_k: int = TOP_K_CHUNKS,
) -> List[Dict[str, Any]]:
    """
    Embed a query and retrieve the top-K most semantically similar chunks.
    """
    logger.info(f"Embedding query for similarity search...")
    query_embedding = embed_text(query, task_type="RETRIEVAL_QUERY")

    vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    if document_id:
        sql = """
            SELECT content, page_number, document_id,
                   (embedding <=> %s::vector) AS score
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vector_str, document_id, vector_str, top_k)
    else:
        sql = """
            SELECT content, page_number, document_id,
                   (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vector_str, vector_str, top_k)

    conn = psycopg2.connect(NEON_DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = [
            {
                "content": row[0],
                "page_number": row[1],
                "document_id": row[2],
                "score": float(row[3]),
            }
            for row in rows
        ]
        logger.info(f"Similarity search returned {len(results)} chunks.")
        return results

    finally:
        conn.close()
