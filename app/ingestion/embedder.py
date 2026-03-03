"""
embedder.py - Generates embeddings for text chunks using Gemini Embeddings API.

Uses: google.genai SDK (latest)
Model: embedding-001 with output_dimensionality=768 (truncated from 3072)
"""

import time
from typing import List, Tuple

from google import genai
from google.genai import types as genai_types

from app.config import GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL, EMBEDDING_DIMENSION
from app.ingestion.chunker import TextChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_client() -> genai.Client:
    """Return a configured Gemini API client."""
    return genai.Client(api_key=GOOGLE_API_KEY)


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Generate an embedding vector for a single text string.

    Uses Gemini's Elastic Embedding feature (output_dimensionality) to truncate
    the 3072-dim embedding-001 output to EMBEDDING_DIMENSION (768) so it fits
    within the pgvector ivfflat index limit of 2000 dimensions.

    Args:
        text: The text to embed.
        task_type: 'RETRIEVAL_DOCUMENT' for storage, 'RETRIEVAL_QUERY' for queries.

    Returns:
        List of floats (EMBEDDING_DIMENSION-dimensional vector).
    """
    client = _get_client()
    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSION,  # truncate 3072 → 768
        ),
    )
    return response.embeddings[0].values


def embed_chunks(
    chunks: List[TextChunk],
    batch_delay_sec: float = 0.5,
) -> List[Tuple[TextChunk, List[float]]]:
    """
    Generate embeddings for a list of TextChunk objects.
    """
    results: List[Tuple[TextChunk, List[float]]] = []
    logger.info(f"Embedding {len(chunks)} chunks via Gemini API (dim={EMBEDDING_DIMENSION})...")

    for idx, chunk in enumerate(chunks):
        try:
            embedding = embed_text(chunk.content, task_type="RETRIEVAL_DOCUMENT")
            results.append((chunk, embedding))

            if (idx + 1) % 10 == 0:
                logger.info(f"  Embedded {idx + 1}/{len(chunks)} chunks...")

            time.sleep(batch_delay_sec)

        except Exception as exc:
            logger.error(
                f"Failed to embed chunk {chunk.chunk_index} "
                f"(page {chunk.page_number}): {exc}"
            )
            continue

    logger.info(f"Embedding complete: {len(results)}/{len(chunks)} chunks embedded.")
    return results
