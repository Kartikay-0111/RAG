"""
chunker.py - Splits cleaned text into overlapping character-level chunks.

Strategy:
- Chunk size: 500 characters
- Overlap: 100 characters
- Each chunk carries document_id and page_number metadata
"""

from dataclasses import dataclass
from typing import List, Tuple

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A single text chunk with associated metadata."""
    document_id: str
    page_number: int
    chunk_index: int          # 0-based index within the document
    content: str
    char_start: int           # character offset within the page text


def chunk_pages(
    pages: List[Tuple[int, str]],
    document_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[TextChunk]:
    """
    Split a list of (page_number, text) tuples into overlapping TextChunks.
    """
    all_chunks: List[TextChunk] = []
    global_chunk_idx = 0

    for page_num, text in pages:
        if not text:
            continue

        step = max(chunk_size - overlap, 1)
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                all_chunks.append(
                    TextChunk(
                        document_id=document_id,
                        page_number=page_num,
                        chunk_index=global_chunk_idx,
                        content=chunk_text,
                        char_start=start,
                    )
                )
                global_chunk_idx += 1

            start += step

    logger.info(
        f"Chunking complete: {len(all_chunks)} chunks from "
        f"{len(pages)} pages (size={chunk_size}, overlap={overlap})."
    )
    return all_chunks
