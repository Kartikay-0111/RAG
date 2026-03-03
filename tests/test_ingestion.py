"""
test_ingestion.py - Unit tests for the ingestion pipeline modules.
External APIs (Gemini) are mocked so tests run fully offline.
"""

import pytest
from unittest.mock import patch, MagicMock

# ── cleaner tests ─────────────────────────────────────────────────────────────

from app.ingestion.cleaner import clean_text, clean_pages


class TestCleanText:

    def test_removes_extra_whitespace(self):
        raw = "Hello    World   foo"
        assert "  " not in clean_text(raw)

    def test_preserves_numbers_and_currency(self):
        raw = "Revenue: ₹12,345.67 crores  (FY2024)"
        cleaned = clean_text(raw)
        assert "12,345.67" in cleaned
        assert "FY2024" in cleaned

    def test_collapses_excessive_blank_lines(self):
        raw = "Line1\n\n\n\n\nLine2"
        cleaned = clean_text(raw)
        assert "\n\n\n" not in cleaned

    def test_empty_input_returns_empty(self):
        assert clean_text("") == ""
        assert clean_text("   \n  \t  ") == ""

    def test_unicode_dashes_normalized(self):
        raw = "Revenue – 100"
        cleaned = clean_text(raw)
        assert "–" not in cleaned
        assert "-" in cleaned


class TestCleanPages:

    def test_skips_empty_pages(self):
        pages = [(1, "Hello world"), (2, "   \n  "), (3, "More text")]
        result = clean_pages(pages)
        page_nums = [p for p, _ in result]
        assert 2 not in page_nums
        assert 1 in page_nums
        assert 3 in page_nums

    def test_preserves_page_number(self):
        pages = [(42, "Some important text here")]
        result = clean_pages(pages)
        assert result[0][0] == 42


# ── chunker tests ─────────────────────────────────────────────────────────────

from app.ingestion.chunker import chunk_pages, TextChunk


class TestChunkPages:

    def _make_page(self, n_chars: int = 1200) -> str:
        return "A" * n_chars

    def test_basic_chunking_produces_multiple_chunks(self):
        pages = [(1, self._make_page(1200))]
        chunks = chunk_pages(pages, document_id="test_doc", chunk_size=500, overlap=100)
        assert len(chunks) > 1

    def test_chunk_max_length_respected(self):
        pages = [(1, self._make_page(2000))]
        chunks = chunk_pages(pages, document_id="test_doc", chunk_size=500, overlap=100)
        for chunk in chunks:
            assert len(chunk.content) <= 500

    def test_metadata_attached(self):
        pages = [(7, "Some text on page seven for metadata testing purposes here")]
        chunks = chunk_pages(pages, document_id="my_doc", chunk_size=500, overlap=100)
        assert chunks[0].document_id == "my_doc"
        assert chunks[0].page_number == 7

    def test_overlap_means_consecutive_chunks_share_content(self):
        text = "ABCDEFGHIJ" * 60  # 600 chars
        pages = [(1, text)]
        chunks = chunk_pages(pages, document_id="doc", chunk_size=100, overlap=20)
        if len(chunks) >= 2:
            end_of_first = chunks[0].content[-20:]
            start_of_second = chunks[1].content[:20]
            assert any(c in start_of_second for c in end_of_first)

    def test_empty_pages_ignored(self):
        pages = [(1, "   "), (2, "Real content here for page two")]
        chunks = chunk_pages(pages, document_id="doc")
        page_nums = {c.page_number for c in chunks}
        assert 1 not in page_nums
        assert 2 in page_nums

    def test_chunk_index_monotonically_increases(self):
        pages = [(1, "A" * 2000)]
        chunks = chunk_pages(pages, document_id="doc", chunk_size=300, overlap=50)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))


# ── embedder tests (mocked) ───────────────────────────────────────────────────

from app.ingestion.embedder import embed_text, embed_chunks


def _make_embedding_response(values):
    """Build a mock response matching google.genai embed_content return shape."""
    mock_embedding = MagicMock()
    mock_embedding.values = values
    mock_resp = MagicMock()
    mock_resp.embeddings = [mock_embedding]
    return mock_resp


class TestEmbedder:

    @patch("app.ingestion.embedder.genai.Client")
    def test_embed_text_calls_gemini_api(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.models.embed_content.return_value = _make_embedding_response([0.1] * 768)
        result = embed_text("hello world")
        assert len(result) == 768
        mock_instance.models.embed_content.assert_called_once()

    @patch("app.ingestion.embedder.genai.Client")
    def test_embed_chunks_skips_failed(self, MockClient):
        """A single failed embedding should not crash the whole batch."""
        mock_instance = MockClient.return_value
        mock_instance.models.embed_content.side_effect = [
            _make_embedding_response([0.0] * 768),
            Exception("API error"),
            _make_embedding_response([0.5] * 768),
        ]
        chunks = [
            TextChunk("doc", 1, 0, "chunk one text", 0),
            TextChunk("doc", 1, 1, "chunk two text", 500),
            TextChunk("doc", 2, 2, "chunk three text", 0),
        ]
        results = embed_chunks(chunks, batch_delay_sec=0)
        assert len(results) == 2

    @patch("app.ingestion.embedder.genai.Client")
    def test_embed_chunks_returns_correct_types(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.models.embed_content.return_value = _make_embedding_response([0.1] * 768)
        chunks = [TextChunk("doc", 1, 0, "some text here", 0)]
        results = embed_chunks(chunks, batch_delay_sec=0)
        chunk, emb = results[0]
        assert isinstance(chunk, TextChunk)
        assert isinstance(emb, list)
        assert len(emb) == 768
