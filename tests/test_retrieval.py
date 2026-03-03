"""
test_retrieval.py - Unit tests for retrieval modules.
Neon DB and Gemini API are fully mocked — no external calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.retrieval.similarity_search import search


class TestSimilaritySearch:

    @patch("app.retrieval.similarity_search.psycopg2.connect")
    @patch("app.retrieval.similarity_search.embed_text")
    def test_returns_list_of_dicts(self, mock_embed, mock_connect):
        mock_embed.return_value = [0.1] * 768
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("Some content text here", 5, "doc123", 0.12),
            ("More financial data", 10, "doc123", 0.25),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        results = search("What is revenue?", document_id="doc123", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 2
        assert "content" in results[0]
        assert "page_number" in results[0]
        assert "score" in results[0]

    @patch("app.retrieval.similarity_search.psycopg2.connect")
    @patch("app.retrieval.similarity_search.embed_text")
    def test_query_embedding_uses_retrieval_query_task(self, mock_embed, mock_connect):
        mock_embed.return_value = [0.0] * 768
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        search("test query")
        mock_embed.assert_called_once_with("test query", task_type="RETRIEVAL_QUERY")

    @patch("app.retrieval.similarity_search.psycopg2.connect")
    @patch("app.retrieval.similarity_search.embed_text")
    def test_empty_results_returns_empty_list(self, mock_embed, mock_connect):
        mock_embed.return_value = [0.0] * 768
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        results = search("nonexistent topic")
        assert results == []


from app.retrieval.prompt_builder import build_prompt


class TestPromptBuilder:

    def test_prompt_contains_question(self):
        question = "What is Swiggy's net profit?"
        prompt = build_prompt(question, [])
        assert question in prompt

    def test_prompt_contains_strict_rules(self):
        prompt = build_prompt("test?", [])
        assert "not available in the document" in prompt
        assert "Do NOT" in prompt

    def test_prompt_contains_context_with_page_numbers(self):
        chunks = [
            {"content": "Revenue was ₹12,000 crores.", "page_number": 42},
            {"content": "EBITDA increased by 15%.", "page_number": 55},
        ]
        prompt = build_prompt("What is revenue?", chunks)
        assert "Page 42" in prompt
        assert "Page 55" in prompt
        assert "₹12,000 crores" in prompt

    def test_empty_context_handled_gracefully(self):
        prompt = build_prompt("Any question?", [])
        assert "No relevant context found" in prompt

    def test_multiple_excerpts_are_numbered(self):
        chunks = [
            {"content": "Text A", "page_number": 1},
            {"content": "Text B", "page_number": 2},
            {"content": "Text C", "page_number": 3},
        ]
        prompt = build_prompt("question", chunks)
        assert "Excerpt 1" in prompt
        assert "Excerpt 2" in prompt
        assert "Excerpt 3" in prompt
