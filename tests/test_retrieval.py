"""
test_retrieval.py — Unit tests for the LlamaIndex query engine.
VectorStoreIndex and Gemini are fully mocked — no external API calls.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestGetQueryEngine:

    @patch("app.retrieval.query_engine.VectorStoreIndex")
    @patch("app.retrieval.query_engine.PGVectorStore")
    @patch("app.retrieval.query_engine.configure_llama_settings")
    def test_returns_query_engine(self, mock_configure, MockPG, MockIndex):
        """get_query_engine should return an object (the query engine)."""
        from app.retrieval.query_engine import get_query_engine

        mock_engine = MagicMock()
        MockIndex.from_vector_store.return_value.as_query_engine.return_value = mock_engine

        engine = get_query_engine()
        assert engine is mock_engine

    @patch("app.retrieval.query_engine.VectorStoreIndex")
    @patch("app.retrieval.query_engine.PGVectorStore")
    @patch("app.retrieval.query_engine.configure_llama_settings")
    def test_index_loaded_from_vector_store(self, mock_configure, MockPG, MockIndex):
        """Must load the index from the existing vector store, not rebuild it."""
        from app.retrieval.query_engine import get_query_engine

        get_query_engine()
        MockIndex.from_vector_store.assert_called_once()

    @patch("app.retrieval.query_engine.VectorStoreIndex")
    @patch("app.retrieval.query_engine.PGVectorStore")
    @patch("app.retrieval.query_engine.configure_llama_settings")
    def test_pg_vector_store_from_params_called(self, mock_configure, MockPG, MockIndex):
        """PGVectorStore.from_params must be called to connect to Neon."""
        from app.retrieval.query_engine import get_query_engine

        get_query_engine()
        MockPG.from_params.assert_called_once()

    @patch("app.retrieval.query_engine.VectorStoreIndex")
    @patch("app.retrieval.query_engine.PGVectorStore")
    @patch("app.retrieval.query_engine.configure_llama_settings")
    def test_pg_vector_store_receives_async_connection_string(self, mock_configure, MockPG, MockIndex):
        """PGVectorStore.from_params must receive async_connection_string."""
        from app.retrieval.query_engine import get_query_engine

        get_query_engine()

        call_kwargs = MockPG.from_params.call_args.kwargs
        assert "async_connection_string" in call_kwargs, (
            "Must supply async_connection_string for asyncpg support."
        )

    @patch("app.retrieval.query_engine.VectorStoreIndex")
    @patch("app.retrieval.query_engine.PGVectorStore")
    @patch("app.retrieval.query_engine.configure_llama_settings")
    def test_as_query_engine_uses_top_k(self, mock_configure, MockPG, MockIndex):
        """QueryEngine must be configured with the TOP_K_CHUNKS setting."""
        from app.retrieval.query_engine import get_query_engine
        from app.config import TOP_K_CHUNKS

        mock_index = MagicMock()
        MockIndex.from_vector_store.return_value = mock_index

        get_query_engine()

        call_kwargs = mock_index.as_query_engine.call_args.kwargs
        assert call_kwargs.get("similarity_top_k") == TOP_K_CHUNKS

    @patch("app.retrieval.query_engine.VectorStoreIndex")
    @patch("app.retrieval.query_engine.PGVectorStore")
    @patch("app.retrieval.query_engine.configure_llama_settings")
    def test_configure_llama_settings_called(self, mock_configure, MockPG, MockIndex):
        """get_query_engine must call configure_llama_settings."""
        from app.retrieval.query_engine import get_query_engine

        get_query_engine()
        mock_configure.assert_called_once()

    @patch("app.retrieval.query_engine.PGVectorStore")
    @patch("app.retrieval.query_engine.configure_llama_settings")
    def test_raises_runtime_error_on_failure(self, mock_configure, MockPG):
        """get_query_engine must raise RuntimeError if connection fails."""
        from app.retrieval.query_engine import get_query_engine

        MockPG.from_params.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError, match="Query engine initialisation failed"):
            get_query_engine()


class TestQaPromptTemplate:

    def test_prompt_contains_strict_rules(self):
        """The QA prompt must contain hallucination prevention rules."""
        from app.retrieval.query_engine import _QA_PROMPT_TEMPLATE

        template_str = _QA_PROMPT_TEMPLATE.template
        assert "ONLY" in template_str
        assert "not available in the provided document" in template_str

    def test_prompt_contains_context_and_query_vars(self):
        """The QA prompt must reference {context_str} and {query_str}."""
        from app.retrieval.query_engine import _QA_PROMPT_TEMPLATE

        template_str = _QA_PROMPT_TEMPLATE.template
        assert "{context_str}" in template_str
        assert "{query_str}" in template_str

    def test_prompt_instructs_table_reading(self):
        """The QA prompt must specifically instruct on reading Markdown tables."""
        from app.retrieval.query_engine import _QA_PROMPT_TEMPLATE

        template_str = _QA_PROMPT_TEMPLATE.template
        assert "Markdown" in template_str or "tables" in template_str.lower()

    def test_prompt_requires_page_citations(self):
        """The QA prompt must require page number citations."""
        from app.retrieval.query_engine import _QA_PROMPT_TEMPLATE

        template_str = _QA_PROMPT_TEMPLATE.template
        assert "Page" in template_str
        assert "citation" in template_str.lower() or "cite" in template_str.lower()

    def test_prompt_mentions_document_name_metadata(self):
        """The QA prompt should instruct the LLM to use document_name metadata."""
        from app.retrieval.query_engine import _QA_PROMPT_TEMPLATE

        template_str = _QA_PROMPT_TEMPLATE.template
        assert "document_name" in template_str

    def test_prompt_is_generic_not_hardcoded(self):
        """The QA prompt must NOT contain hardcoded document names."""
        from app.retrieval.query_engine import _QA_PROMPT_TEMPLATE

        template_str = _QA_PROMPT_TEMPLATE.template
        assert "Swiggy" not in template_str
        assert "swiggy" not in template_str.lower()
