"""
test_ingestion.py — Unit tests for the LlamaIndex ingestion pipeline.
LlamaParse and VectorStoreIndex are fully mocked — no external API calls.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestParsePdf:

    @patch("app.ingestion.pipeline.LlamaParse")
    def test_returns_list_of_documents(self, MockLlamaParse):
        """parse_pdf should return a list of LlamaIndex Document objects."""
        from app.ingestion.pipeline import parse_pdf

        mock_doc1 = MagicMock()
        mock_doc1.text = "# Revenue\n| Year | Amount |\n|---|---------|\n| FY24 | 12000 |"
        mock_doc2 = MagicMock()
        mock_doc2.text = "## Highlights\nGMV grew by 25%."

        mock_instance = MockLlamaParse.return_value
        mock_instance.load_data.return_value = [mock_doc1, mock_doc2]

        docs = parse_pdf("/tmp/test.pdf")

        assert len(docs) == 2
        mock_instance.load_data.assert_called_once_with("/tmp/test.pdf")

    @patch("app.ingestion.pipeline.LlamaParse")
    def test_llamaparse_uses_markdown_result_type(self, MockLlamaParse):
        """LlamaParse must be instantiated with result_type='markdown'."""
        from app.ingestion.pipeline import parse_pdf

        mock_instance = MockLlamaParse.return_value
        mock_instance.load_data.return_value = []

        parse_pdf("/tmp/report.pdf")

        MockLlamaParse.assert_called_once()
        call_kwargs = MockLlamaParse.call_args.kwargs
        assert call_kwargs.get("result_type") == "markdown", (
            "LlamaParse must use result_type='markdown' to preserve tables."
        )

    @patch("app.ingestion.pipeline.LlamaParse")
    def test_empty_pdf_returns_empty_list(self, MockLlamaParse):
        """parse_pdf should gracefully return empty list for a blank PDF."""
        from app.ingestion.pipeline import parse_pdf

        mock_instance = MockLlamaParse.return_value
        mock_instance.load_data.return_value = []

        docs = parse_pdf("/tmp/blank.pdf")
        assert docs == []

    @patch("app.ingestion.pipeline.LlamaParse")
    def test_parse_pdf_raises_runtime_error_on_failure(self, MockLlamaParse):
        """parse_pdf should wrap LlamaParse errors in RuntimeError."""
        from app.ingestion.pipeline import parse_pdf

        mock_instance = MockLlamaParse.return_value
        mock_instance.load_data.side_effect = Exception("API timeout")

        with pytest.raises(RuntimeError, match="PDF parsing failed"):
            parse_pdf("/tmp/bad.pdf")


class TestTagDocuments:

    def test_injects_document_name_metadata(self):
        """_tag_documents should add document_name to each doc's metadata."""
        from app.ingestion.pipeline import _tag_documents

        doc1 = MagicMock()
        doc1.metadata = {"page_label": "1"}
        doc2 = MagicMock()
        doc2.metadata = {}

        _tag_documents([doc1, doc2], "Annual-Report")

        assert doc1.metadata["document_name"] == "Annual-Report"
        assert doc2.metadata["document_name"] == "Annual-Report"

    def test_handles_none_metadata(self):
        """_tag_documents should handle docs with metadata=None."""
        from app.ingestion.pipeline import _tag_documents

        doc = MagicMock()
        doc.metadata = None

        _tag_documents([doc], "my-doc")

        assert doc.metadata["document_name"] == "my-doc"


class TestRunIngestionPipeline:

    @patch("app.ingestion.pipeline._store_hash_in_db")
    @patch("app.ingestion.pipeline._hash_exists_in_db", return_value=False)
    @patch("app.ingestion.pipeline.VectorStoreIndex")
    @patch("app.ingestion.pipeline.StorageContext")
    @patch("app.ingestion.pipeline._get_vector_store")
    @patch("app.ingestion.pipeline.parse_pdf")
    @patch("app.ingestion.pipeline.configure_llama_settings")
    @patch("app.ingestion.pipeline.compute_file_hash", return_value="abc123")
    def test_pipeline_calls_steps_in_order(
        self,
        mock_hash,
        mock_configure,
        mock_parse,
        mock_get_store,
        mock_storage_ctx,
        MockIndex,
        mock_hash_exists,
        mock_store_hash,
    ):
        """run_ingestion_pipeline must call configure -> parse -> index in order."""
        from app.ingestion.pipeline import run_ingestion_pipeline

        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_parse.return_value = [mock_doc]
        mock_get_store.return_value = MagicMock()
        mock_storage_ctx.from_defaults.return_value = MagicMock()
        MockIndex.from_documents.return_value = MagicMock()

        run_ingestion_pipeline("/tmp/test.pdf")

        mock_configure.assert_called_once()
        mock_parse.assert_called_once()
        MockIndex.from_documents.assert_called_once()
        mock_store_hash.assert_called_once_with("abc123", "test")

    @patch("app.ingestion.pipeline._store_hash_in_db")
    @patch("app.ingestion.pipeline._hash_exists_in_db", return_value=False)
    @patch("app.ingestion.pipeline.VectorStoreIndex")
    @patch("app.ingestion.pipeline.StorageContext")
    @patch("app.ingestion.pipeline._get_vector_store")
    @patch("app.ingestion.pipeline.parse_pdf")
    @patch("app.ingestion.pipeline.configure_llama_settings")
    @patch("app.ingestion.pipeline.compute_file_hash", return_value="def456")
    def test_pipeline_passes_documents_to_index(
        self,
        mock_hash,
        mock_configure,
        mock_parse,
        mock_get_store,
        mock_storage_ctx,
        MockIndex,
        mock_hash_exists,
        mock_store_hash,
    ):
        """Documents from parse_pdf must be passed to VectorStoreIndex."""
        from app.ingestion.pipeline import run_ingestion_pipeline

        mock_docs = [MagicMock(), MagicMock()]
        for d in mock_docs:
            d.metadata = {}
        mock_parse.return_value = mock_docs
        mock_get_store.return_value = MagicMock()
        mock_storage_ctx.from_defaults.return_value = MagicMock()
        MockIndex.from_documents.return_value = MagicMock()

        run_ingestion_pipeline("/tmp/test.pdf")

        call_args = MockIndex.from_documents.call_args
        assert call_args[0][0] == mock_docs

    @patch("app.ingestion.pipeline._store_hash_in_db")
    @patch("app.ingestion.pipeline._hash_exists_in_db", return_value=False)
    @patch("app.ingestion.pipeline.VectorStoreIndex")
    @patch("app.ingestion.pipeline.StorageContext")
    @patch("app.ingestion.pipeline._get_vector_store")
    @patch("app.ingestion.pipeline.parse_pdf")
    @patch("app.ingestion.pipeline.configure_llama_settings")
    @patch("app.ingestion.pipeline.compute_file_hash", return_value="meta01")
    def test_pipeline_tags_documents_with_metadata(
        self,
        mock_hash,
        mock_configure,
        mock_parse,
        mock_get_store,
        mock_storage_ctx,
        MockIndex,
        mock_hash_exists,
        mock_store_hash,
    ):
        """run_ingestion_pipeline must tag every chunk with document_name metadata."""
        from app.ingestion.pipeline import run_ingestion_pipeline

        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_parse.return_value = [mock_doc]
        mock_get_store.return_value = MagicMock()
        mock_storage_ctx.from_defaults.return_value = MagicMock()
        MockIndex.from_documents.return_value = MagicMock()

        run_ingestion_pipeline("/tmp/my-report.pdf", doc_name="my-report")

        assert mock_doc.metadata["document_name"] == "my-report"

    @patch("app.ingestion.pipeline._hash_exists_in_db", return_value=True)
    @patch("app.ingestion.pipeline.parse_pdf")
    @patch("app.ingestion.pipeline.configure_llama_settings")
    @patch("app.ingestion.pipeline.compute_file_hash", return_value="dup789")
    def test_pipeline_raises_on_duplicate_document(
        self,
        mock_hash,
        mock_configure,
        mock_parse,
        mock_hash_exists,
    ):
        """run_ingestion_pipeline must raise ValueError for duplicate documents."""
        from app.ingestion.pipeline import run_ingestion_pipeline

        with pytest.raises(ValueError, match="already been processed"):
            run_ingestion_pipeline("/tmp/duplicate.pdf")

    @patch("app.ingestion.pipeline._hash_exists_in_db", return_value=False)
    @patch("app.ingestion.pipeline.parse_pdf", return_value=[])
    @patch("app.ingestion.pipeline.configure_llama_settings")
    @patch("app.ingestion.pipeline.compute_file_hash", return_value="empty000")
    def test_pipeline_raises_on_empty_parse(
        self,
        mock_hash,
        mock_configure,
        mock_parse,
        mock_hash_exists,
    ):
        """run_ingestion_pipeline must raise RuntimeError if parse returns nothing."""
        from app.ingestion.pipeline import run_ingestion_pipeline

        with pytest.raises(RuntimeError, match="no extractable content"):
            run_ingestion_pipeline("/tmp/empty.pdf")

    @patch("app.ingestion.pipeline._store_hash_in_db")
    @patch("app.ingestion.pipeline._hash_exists_in_db", return_value=False)
    @patch("app.ingestion.pipeline.VectorStoreIndex")
    @patch("app.ingestion.pipeline.StorageContext")
    @patch("app.ingestion.pipeline._get_vector_store")
    @patch("app.ingestion.pipeline.parse_pdf")
    @patch("app.ingestion.pipeline.configure_llama_settings")
    @patch("app.ingestion.pipeline.compute_file_hash", return_value="persist01")
    def test_pipeline_persists_hash_to_db(
        self,
        mock_hash,
        mock_configure,
        mock_parse,
        mock_get_store,
        mock_storage_ctx,
        MockIndex,
        mock_hash_exists,
        mock_store_hash,
    ):
        """run_ingestion_pipeline must persist the file hash in the DB after success."""
        from app.ingestion.pipeline import run_ingestion_pipeline

        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_parse.return_value = [mock_doc]
        mock_get_store.return_value = MagicMock()
        mock_storage_ctx.from_defaults.return_value = MagicMock()
        MockIndex.from_documents.return_value = MagicMock()

        run_ingestion_pipeline("/tmp/report.pdf", doc_name="report")

        mock_store_hash.assert_called_once_with("persist01", "report")

