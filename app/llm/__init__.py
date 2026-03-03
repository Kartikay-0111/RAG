"""
llm/__init__.py — Centralized LlamaIndex Settings configuration.

Provides a single `configure_llama_settings()` entry point that sets the
global LLM, embedding model, and node parser for the entire application.
Call once at startup; subsequent calls are safe no-ops.
"""

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from app.config import (
    GOOGLE_API_KEY,
    GEMINI_LLM_MODEL,
    GEMINI_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    LLM_TEMPERATURE,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_configured = False


def configure_llama_settings() -> None:
    """
    Configure Gemini LLM + Embedding + SentenceSplitter globally.

    Sets ``Settings.llm``, ``Settings.embed_model`` and
    ``Settings.node_parser`` so every LlamaIndex component picks them
    up automatically.  Safe to call multiple times — only the first
    invocation has any effect.
    """
    global _configured
    if _configured:
        return

    Settings.llm = GoogleGenAI(
        model=GEMINI_LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
    Settings.embed_model = GeminiEmbedding(
        model_name=GEMINI_EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY,
        title="document_embedding",
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )

    _configured = True
    logger.info(
        f"LlamaIndex Settings configured: LLM={GEMINI_LLM_MODEL}, "
        f"Embedding={GEMINI_EMBEDDING_MODEL} (dim={EMBEDDING_DIMENSION})"
    )


def reset_settings() -> None:
    """Reset the configured flag (used in tests)."""
    global _configured
    _configured = False
