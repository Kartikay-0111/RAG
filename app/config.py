"""
config.py — Central configuration for the LlamaIndex RAG system.

All settings loaded from environment variables (.env file).
Provides helper functions for connection-string derivation and validation.
"""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (one level above /app)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ── Streamlit secrets bridge ─────────────────────────────────────────────────
# When running inside Streamlit, inject secrets into os.environ so that
# libraries like LlamaIndex and the Gemini SDK pick them up automatically.
try:
    import streamlit as st

    if hasattr(st, "secrets"):
        for _key in ("GOOGLE_API_KEY", "LLAMA_CLOUD_API_KEY", "NEON_DATABASE_URL"):
            if _key in st.secrets:
                os.environ.setdefault(_key, st.secrets[_key])
except ImportError:
    pass  # streamlit not installed — running outside Streamlit is fine


# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
NEON_DATABASE_URL: str = os.environ.get("NEON_DATABASE_URL", "")
LLAMA_CLOUD_API_KEY: str = os.environ.get("LLAMA_CLOUD_API_KEY", "")

# ── Gemini model names ────────────────────────────────────────────────────────
GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
EMBEDDING_DIMENSION: int = 3072

# ── LLM generation settings ───────────────────────────────────────────────────
LLM_TEMPERATURE: float = 0.1

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K_CHUNKS: int = 5

# ── Vector store table name ───────────────────────────────────────────────────
VECTOR_TABLE_NAME: str = "document_chunks_llama"
HASH_TABLE_NAME: str = "ingested_document_hashes"


# ── Connection helpers ────────────────────────────────────────────────────────

def get_async_connection_string() -> str:
    """
    Derive an asyncpg connection URL from NEON_DATABASE_URL.

    PGVectorStore needs *both* a sync (psycopg2) and an async (asyncpg)
    connection string.  Neon typically provides ``postgresql://…``, which
    works as-is for psycopg2 but must be prefixed for asyncpg.
    """
    url = NEON_DATABASE_URL
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def compute_file_hash(file_path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file (for dedup checks)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_config() -> None:
    """Raise an error if required environment variables are missing."""
    missing = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not NEON_DATABASE_URL:
        missing.append("NEON_DATABASE_URL")
    if not LLAMA_CLOUD_API_KEY:
        missing.append("LLAMA_CLOUD_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}.\n"
            "Please set them in the .env file."
        )
