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
try:
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError

    # Only attempt to access secrets if we are actually running inside Streamlit
    # and streamlit is imported successfully.
    if hasattr(st, "runtime") and st.runtime.exists():
        try:
            if hasattr(st, "secrets"):
                for _key in ("GROQ_API_KEY", "LLAMA_CLOUD_API_KEY", "NEON_DATABASE_URL"):
                    if _key in st.secrets:
                        os.environ.setdefault(_key, st.secrets[_key])
        except (StreamlitSecretNotFoundError, Exception):
            pass # Fall back to environment variables
except ImportError:
    pass  # streamlit not installed — running outside Streamlit is fine


# ── API Keys ──────────────────────────────────────────────────────────────────
NEON_DATABASE_URL: str = os.environ.get("NEON_DATABASE_URL", "")
LLAMA_CLOUD_API_KEY: str = os.environ.get("LLAMA_CLOUD_API_KEY", "")
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# ── Local HuggingFace embedding model ────────────────────────────────────────
# Runs entirely on CPU/GPU — no API key, no rate limits.
# Model is downloaded once (~133 MB) to ~/.cache/huggingface/ on first use.
HF_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION: int = 384

# ── LLM generation settings ───────────────────────────────────────────────────
LLM_TEMPERATURE: float = 0.1

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K_CHUNKS: int = 5

# ── Vector store table name ───────────────────────────────────────────────────
VECTOR_TABLE_NAME: str = "document_chunks_llama"
HASH_TABLE_NAME: str = "ingested_document_hashes"

# ── Default document (auto-ingested on first launch) ─────────────────────────
DEFAULT_PDF_PATH: str = str(Path(__file__).parent.parent / "Annual-Report-FY-2023-24.pdf")
DEFAULT_PDF_DOC_NAME: str = "Annual-Report-FY-2023-24"


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
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not NEON_DATABASE_URL:
        missing.append("NEON_DATABASE_URL")
    if not LLAMA_CLOUD_API_KEY:
        missing.append("LLAMA_CLOUD_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}.\n"
            "Please set them in the .env file."
        )