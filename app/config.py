"""
config.py - Central configuration for the RAG system.
All settings loaded from environment variables (.env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level above /app)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
NEON_DATABASE_URL: str = os.environ.get("NEON_DATABASE_URL", "")

# ── Gemini model names ────────────────────────────────────────────────────────
GEMINI_LLM_MODEL: str = "gemini-2.5-flash-lite"
GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSION: int = 768  # truncated from 3072 via output_dimensionality

# ── LLM generation settings ───────────────────────────────────────────────────
LLM_TEMPERATURE: float = 0.1
LLM_TOP_P: float = 0.8

# ── Chunking settings ────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500        # characters
CHUNK_OVERLAP: int = 100     # characters

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K_CHUNKS: int = 5        # number of chunks to retrieve per query


# ── Validation ────────────────────────────────────────────────────────────────
def validate_config() -> None:
    """Raise an error if required environment variables are missing."""
    missing = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not NEON_DATABASE_URL:
        missing.append("NEON_DATABASE_URL")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}.\n"
            "Please set them in the .env file."
        )
