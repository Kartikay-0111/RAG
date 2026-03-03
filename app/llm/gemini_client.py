"""
gemini_client.py - Wrapper for the Gemini LLM API for answer generation.

Uses: google.genai SDK (latest)
Settings: temperature=0.1, top_p=0.8
"""

from google import genai
from google.genai import types as genai_types

from app.config import GOOGLE_API_KEY, GEMINI_LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_client() -> genai.Client:
    """Return a configured Gemini API client."""
    return genai.Client(api_key=GOOGLE_API_KEY)


def generate_answer(prompt: str) -> str:
    """
    Send a fully-formed prompt to Gemini and return the text response.
    """
    logger.info("Sending prompt to Gemini LLM...")
    client = _get_client()

    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
        ),
    )

    answer = response.text.strip()
    logger.info("Gemini LLM response received.")
    return answer
