"""
cleaner.py - Cleans raw OCR text while preserving financial/accounting data integrity.

Rules:
- Remove excessive whitespace and fix broken lines
- Preserve numbers, currency symbols, and table formatting
- Do NOT strip numeric values or accounting notations
"""

import re
from typing import List, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(raw_text: str) -> str:
    """
    Clean OCR output for a single page.

    Args:
        raw_text: Raw text string from Tesseract.

    Returns:
        Cleaned text string.
    """
    # Normalize unicode dashes, bullets and quotes
    text = raw_text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2022", "*")

    # Remove form-feed and carriage-return characters
    text = text.replace("\f", "\n").replace("\r", "\n")

    # Collapse runs of spaces/tabs INTO a single space (per line)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    # Merge lines: if a line ends mid-word (no punctuation and next line
    # starts with lowercase), join with a space. Otherwise keep newline.
    merged: List[str] = []
    i = 0
    while i < len(cleaned_lines):
        current = cleaned_lines[i]
        if (
            current
            and i + 1 < len(cleaned_lines)
            and cleaned_lines[i + 1]
            and not current[-1] in ".!?:;,\u201d\u2019"
            and cleaned_lines[i + 1][0].islower()
        ):
            merged.append(current + " " + cleaned_lines[i + 1])
            i += 2
        else:
            merged.append(current)
            i += 1

    # Remove excessive blank lines (keep at most 2 consecutive blank lines)
    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_pages(
    pages: List[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    """
    Clean a list of (page_number, raw_text) tuples.

    Returns:
        List of (page_number, cleaned_text) tuples.
        Pages with empty text after cleaning are excluded.
    """
    result = []
    for page_num, raw_text in pages:
        cleaned = clean_text(raw_text)
        if cleaned:
            result.append((page_num, cleaned))
        else:
            logger.debug(f"Page {page_num} empty after cleaning — skipping.")
    logger.info(f"Cleaning done. {len(result)} non-empty pages.")
    return result
