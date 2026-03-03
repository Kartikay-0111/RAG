"""
ocr_engine.py - Runs Tesseract OCR on each PIL image and returns raw text per page.
"""

from typing import List, Tuple

import pytesseract
from PIL.Image import Image

from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_ocr(
    pages: List[Tuple[int, Image]],
    lang: str = "eng",
) -> List[Tuple[int, str]]:
    """
    Run Tesseract OCR on a list of (page_number, PIL.Image) tuples.

    Args:
        pages: List of (page_number, image) from pdf_loader.
        lang: Tesseract language code (default: 'eng').

    Returns:
        List of (page_number, raw_text) tuples.
    """
    results: List[Tuple[int, str]] = []

    for page_num, image in pages:
        logger.info(f"Running OCR on page {page_num} ...")
        try:
            text = pytesseract.image_to_string(image, lang=lang)
            results.append((page_num, text))
        except Exception as exc:
            logger.warning(f"OCR failed on page {page_num}: {exc}")
            results.append((page_num, ""))

    logger.info(f"OCR complete. Processed {len(results)} pages.")
    return results
