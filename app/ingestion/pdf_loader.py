"""
pdf_loader.py - Converts a PDF file into per-page PIL images using pdf2image.
Each page becomes an image that is handed off to the OCR engine.
"""

from pathlib import Path
from typing import List, Tuple

from pdf2image import convert_from_path
from PIL.Image import Image

from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_pdf_as_images(
    pdf_path: str | Path,
    dpi: int = 300,
) -> List[Tuple[int, Image]]:
    """
    Convert a PDF file into a list of (page_number, PIL.Image) tuples.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        dpi: Resolution for rendering pages. 300 DPI is recommended for OCR.

    Returns:
        List of (1-based page_number, PIL.Image) tuples.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Loading PDF: {pdf_path.name}")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info(f"Loaded {len(images)} pages from PDF.")

    return [(page_num + 1, img) for page_num, img in enumerate(images)]
