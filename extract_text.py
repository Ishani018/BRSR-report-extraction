"""
Module for extracting text from PDFs (both text-based and scanned).
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional
import io

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from config import OCR_DPI

logger = logging.getLogger(__name__)


class PageText:
    """Container for extracted page text with metadata."""
    
    def __init__(self, page_number: int, text: str, method: str):
        self.page_number = page_number
        self.text = text
        self.method = method  # 'direct' or 'ocr'
        self.char_count = len(text)


def extract_text_from_text_pdf(pdf_path: Path) -> List[PageText]:
    """
    Extract text from a text-based PDF using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of PageText objects containing extracted text for each page
    """
    logger.info(f"Extracting text from text-based PDF: {pdf_path.name}")
    pages = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                    pages.append(PageText(
                        page_number=i + 1,
                        text=text,
                        method='direct'
                    ))
                    logger.debug(f"Extracted {len(text)} characters from page {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i+1}: {e}")
                    pages.append(PageText(
                        page_number=i + 1,
                        text="",
                        method='direct'
                    ))
                    
    except Exception as e:
        logger.error(f"Error opening PDF with pdfplumber: {e}")
        # Fallback to PyMuPDF
        return extract_text_with_pymupdf(pdf_path)
    
    logger.info(f"Extracted text from {len(pages)} pages")
    return pages


def extract_text_with_pymupdf(pdf_path: Path) -> List[PageText]:
    """
    Extract text using PyMuPDF as a fallback method.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of PageText objects
    """
    logger.info(f"Extracting text with PyMuPDF: {pdf_path.name}")
    pages = []
    
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text()
                pages.append(PageText(
                    page_number=page_num + 1,
                    text=text,
                    method='direct'
                ))
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num+1}: {e}")
                pages.append(PageText(
                    page_number=page_num + 1,
                    text="",
                    method='direct'
                ))
        doc.close()
        
    except Exception as e:
        logger.error(f"Error with PyMuPDF extraction: {e}")
    
    return pages


def extract_text_from_scanned_pdf(pdf_path: Path, dpi: int = OCR_DPI) -> List[PageText]:
    """
    Extract text from a scanned PDF using OCR (Tesseract).
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI resolution for rendering PDF pages
        
    Returns:
        List of PageText objects containing OCR-extracted text
    """
    logger.info(f"Extracting text from scanned PDF using OCR: {pdf_path.name}")
    pages = []
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            try:
                logger.debug(f"OCR processing page {page_num+1}/{total_pages}")
                page = doc[page_num]
                
                # Render page to image
                pix = page.get_pixmap(dpi=dpi)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Perform OCR
                text = pytesseract.image_to_string(image)
                
                pages.append(PageText(
                    page_number=page_num + 1,
                    text=text,
                    method='ocr'
                ))
                
                logger.debug(f"Extracted {len(text)} characters from page {page_num+1} via OCR")
                
            except Exception as e:
                logger.error(f"Error performing OCR on page {page_num+1}: {e}")
                pages.append(PageText(
                    page_number=page_num + 1,
                    text="",
                    method='ocr'
                ))
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error during OCR extraction: {e}")
    
    logger.info(f"OCR completed for {len(pages)} pages")
    return pages


def extract_text(pdf_path: Path, pdf_type: str) -> List[PageText]:
    """
    Main function to extract text from a PDF based on its type.
    
    Args:
        pdf_path: Path to the PDF file
        pdf_type: Type of PDF ('text' or 'scanned')
        
    Returns:
        List of PageText objects
    """
    if pdf_type == "scanned":
        return extract_text_from_scanned_pdf(pdf_path)
    else:
        return extract_text_from_text_pdf(pdf_path)


def get_full_text(pages: List[PageText]) -> str:
    """
    Combine all page texts into a single string.
    
    Args:
        pages: List of PageText objects
        
    Returns:
        Combined text from all pages
    """
    return "\n\n".join([f"--- Page {p.page_number} ---\n{p.text}" for p in pages])


def get_page_text(pages: List[PageText], page_number: int) -> Optional[str]:
    """
    Get text for a specific page.
    
    Args:
        pages: List of PageText objects
        page_number: Page number (1-indexed)
        
    Returns:
        Text for the specified page or None if not found
    """
    for page in pages:
        if page.page_number == page_number:
            return page.text
    return None
