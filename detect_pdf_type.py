"""
Module for detecting whether a PDF is text-based or requires OCR.
"""
import logging
from pathlib import Path
from typing import Tuple

import pdfplumber
import fitz  # PyMuPDF

from config import MIN_TEXT_LENGTH

logger = logging.getLogger(__name__)


def detect_pdf_type(pdf_path: Path) -> Tuple[str, dict]:
    """
    Detect whether a PDF contains extractable text or requires OCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (pdf_type, metadata) where pdf_type is 'text' or 'scanned'
        and metadata contains diagnostic information
    """
    logger.info(f"Detecting PDF type for: {pdf_path.name}")
    
    metadata = {
        "total_pages": 0,
        "text_pages": 0,
        "total_characters": 0,
        "avg_chars_per_page": 0
    }
    
    try:
        # Try with pdfplumber first
        with pdfplumber.open(pdf_path) as pdf:
            metadata["total_pages"] = len(pdf.pages)
            
            # Sample first 10 pages to determine type
            sample_size = min(10, len(pdf.pages))
            total_chars = 0
            
            for i in range(sample_size):
                try:
                    text = pdf.pages[i].extract_text()
                    if text:
                        char_count = len(text.strip())
                        total_chars += char_count
                        if char_count > MIN_TEXT_LENGTH:
                            metadata["text_pages"] += 1
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i+1}: {e}")
                    continue
            
            metadata["total_characters"] = total_chars
            metadata["avg_chars_per_page"] = total_chars / sample_size if sample_size > 0 else 0
            
            # Determine PDF type based on text density
            if metadata["text_pages"] >= sample_size * 0.5:  # At least 50% pages have text
                logger.info(f"PDF type: TEXT-BASED ({metadata['text_pages']}/{sample_size} pages with text)")
                return "text", metadata
            else:
                logger.info(f"PDF type: SCANNED ({metadata['text_pages']}/{sample_size} pages with text)")
                return "scanned", metadata
                
    except Exception as e:
        logger.error(f"Error detecting PDF type: {e}")
        # Fallback: Try with PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            metadata["total_pages"] = len(doc)
            
            sample_size = min(10, len(doc))
            total_chars = 0
            text_pages = 0
            
            for page_num in range(sample_size):
                page = doc[page_num]
                text = page.get_text()
                char_count = len(text.strip())
                total_chars += char_count
                if char_count > MIN_TEXT_LENGTH:
                    text_pages += 1
            
            doc.close()
            
            metadata["text_pages"] = text_pages
            metadata["total_characters"] = total_chars
            metadata["avg_chars_per_page"] = total_chars / sample_size if sample_size > 0 else 0
            
            if text_pages >= sample_size * 0.5:
                logger.info(f"PDF type: TEXT-BASED (fallback detection)")
                return "text", metadata
            else:
                logger.info(f"PDF type: SCANNED (fallback detection)")
                return "scanned", metadata
                
        except Exception as fallback_error:
            logger.error(f"Fallback detection also failed: {fallback_error}")
            return "unknown", metadata


def get_pdf_info(pdf_path: Path) -> dict:
    """
    Extract basic metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    info = {
        "filename": pdf_path.name,
        "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
        "pages": 0,
        "title": "",
        "author": "",
        "creation_date": ""
    }
    
    try:
        doc = fitz.open(pdf_path)
        info["pages"] = len(doc)
        
        metadata = doc.metadata
        if metadata:
            info["title"] = metadata.get("title", "")
            info["author"] = metadata.get("author", "")
            info["creation_date"] = metadata.get("creationDate", "")
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error extracting PDF info: {e}")
    
    return info
