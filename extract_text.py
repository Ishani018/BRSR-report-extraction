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
    Extract text from a text-based PDF using pdfplumber with proper column handling.
    
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
                    # Try to detect and handle multi-column layouts
                    text = extract_text_with_column_detection(page)
                    
                    # Fallback to standard extraction if column detection fails
                    if not text or len(text.strip()) < 50:
                        text = page.extract_text() or ""
                    
                    pages.append(PageText(
                        page_number=i + 1,
                        text=text,
                        method='direct'
                    ))
                    logger.debug(f"Extracted {len(text)} characters from page {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i+1}: {e}")
                    # Try fallback
                    try:
                        text = page.extract_text() or ""
                        pages.append(PageText(
                            page_number=i + 1,
                            text=text,
                            method='direct'
                        ))
                    except:
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


def extract_text_with_column_detection(page) -> str:
    """
    Extract text from a page with automatic column detection.
    Dynamically handles any number of pages per PDF page (1, 2, 3, etc.), 
    each with their own column structure.
    
    Args:
        page: pdfplumber page object
        
    Returns:
        Extracted text with proper column order
    """
    # Get page dimensions
    page_width = page.width
    page_height = page.height
    
    # Get all words with their bounding boxes
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    
    if not words:
        return ""
    
    if len(words) < 20:
        # Not enough words to determine columns, use standard extraction
        return page.extract_text() or ""
    
    # Helper function to find gaps in a list of word centers
    def find_significant_gaps(word_centers, min_x, max_x, min_gap_size=30):
        sorted_centers = sorted(set(word_centers))
        gaps = []
        for i in range(len(sorted_centers) - 1):
            x_pos = sorted_centers[i]
            if min_x < x_pos < max_x:
                gap = sorted_centers[i + 1] - sorted_centers[i]
                if gap > min_gap_size:
                    gaps.append((gap, (sorted_centers[i] + sorted_centers[i + 1]) / 2))
        return sorted(gaps, reverse=True)  # Largest gaps first
    
    # Helper function to reconstruct text from words
    def words_to_text(word_list):
        if not word_list:
            return ""
        
        # Sort by vertical position, then horizontal
        word_list.sort(key=lambda w: (round(w['top']), w['x0']))
        
        lines = []
        current_line = []
        current_top = word_list[0]['top']
        
        for word in word_list:
            # If word is on roughly the same line (within 5 pixels), add to current line
            if abs(word['top'] - current_top) <= 5:
                current_line.append(word['text'])
            else:
                # New line - save current line and start new one
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word['text']]
                current_top = word['top']
        
        # Don't forget the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    # Helper function to extract columns from a page section
    def extract_page_with_columns(page_words, page_min_x, page_max_x):
        if not page_words:
            return ""
        
        # Find gaps within this page section
        page_centers = [(w['x0'] + w['x1']) / 2 for w in page_words]
        page_gaps = find_significant_gaps(page_centers, page_min_x, page_max_x, min_gap_size=30)
        
        if page_gaps and len(page_words) > 20:
            # This page has columns - use the largest gap as column boundary
            col_split = page_gaps[0][1]
            left_col = [w for w in page_words if (w['x0'] + w['x1']) / 2 < col_split]
            right_col = [w for w in page_words if (w['x0'] + w['x1']) / 2 >= col_split]
            
            if len(left_col) > 5 and len(right_col) > 5:
                # Both columns have content
                left_text = words_to_text(left_col)
                right_text = words_to_text(right_col)
                return left_text + "\n\n" + right_text
        
        # No columns or insufficient content - extract as single block
        return words_to_text(page_words)
    
    try:
        # Step 1: Find all large gaps that indicate page boundaries (>80 pixels)
        word_centers = [(w['x0'] + w['x1']) / 2 for w in words]
        all_gaps = find_significant_gaps(word_centers, page_width * 0.05, page_width * 0.95, min_gap_size=80)
        
        # Step 2: Determine page boundaries based on large gaps
        if all_gaps:
            # Create boundaries for each page section
            # Sort split points left to right
            split_points = sorted([gap[1] for gap in all_gaps])
            
            # Create page boundaries: [0, split1, split2, ..., width]
            boundaries = [0] + split_points + [page_width]
            
            # Extract text from each page section
            page_texts = []
            for i in range(len(boundaries) - 1):
                section_min_x = boundaries[i]
                section_max_x = boundaries[i + 1]
                
                # Get words in this section
                section_words = [w for w in words 
                               if section_min_x <= (w['x0'] + w['x1']) / 2 < section_max_x]
                
                if section_words:
                    section_text = extract_page_with_columns(section_words, section_min_x, section_max_x)
                    if section_text:
                        page_texts.append(section_text.strip())
            
            # Combine all page sections with separators
            if page_texts:
                return "\n\n=== PAGE BREAK ===\n\n".join(page_texts)
        
        # Step 3: No large gaps found - treat as single page with possible columns
        smaller_gaps = find_significant_gaps(word_centers, page_width * 0.1, page_width * 0.9, min_gap_size=30)
        
        if smaller_gaps:
            col_split = smaller_gaps[0][1]
            left_col = [w for w in words if (w['x0'] + w['x1']) / 2 < col_split]
            right_col = [w for w in words if (w['x0'] + w['x1']) / 2 >= col_split]
            
            if len(left_col) > 10 and len(right_col) > 10:
                left_text = words_to_text(left_col)
                right_text = words_to_text(right_col)
                return left_text.strip() + "\n\n" + right_text.strip()
        
        # No structure detected - extract normally
        return page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    
    except Exception as e:
        logger.debug(f"Column extraction failed, using standard: {e}")
        return page.extract_text() or ""


def extract_text_with_pymupdf(pdf_path: Path) -> List[PageText]:
    """
    Extract text using PyMuPDF with layout preservation.
    
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
                # Use "blocks" mode to preserve layout better than simple text extraction
                text = page.get_text("text", sort=True)
                
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
