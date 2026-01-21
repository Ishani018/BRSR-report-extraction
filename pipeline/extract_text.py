"""
Module for extracting text from PDFs (both text-based and scanned).
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional
import io

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import pytesseract

from config.config import OCR_DPI

logger = logging.getLogger(__name__)


class PageText:
    """Container for extracted page text with metadata."""
    
    def __init__(self, page_number: int, text: str, method: str):
        self.page_number = page_number
        self.text = text
        self.method = method  # 'direct' or 'ocr'
        self.char_count = len(text)


def extract_text_with_table_support(page) -> str:
    """
    Extract text from a page using pdfplumber's native table extraction.
    Preserves table structures while extracting regular text.
    
    Args:
        page: pdfplumber page object
        
    Returns:
        Extracted text with tables formatted using pipe separators
    """
    try:
        # Find all tables on the page
        tables = page.find_tables()
        
        if not tables:
            # No tables found, extract text normally
            return page.extract_text() or ""
        
        # Get text outside table bounding boxes
        # Create a list of table bounding boxes to exclude
        table_boxes = []
        for table in tables:
            # Get bounding box of the table
            bbox = table.bbox
            table_boxes.append({
                'x0': bbox[0],
                'y0': bbox[1],
                'x1': bbox[2],
                'y1': bbox[3]
            })
        
        # Extract words and filter out those inside table boxes
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        non_table_words = []
        
        for word in words:
            word_center_x = (word['x0'] + word['x1']) / 2
            word_center_y = (word['top'] + word['bottom']) / 2
            
            # Check if word is inside any table box
            is_in_table = False
            for box in table_boxes:
                if (box['x0'] <= word_center_x <= box['x1'] and
                    box['y0'] <= word_center_y <= box['y1']):
                    is_in_table = True
                    break
            
            if not is_in_table:
                non_table_words.append(word)
        
        # Convert non-table words to text (preserving layout)
        non_table_text = ""
        if non_table_words:
            # Sort by vertical then horizontal position
            non_table_words.sort(key=lambda w: (round(w['top']), w['x0']))
            
            current_line = []
            current_top = non_table_words[0]['top'] if non_table_words else 0
            
            for word in non_table_words:
                word_text = word['text'].strip()
                if not word_text:
                    continue
                    
                if abs(word['top'] - current_top) <= 5:
                    # Add word to current line (will be space-separated on join)
                    current_line.append(word_text)
                else:
                    # New line - join current line with spaces and add newline
                    if current_line:
                        non_table_text += ' '.join(current_line) + '\n'
                    current_line = [word_text]
                    current_top = word['top']
            
            # Final line - ensure space-separated
            if current_line:
                non_table_text += ' '.join(current_line) + '\n'
        
        # Extract tables row by row with pipe separators
        table_texts = []
        for table in tables:
            try:
                extracted_table = table.extract()
                if extracted_table:
                    table_rows = []
                    for row in extracted_table:
                        if row:
                            # Filter out None values and join with pipe separator
                            clean_row = [str(cell) if cell else '' for cell in row]
                            table_rows.append(' | '.join(clean_row))
                    
                    if table_rows:
                        table_texts.append('\n'.join(table_rows))
            except Exception as e:
                logger.debug(f"Error extracting table: {e}")
                continue
        
        # Combine non-table text and table text
        result_parts = []
        if non_table_text.strip():
            result_parts.append(non_table_text.strip())
        
        for table_text in table_texts:
            if table_text.strip():
                result_parts.append("\n[TABLE]\n" + table_text.strip() + "\n[/TABLE]")
        
        return '\n\n'.join(result_parts) if result_parts else ""
        
    except Exception as e:
        logger.debug(f"Table extraction failed, using standard extraction: {e}")
        return page.extract_text() or ""


def extract_text_from_text_pdf(pdf_path: Path) -> List[PageText]:
    """
    Extract text from a text-based PDF using pdfplumber with proper column and table handling.
    
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
                    # First try table-aware extraction
                    text = extract_text_with_table_support(page)
                    
                    # Fallback to column detection if table extraction yields little
                    if not text or len(text.strip()) < 50:
                        text = extract_text_with_column_detection(page)
                    
                    # Final fallback to standard extraction
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
    
    # Calculate extraction statistics with detailed metrics
    char_counts = [len(p.text) for p in pages]
    stripped_counts = [len(p.text.strip()) for p in pages]
    
    # Categorize pages by content quality
    empty_pages = sum(1 for c in stripped_counts if c == 0)
    low_content_pages = sum(1 for c in stripped_counts if 0 < c <= 100)
    moderate_content_pages = sum(1 for c in stripped_counts if 100 < c <= 1000)
    good_content_pages = sum(1 for c in stripped_counts if c > 1000)
    
    total_chars = sum(char_counts)
    pages_with_content = sum(1 for c in stripped_counts if c > 50)
    
    # Calculate statistical measures
    import statistics
    non_empty_counts = [c for c in stripped_counts if c > 0]
    
    extraction_stats = {
        "total_pages_in_pdf": len(pages),
        "pages_extracted": len(pages),
        "pages_with_content": pages_with_content,
        "total_characters": total_chars,
        "avg_chars_per_page": total_chars / len(pages) if pages else 0,
        "extraction_coverage": (pages_with_content / len(pages) * 100) if pages else 0,
        "page_quality_distribution": {
            "empty_pages": empty_pages,
            "low_content_pages": low_content_pages,  # 1-100 chars
            "moderate_content_pages": moderate_content_pages,  # 101-1000 chars
            "good_content_pages": good_content_pages  # >1000 chars
        },
        "character_statistics": {
            "min_chars_per_page": min(stripped_counts) if stripped_counts else 0,
            "max_chars_per_page": max(stripped_counts) if stripped_counts else 0,
            "median_chars_per_page": statistics.median(non_empty_counts) if non_empty_counts else 0,
            "std_dev_chars_per_page": round(statistics.stdev(non_empty_counts), 2) if len(non_empty_counts) > 1 else 0
        },
        "potential_issues": {
            "empty_or_failed_pages": empty_pages,
            "suspiciously_low_content": low_content_pages,
            "page_numbers_with_low_content": [p.page_number for p in pages if 0 < len(p.text.strip()) <= 100]
        }
    }
    
    logger.info(f"Extracted text from {len(pages)} pages")
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Pages with content: {pages_with_content}/{len(pages)} ({extraction_stats['extraction_coverage']:.1f}%)")
    logger.info(f"Quality: Empty={empty_pages}, Low={low_content_pages}, Moderate={moderate_content_pages}, Good={good_content_pages}")
    
    return pages, extraction_stats


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
            word_text = word['text'].strip()
            if not word_text:
                continue
            
            # If word is on roughly the same line (within 5 pixels), add to current line
            if abs(word['top'] - current_top) <= 5:
                # Ensure word is appended with space preservation
                current_line.append(word_text)
            else:
                # New line - save current line with space-separated words
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word_text]
                current_top = word['top']
        
        # Don't forget the last line - ensure space-separated
        if current_line:
            lines.append(' '.join(current_line))
        
        # Join lines with newlines (which preserve word boundaries)
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
                # Join columns with double newline to preserve structure
                # Each column already has words space-separated via words_to_text
                return left_text.rstrip() + "\n\n" + right_text.lstrip()
        
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
                # Join columns with double newline to preserve structure
                # Each column already has words space-separated via words_to_text
                return left_text.rstrip() + "\n\n" + right_text.lstrip()
        
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
        logger.error(f"Error with PyMuPDF fallback: {e}")
    
    # Calculate extraction statistics with detailed metrics
    char_counts = [len(p.text) for p in pages]
    stripped_counts = [len(p.text.strip()) for p in pages]
    
    # Categorize pages by content quality
    empty_pages = sum(1 for c in stripped_counts if c == 0)
    low_content_pages = sum(1 for c in stripped_counts if 0 < c <= 100)
    moderate_content_pages = sum(1 for c in stripped_counts if 100 < c <= 1000)
    good_content_pages = sum(1 for c in stripped_counts if c > 1000)
    
    total_chars = sum(char_counts)
    pages_with_content = sum(1 for c in stripped_counts if c > 50)
    
    # Calculate statistical measures
    import statistics
    non_empty_counts = [c for c in stripped_counts if c > 0]
    
    extraction_stats = {
        "total_pages_in_pdf": len(pages),
        "pages_extracted": len(pages),
        "pages_with_content": pages_with_content,
        "total_characters": total_chars,
        "avg_chars_per_page": total_chars / len(pages) if pages else 0,
        "extraction_coverage": (pages_with_content / len(pages) * 100) if pages else 0,
        "page_quality_distribution": {
            "empty_pages": empty_pages,
            "low_content_pages": low_content_pages,
            "moderate_content_pages": moderate_content_pages,
            "good_content_pages": good_content_pages
        },
        "character_statistics": {
            "min_chars_per_page": min(stripped_counts) if stripped_counts else 0,
            "max_chars_per_page": max(stripped_counts) if stripped_counts else 0,
            "median_chars_per_page": statistics.median(non_empty_counts) if non_empty_counts else 0,
            "std_dev_chars_per_page": round(statistics.stdev(non_empty_counts), 2) if len(non_empty_counts) > 1 else 0
        },
        "potential_issues": {
            "empty_or_failed_pages": empty_pages,
            "suspiciously_low_content": low_content_pages,
            "page_numbers_with_low_content": [p.page_number for p in pages if 0 < len(p.text.strip()) <= 100]
        }
    }
    
    return pages, extraction_stats


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
                
                # Image pre-processing for better OCR accuracy
                # Step 1: Convert to grayscale
                if image.mode != 'L':
                    image = image.convert('L')
                
                # Step 2: Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)  # Increase contrast by 50%
                
                # Step 3: Apply binary thresholding to clean up gray backgrounds
                # Convert to numpy array for thresholding (if available)
                try:
                    import numpy as np
                    img_array = np.array(image)
                    # Apply adaptive thresholding
                    threshold = np.mean(img_array)
                    img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
                    image = Image.fromarray(img_array, mode='L')
                except ImportError:
                    # If numpy not available, use simple threshold
                    from PIL import ImageOps
                    image = ImageOps.autocontrast(image)
                
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
    
    # Calculate extraction statistics with detailed metrics
    char_counts = [len(p.text) for p in pages]
    stripped_counts = [len(p.text.strip()) for p in pages]
    
    # Categorize pages by content quality
    empty_pages = sum(1 for c in stripped_counts if c == 0)
    low_content_pages = sum(1 for c in stripped_counts if 0 < c <= 100)
    moderate_content_pages = sum(1 for c in stripped_counts if 100 < c <= 1000)
    good_content_pages = sum(1 for c in stripped_counts if c > 1000)
    
    total_chars = sum(char_counts)
    pages_with_content = sum(1 for c in stripped_counts if c > 50)
    
    # Calculate statistical measures
    import statistics
    non_empty_counts = [c for c in stripped_counts if c > 0]
    
    extraction_stats = {
        "total_pages_in_pdf": len(pages),
        "pages_extracted": len(pages),
        "pages_with_content": pages_with_content,
        "total_characters": total_chars,
        "avg_chars_per_page": total_chars / len(pages) if pages else 0,
        "extraction_coverage": (pages_with_content / len(pages) * 100) if pages else 0,
        "page_quality_distribution": {
            "empty_pages": empty_pages,
            "low_content_pages": low_content_pages,
            "moderate_content_pages": moderate_content_pages,
            "good_content_pages": good_content_pages
        },
        "character_statistics": {
            "min_chars_per_page": min(stripped_counts) if stripped_counts else 0,
            "max_chars_per_page": max(stripped_counts) if stripped_counts else 0,
            "median_chars_per_page": statistics.median(non_empty_counts) if non_empty_counts else 0,
            "std_dev_chars_per_page": round(statistics.stdev(non_empty_counts), 2) if len(non_empty_counts) > 1 else 0
        },
        "potential_issues": {
            "empty_or_failed_pages": empty_pages,
            "suspiciously_low_content": low_content_pages,
            "page_numbers_with_low_content": [p.page_number for p in pages if 0 < len(p.text.strip()) <= 100]
        }
    }
    
    logger.info(f"OCR completed for {len(pages)} pages")
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Pages with content: {pages_with_content}/{len(pages)} ({extraction_stats['extraction_coverage']:.1f}%)")
    logger.info(f"Quality: Empty={empty_pages}, Low={low_content_pages}, Moderate={moderate_content_pages}, Good={good_content_pages}")
    
    return pages, extraction_stats


def extract_text(pdf_path: Path, pdf_type: str) -> tuple[List[PageText], dict]:
    """
    Extract text from a PDF file based on its type.
    
    Args:
        pdf_path: Path to the PDF file
        pdf_type: Type of PDF ('text' or 'scanned')
        
    Returns:
        Tuple of (List of PageText objects, extraction statistics dict)
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
