"""
Module for cleaning extracted text from PDFs.
"""
import logging
import re
from typing import List, Set, Dict
from collections import Counter

from extract_text import PageText
from config import MIN_LINE_LENGTH, HEADER_FOOTER_THRESHOLD

logger = logging.getLogger(__name__)


def remove_extra_whitespace(text: str) -> str:
    """
    Remove excessive whitespace while preserving paragraph structure.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with maximum two
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove spaces at line starts/ends
    lines = [line.strip() for line in text.split('\n')]
    
    return '\n'.join(lines)


def fix_broken_lines(text: str) -> str:
    """
    Attempt to fix lines that were broken by PDF extraction.
    
    Args:
        text: Input text
        
    Returns:
        Text with fixed line breaks
    """
    lines = text.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # If line is too short and doesn't end with punctuation, might be broken
        if line and len(line) < 80 and not line[-1] in '.!?:;,':
            # Check if next line exists and starts with lowercase
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and next_line[0].islower():
                    # Merge lines
                    fixed_lines.append(line + ' ' + next_line)
                    i += 2
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)


def detect_repeated_elements(pages: List[PageText], threshold: float = HEADER_FOOTER_THRESHOLD) -> Dict[str, Set[str]]:
    """
    Detect repeated headers and footers across pages.
    
    Args:
        pages: List of PageText objects
        threshold: Minimum frequency (as ratio) to consider as repeated
        
    Returns:
        Dictionary with 'headers' and 'footers' sets
    """
    logger.info("Detecting repeated headers and footers...")
    
    first_lines = []
    last_lines = []
    
    for page in pages:
        lines = [l.strip() for l in page.text.split('\n') if l.strip()]
        if lines:
            # Get first few lines
            first_lines.extend(lines[:3])
            # Get last few lines
            last_lines.extend(lines[-3:])
    
    # Count occurrences
    first_counter = Counter(first_lines)
    last_counter = Counter(last_lines)
    
    # Identify repeated elements
    min_occurrences = len(pages) * threshold
    
    headers = {line for line, count in first_counter.items() 
               if count >= min_occurrences and len(line) > MIN_LINE_LENGTH}
    footers = {line for line, count in last_counter.items() 
               if count >= min_occurrences and len(line) > MIN_LINE_LENGTH}
    
    logger.info(f"Detected {len(headers)} header patterns and {len(footers)} footer patterns")
    
    return {'headers': headers, 'footers': footers}


def remove_headers_footers(text: str, patterns: Dict[str, Set[str]]) -> str:
    """
    Remove identified headers and footers from text.
    
    Args:
        text: Input text
        patterns: Dictionary with 'headers' and 'footers' sets
        
    Returns:
        Cleaned text
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip if line matches a header or footer pattern
        if line_stripped in patterns['headers'] or line_stripped in patterns['footers']:
            continue
        
        # Skip page numbers (simple pattern)
        if re.match(r'^\d+$', line_stripped) or re.match(r'^Page \d+$', line_stripped, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_noise_patterns(text: str) -> str:
    """
    Remove common noise patterns from PDF extraction.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove form feed characters
    text = text.replace('\f', '\n')
    
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
    
    # Remove excessive dashes/underscores (likely formatting artifacts)
    text = re.sub(r'-{5,}', '', text)
    text = re.sub(r'_{5,}', '', text)
    
    # Remove lone special characters on their own lines
    text = re.sub(r'\n[^\w\s]\n', '\n', text)
    
    return text


def clean_text(text: str) -> str:
    """
    Apply all cleaning operations to text.
    
    Args:
        text: Input text
        
    Returns:
        Fully cleaned text
    """
    # Apply cleaning operations in sequence
    text = remove_noise_patterns(text)
    text = fix_broken_lines(text)
    text = remove_extra_whitespace(text)
    
    return text


def clean_pages(pages: List[PageText]) -> List[PageText]:
    """
    Clean text for all pages, including header/footer removal.
    
    Args:
        pages: List of PageText objects
        
    Returns:
        List of cleaned PageText objects
    """
    logger.info(f"Cleaning text from {len(pages)} pages...")
    
    # Detect repeated elements across all pages
    repeated_patterns = detect_repeated_elements(pages)
    
    # Clean each page
    cleaned_pages = []
    for page in pages:
        cleaned_text = clean_text(page.text)
        cleaned_text = remove_headers_footers(cleaned_text, repeated_patterns)
        
        # Create new PageText object with cleaned text
        cleaned_page = PageText(
            page_number=page.page_number,
            text=cleaned_text,
            method=page.method
        )
        cleaned_pages.append(cleaned_page)
    
    logger.info("Text cleaning completed")
    return cleaned_pages


def remove_short_lines(text: str, min_length: int = MIN_LINE_LENGTH) -> str:
    """
    Remove lines that are too short (likely artifacts).
    
    Args:
        text: Input text
        min_length: Minimum line length to keep
        
    Returns:
        Text with short lines removed
    """
    lines = text.split('\n')
    filtered_lines = [line for line in lines if len(line.strip()) >= min_length or line.strip() == '']
    return '\n'.join(filtered_lines)


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to standard forms.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Replace common Unicode issues
    replacements = {
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u00a0': ' ',  # Non-breaking space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text
