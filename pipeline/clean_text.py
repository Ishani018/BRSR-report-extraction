"""
Module for cleaning extracted text from PDFs.
"""
import logging
import re
import unicodedata
from typing import List, Set, Dict
from collections import Counter
from difflib import SequenceMatcher

from pipeline.extract_text import PageText
from config.config import MIN_LINE_LENGTH, HEADER_FOOTER_THRESHOLD

logger = logging.getLogger(__name__)


def remove_extra_whitespace(text: str) -> str:
    """
    Remove excessive whitespace while preserving paragraph structure and layout.
    More conservative to maintain readability.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace tabs with spaces
    text = text.replace('\t', '    ')
    
    # Replace multiple spaces with single space (but preserve intentional spacing)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Don't collapse spaces if line seems to be formatted (e.g., tables, aligned text)
        if '  ' in line and len(line) > 20:
            # Preserve formatting for what looks like tabular data
            cleaned_lines.append(line.rstrip())
        else:
            # Normal line - collapse spaces
            cleaned_lines.append(' '.join(line.split()))
    
    text = '\n'.join(cleaned_lines)
    
    # Replace excessive newlines (more than 3) with maximum 2
    text = re.sub(r'\n{4,}', '\n\n', text)
    
    return text


def fix_broken_lines(text: str) -> str:
    """
    Attempt to fix lines that were broken by PDF extraction.
    More conservative to avoid breaking intentional line breaks.
    
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
        
        # Skip empty lines
        if not line:
            fixed_lines.append(line)
            i += 1
            continue
        
        # If line is very short (< 40 chars) and doesn't end with punctuation or colon
        # AND next line starts with lowercase, might be broken
        if (len(line) < 40 and 
            line and 
            line[-1] not in '.!?:;,' and 
            i + 1 < len(lines)):
            
            next_line = lines[i + 1].strip()
            # Only merge if next line exists, starts with lowercase, and is not too short
            if next_line and len(next_line) > 5 and next_line[0].islower():
                # Merge lines
                fixed_lines.append(line + ' ' + next_line)
                i += 2
                continue
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)


def fuzzy_match_similar_lines(lines: List[str], similarity_threshold: float = 0.85) -> List[Set[str]]:
    """
    Group lines that are similar (>85% similarity) using fuzzy matching.
    
    Args:
        lines: List of lines to group
        similarity_threshold: Minimum similarity ratio (0-1) to consider lines as similar
        
    Returns:
        List of sets, where each set contains similar lines
    """
    groups = []
    processed = set()
    
    for i, line1 in enumerate(lines):
        if i in processed:
            continue
        
        # Start a new group with this line
        group = {line1}
        processed.add(i)
        
        # Find all similar lines
        for j, line2 in enumerate(lines[i+1:], start=i+1):
            if j in processed:
                continue
            
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, line1, line2).ratio()
            
            if similarity >= similarity_threshold:
                group.add(line2)
                processed.add(j)
        
        if group:
            groups.append(group)
    
    return groups


def detect_repeated_elements(pages: List[PageText], threshold: float = HEADER_FOOTER_THRESHOLD) -> Dict[str, Set[str]]:
    """
    Detect repeated headers and footers across pages using fuzzy matching.
    This handles cases where headers/footers have changing page numbers.
    
    Args:
        pages: List of PageText objects
        threshold: Minimum frequency (as ratio) to consider as repeated
        
    Returns:
        Dictionary with 'headers' and 'footers' sets
    """
    logger.info("Detecting repeated headers and footers (fuzzy matching)...")
    
    first_lines = []
    last_lines = []
    
    for page in pages:
        lines = [l.strip() for l in page.text.split('\n') if l.strip() and len(l.strip()) > MIN_LINE_LENGTH]
        if lines:
            # Get first few lines
            first_lines.extend(lines[:3])
            # Get last few lines
            last_lines.extend(lines[-3:])
    
    # Use fuzzy matching to group similar lines
    first_groups = fuzzy_match_similar_lines(first_lines, similarity_threshold=0.85)
    last_groups = fuzzy_match_similar_lines(last_lines, similarity_threshold=0.85)
    
    # Count occurrences of each group (a group represents similar lines)
    min_occurrences = len(pages) * threshold
    
    headers = set()
    for group in first_groups:
        # Count how many lines in this group appear in first_lines
        group_count = sum(1 for line in first_lines if any(
            SequenceMatcher(None, line, group_line).ratio() >= 0.85 for group_line in group
        ))
        
        if group_count >= min_occurrences:
            # Add all lines from this group (they're all similar)
            headers.update(group)
    
    footers = set()
    for group in last_groups:
        # Count how many lines in this group appear in last_lines
        group_count = sum(1 for line in last_lines if any(
            SequenceMatcher(None, line, group_line).ratio() >= 0.85 for group_line in group
        ))
        
        if group_count >= min_occurrences:
            # Add all lines from this group (they're all similar)
            footers.update(group)
    
    logger.info(f"Detected {len(headers)} header patterns and {len(footers)} footer patterns")
    
    return {'headers': headers, 'footers': footers}


def remove_headers_footers(text: str, patterns: Dict[str, Set[str]]) -> str:
    """
    Remove identified headers and footers from text using fuzzy matching.
    
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
        
        # Skip if line matches a header or footer pattern (exact match)
        if line_stripped in patterns['headers'] or line_stripped in patterns['footers']:
            continue
        
        # Also check fuzzy match against patterns (for lines with minor variations)
        is_header = any(
            SequenceMatcher(None, line_stripped, pattern).ratio() >= 0.85
            for pattern in patterns['headers']
        )
        is_footer = any(
            SequenceMatcher(None, line_stripped, pattern).ratio() >= 0.85
            for pattern in patterns['footers']
        )
        
        if is_header or is_footer:
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
    # Step 1: Fix reversed text early (before other operations)
    text = fix_reversed_text(text)
    
    # Step 2: Normalize Unicode
    text = normalize_unicode(text)

    # Step 3: Fix hyphenated/split words across line breaks (logic-first)
    text = fix_split_words(text)
    
    # Step 4: Remove noise patterns
    text = remove_noise_patterns(text)
    
    # Step 5: Fix broken lines
    text = fix_broken_lines(text)
    
    # Step 6: Remove extra whitespace
    text = remove_extra_whitespace(text)

    # Step 7: Standardize empty table cells for pipe-delimited rows
    text = standardize_table_placeholders(text, placeholder="N/A")
    
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
    Normalize Unicode characters to standard forms using NFKC normalization
    and additional mappings for corporate report characters.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Step 1: NFKC normalization (decomposes and recomposes characters)
    text = unicodedata.normalize('NFKC', text)
    
    # Step 2: Map common "fancy" characters to ASCII equivalents
    replacements = {
        # Smart quotes
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2032': "'",  # Prime
        '\u2033': '"',  # Double prime
        
        # Dashes
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2015': '--', # Horizontal bar
        '\u2212': '-',  # Minus sign
        
        # Spaces
        '\u00a0': ' ',  # Non-breaking space
        '\u2000': ' ',  # En quad
        '\u2001': ' ',  # Em quad
        '\u2002': ' ',  # En space
        '\u2003': ' ',  # Em space
        '\u2009': ' ',  # Thin space
        
        # Bullet points
        '\u2022': '•',  # Bullet
        '\u25cf': '•',  # Black circle
        '\u25cb': 'o',  # White circle
        '\u2023': '>',  # Triangular bullet
        '\u25aa': '▪',  # Black small square
        '\u25ab': '▫',  # White small square
        
        # Other common characters
        '\u00ae': '(R)',  # Registered sign
        '\u00a9': '(C)',  # Copyright sign
        '\u2122': '(TM)', # Trade mark sign
        '\u00b0': 'deg',  # Degree sign
        '\u00b1': '+/-',  # Plus-minus sign
        '\u00d7': 'x',    # Multiplication sign
        '\u00f7': '/',    # Division sign
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def fix_reversed_text(text: str) -> str:
    """
    Detect and fix lines that are visually reversed.
    Uses heuristic: if a significant percentage of words look like reversed
    common stopwords, reverse the string back.
    
    Args:
        text: Input text that might contain reversed lines
        
    Returns:
        Text with reversed lines fixed
    """
    # Common English stopwords that when reversed might appear in corrupted text
    reversed_stopwords = {
        'eht',  # 'the' reversed
        'rof',  # 'for' reversed
        'dna',  # 'and' reversed
        'si',   # 'is' reversed
        'fo',   # 'of' reversed
        'a',    # 'a' reversed (still 'a')
        'ta',   # 'at' reversed
        'ot',   # 'to' reversed
        'ni',   # 'in' reversed
        'eh',   # 'he' reversed
        'as',   # 'sa' (might be 'as' reversed, though 'as' is palindrome)
        'tuo',  # 'out' reversed
        'no',   # 'on' reversed
        'sa',   # 'as' (though 'as' is palindrome)
        'ti',   # 'it' reversed
        # Reversed table/header tokens commonly seen in BRSR tables
        'latot',  # total
        'on',     # no
        'etad',   # date
        'eman',   # name
        'srs',    # sr.s
        'sl',     # sl
        'rni',    # inr
        'sy',     # yes
    }
    
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        if not line.strip() or len(line.strip()) < 10:
            # Skip very short lines
            fixed_lines.append(line)
            continue
        
        # Check words in the line
        words = re.findall(r'\b\w+\b', line.lower())
        
        if len(words) < 3:
            # Need at least 3 words to make a determination
            fixed_lines.append(line)
            continue
        
        # Count how many words match reversed stopwords
        reversed_matches = sum(1 for word in words if word in reversed_stopwords)
        match_ratio = reversed_matches / len(words) if words else 0
        
        # If >20% of words are reversed stopwords, likely reversed text
        if match_ratio > 0.2:
            # Reverse the line
            fixed_lines.append(line[::-1])
            logger.debug(f"Fixed reversed text: {line[:50]}... -> {line[::-1][:50]}...")
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_split_words(text: str) -> str:
    """
    Rejoin words split by hyphen + newline.
    Example: "environ-\\nment" -> "environment"
    
    Constraint:
    - Only merge when both sides are alphabetic word parts (avoid list items/codes).
    """
    # Match "Word-\\nWord" or "Word- \\nWord"
    # Require alphabetic on both sides.
    pattern = re.compile(r'([A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,})')
    return pattern.sub(r'\1\2', text)


def standardize_table_placeholders(text: str, placeholder: str = "N/A") -> str:
    """
    Standardize empty values in pipe-delimited table rows.
    If a line contains '|' and has empty segments, replace empties with placeholder.
    Example: "A |  | C" -> "A | N/A | C"
    """
    lines = text.split('\n')
    out_lines = []
    
    for line in lines:
        if '|' not in line:
            out_lines.append(line)
            continue
        
        parts = [p.strip() for p in line.split('|')]
        parts = [p if p != '' else placeholder for p in parts]
        out_lines.append(' | '.join(parts))
    
    return '\n'.join(out_lines)
