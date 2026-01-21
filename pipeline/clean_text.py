"""
Module for cleaning extracted text from PDFs.
Robust logic-first approach to fix mashed words, reversed text, and preserve table alignment.
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


def is_mostly_english(text: str, threshold: float = 0.7) -> bool:
    """
    Check if text is mostly English (ASCII characters).
    
    Args:
        text: Text to check
        threshold: Minimum ratio of ASCII characters (0.0-1.0)
        
    Returns:
        True if mostly English, False otherwise
    """
    if not text.strip():
        return True  # Empty text is considered "English"
    
    ascii_count = sum(1 for char in text if 32 <= ord(char) < 127)
    total_count = len(text)
    
    if total_count == 0:
        return True
    
    ratio = ascii_count / total_count
    return ratio >= threshold


def filter_non_english_text(text: str) -> str:
    """
    Filter out non-English text, keeping only ASCII and common English characters.
    Removes non-ASCII characters (Unicode characters from other languages).
    
    Args:
        text: Input text that may contain non-English characters
        
    Returns:
        Text with non-English characters removed or replaced with spaces
    """
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Only keep ASCII printable characters (0x20-0x7E)
        # This includes: letters, digits, spaces, and common punctuation
        filtered_line = ''.join(char if 32 <= ord(char) < 127 else ' ' for char in line)
        
        # Collapse multiple spaces that might be created from removed characters
        filtered_line = ' '.join(filtered_line.split())
        
        if filtered_line.strip():  # Keep line if it has content after filtering
            filtered_lines.append(filtered_line)
        else:
            # If line is empty after filtering, keep the empty line to preserve structure
            filtered_lines.append('')
    
    return '\n'.join(filtered_lines)


def filter_non_english_lines(text: str) -> str:
    """
    Remove lines that are mostly non-English.
    This provides stricter filtering by removing entire lines that don't meet the English threshold.
    
    Args:
        text: Input text
        
    Returns:
        Text with non-English lines removed
    """
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Keep line if it's mostly English OR if it's empty/short (might be formatting)
        if not line.strip() or len(line.strip()) < 3 or is_mostly_english(line):
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def fix_mashed_words(text: str) -> str:
    """
    Fix concatenated/mashed words like 'KeyHighligFY24' or 'costynapmoC'.
    Inserts spaces at CamelCase transitions and before number sequences.
    
    Args:
        text: Input text with potentially mashed words
        
    Returns:
        Text with spaces inserted at word boundaries
    """
    # Pattern 1: Insert space before uppercase letter following lowercase (CamelCase)
    # Example: "KeyHighlig" -> "Key Highlig"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Pattern 2: Insert space before number sequences following letters
    # Example: "HighligFY24" -> "Highlig FY24"
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    
    # Pattern 3: Insert space after number sequences before letters
    # Example: "FY24Key" -> "FY24 Key"
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    
    # Pattern 4: Fix common reversed words that got mashed (e.g., "costynapmoC" -> "cost ynapmoC")
    # Look for lowercase followed by reversed capitalization pattern
    text = re.sub(r'([a-z]+)([a-z]+[A-Z]\b)', r'\1 \2', text)
    
    return text


def fix_broken_lines(text: str) -> str:
    """
    Merge sentences broken across lines by PDF extraction.
    Smart logic: only merges when line is short and doesn't end with sentence punctuation.
    
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
        
        # Merge Condition: 
        # - Line is short (< 80 chars)
        # - Does NOT end with sentence punctuation (.!?:;)
        # - Next line exists and starts with lowercase letter
        if (len(line) < 80 and 
            line[-1] not in '.!?:;' and 
            i + 1 < len(lines)):
            
            next_line = lines[i + 1].strip()
            # Check if next line starts with lowercase (likely continuation)
            if next_line and next_line[0].islower():
                # CRITICAL: Merge lines with explicit space separator to prevent word mashing
                # Ensure line ends without space and next_line starts without space, then add space
                merged = line.rstrip() + ' ' + next_line.lstrip()
                fixed_lines.append(merged)
                i += 2
                continue
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)


def calculate_orientation_score(segment: str) -> float:
    """
    Calculate a score indicating how likely a segment is reversed.
    Returns a value between 0.0 (normal) and 1.0 (definitely reversed).
    More aggressive detection to catch more reversed text.
    
    Args:
        segment: Text segment to analyze
        
    Returns:
        Score between 0.0 and 1.0
    """
    segment = segment.strip()
    if not segment or len(segment) < 2:
        return 0.0
    
    score = 0.0
    
    # Signal 1: Word ends with capital letter but has lowercase elsewhere (STRONG SIGNAL)
    # Strong indicator: "ynapmoC", "detimiL", "elbirianoS"
    if re.search(r'[a-z]+[A-Z]\b', segment):
        score += 0.8  # Increased from 0.6 - very strong indicator
    
    # Signal 2: Expanded reversed stopwords (more comprehensive)
    reversed_stopwords = {
        'eht', 'rof', 'dna', 'si', 'fo', 'ot', 'ni', 'no', 'ta', 'eh', 'ti',
        'latot', 'etad', 'eman', 'srs', 'sl', 'rni', 'sy', 'tuo', 'sa', 'as',
        'tel', 'hcum', 'ylluf', 'ecnegilletni', 'ecnegilletni', 'gnissecorp',
        'tseb', 'evah', 'siht', 'taht', 'ylluf', 'flesym', 'gnissecorp',
        'yllanif', 'gnidliub', 'ecnegilletni', 'yllacificeps'
    }
    words = re.findall(r'\b\w+\b', segment.lower())
    if len(words) >= 1:  # Lowered threshold from 2 to 1
        reversed_matches = sum(1 for word in words if word in reversed_stopwords)
        if reversed_matches > 0:
            match_ratio = reversed_matches / len(words) if words else 0
            score += 0.5 * match_ratio  # Increased weight
            # If ANY reversed stopword found, add bonus
            if match_ratio >= 0.3:
                score += 0.3
    
    # Signal 3: Punctuation at start but not at end
    if segment and segment[0] in '.,;:' and segment[-1] not in '.,!?;:':
        if len(segment) > 3:  # Lowered threshold from 5 to 3
            score += 0.4  # Increased from 0.3
    
    # Signal 4: Check for common reversed patterns (e.g., "elbirianoS" = "Sustainability")
    # Pattern: lowercase letters, then capital at end
    if re.search(r'[a-z]{3,}[A-Z]\b', segment):
        score += 0.4
    
    # Signal 5: Unusual capitalization patterns (lowercase word ending with capital)
    # Check if words have reversed capitalization
    word_patterns = re.findall(r'\b[a-z]+[A-Z]\b', segment)
    if len(word_patterns) > 0:
        score += 0.3 * min(len(word_patterns), 3)  # Cap at 3 patterns
    
    # Normalize score (cap at 1.0)
    return min(1.0, score)


def fix_reversed_text(text: str) -> str:
    """
    Detect and fix text segments that are visually reversed.
    Cell-aware: splits by pipe first, evaluates each cell independently.
    More aggressive detection to catch more reversed text.
    
    Args:
        text: Input text that might contain reversed segments
        
    Returns:
        Text with reversed segments fixed
    """
    def fix_segment(segment: str) -> str:
        """
        Fix a single text segment if it's reversed.
        
        Args:
            segment: Text segment to fix
            
        Returns:
            Fixed segment (reversed if detected as reversed, otherwise unchanged)
        """
        segment = segment.strip()
        if not segment or len(segment) < 2:
            return segment
        
        # Calculate orientation score
        score = calculate_orientation_score(segment)
        
        # Lowered threshold from 0.5 to 0.35 for more aggressive detection
        # This catches more reversed text while still avoiding false positives
        if score >= 0.35:
            fixed = segment[::-1]
            logger.debug(f"Fixed reversed segment (score={score:.2f}): '{segment[:40]}...' -> '{fixed[:40]}...'")
            return fixed
        
        return segment
    
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        if not line.strip():
            fixed_lines.append(line)
            continue
        
        # Check if line contains table separators (pipe characters)
        if '|' in line:
            # Split by pipe and process each cell independently
            segments = line.split('|')
            fixed_segments = []
            for seg in segments:
                # Preserve leading/trailing spaces around pipe, but clean segment content
                fixed_seg = fix_segment(seg.strip())
                fixed_segments.append(fixed_seg)
            # Rejoin with pipe and ensure spaces around pipe to prevent word mashing
            # This preserves table structure while ensuring word boundaries
            fixed_line = ' | '.join(fixed_segments)
            fixed_lines.append(fixed_line)
        else:
            # Regular line - process as a single segment
            fixed_line = fix_segment(line)
            fixed_lines.append(fixed_line)
    
    return '\n'.join(fixed_lines)


def remove_noise(text: str) -> str:
    """
    Remove common PDF artifacts and noise patterns.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace form feeds with newlines
    text = text.replace('\f', '\n')
    
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
    
    # Remove excessive dashes/underscores (often used for signature lines)
    text = re.sub(r'-{5,}', '', text)
    text = re.sub(r'_{5,}', '', text)
    
    # Remove lone special characters on their own lines (e.g., just `.` or `,`)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just special characters (but keep empty lines)
        if stripped and not re.search(r'[\w]', stripped):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    return text


def remove_extra_whitespace_smart(text: str) -> str:
    """
    Remove excessive whitespace with table-aware logic.
    Only collapses multiple spaces if line does NOT contain pipe character.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace tabs with 4 spaces
    text = text.replace('\t', '    ')
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Check if line contains pipe character (table row)
        if '|' in line:
            # Table row: only strip leading/trailing whitespace, preserve internal spacing
            cleaned_lines.append(line.strip())
        else:
            # Regular line: collapse multiple spaces into one
            cleaned_lines.append(' '.join(line.split()))
    
    text = '\n'.join(cleaned_lines)
    
    # Replace excessive newlines (more than 3) with maximum 2
    text = re.sub(r'\n{4,}', '\n\n', text)
    
    return text


def clean_text(text: str) -> str:
    """
    Apply all cleaning operations in strict order:
    0. Filter non-English text FIRST (English only)
    1. Fix mashed words
    2. Fix broken lines (merge split words)
    3. Fix reversed text (cell-aware flip)
    4. Remove noise (cleanup)
    
    Args:
        text: Input text
        
    Returns:
        Fully cleaned text (English only)
    """
    # Step 0: Filter non-English text FIRST (before any other processing)
    # This removes all non-ASCII characters to ensure only English text passes through
    text = filter_non_english_text(text)
    
    # Step 1: Fix mashed words (before any other processing)
    text = fix_mashed_words(text)
    
    # Step 2: Fix broken lines (merge split sentences)
    text = fix_broken_lines(text)
    
    # Step 3: Fix reversed text (cell-aware)
    text = fix_reversed_text(text)
    
    # Step 4: Remove noise patterns
    text = remove_noise(text)
    
    # Step 5: Smart whitespace (table-aware)
    text = remove_extra_whitespace_smart(text)
    
    # Step 6: Normalize Unicode
    text = normalize_unicode(text)
    
    # Step 7: Fix hyphenated/split words
    text = fix_split_words(text)
    
    # Step 8: Standardize table placeholders
    text = standardize_table_placeholders(text, placeholder="N/A")
    
    return text


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


# ============================================================================
# Header/Footer Detection (for clean_pages)
# ============================================================================

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
