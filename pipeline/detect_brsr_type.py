"""
BRSR Type Detector - Classifies whether BRSR is standalone or embedded in annual report.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import re

from config.config import (
    BRSR_STANDALONE_MAX_PAGES,
    BRSR_EMBEDDED_MIN_PAGES,
    BRSR_CONTENT_THRESHOLD,
    BRSR_KEYWORDS
)

from pipeline.detect_pdf_type import get_pdf_info
from pipeline.extract_text import extract_text, PageText
from pipeline.section_metadata import SectionType, SECTION_KEYWORDS

logger = logging.getLogger(__name__)


def count_brsr_keywords(text: str, keywords: list) -> int:
    """
    Count occurrences of BRSR keywords in text.
    
    Args:
        text: Text to search
        keywords: List of keywords to search for
        
    Returns:
        Number of keyword matches found
    """
    text_lower = text.lower()
    count = 0
    
    for keyword in keywords:
        # Count occurrences (case-insensitive)
        pattern = re.escape(keyword.lower())
        matches = len(re.findall(pattern, text_lower))
        count += matches
    
    return count


def analyze_first_pages(pages: list, num_pages: int = 5) -> Dict:
    """
    Analyze first few pages for BRSR keywords and structure.
    
    Args:
        pages: List of PageText objects
        num_pages: Number of pages to analyze
        
    Returns:
        Dictionary with analysis results
    """
    if not pages:
        return {
            'total_chars': 0,
            'keyword_count': 0,
            'keyword_density': 0.0,
            'has_brsr_title': False,
            'brsr_title_page': None
        }
    
    # Analyze first num_pages
    first_pages = pages[:min(num_pages, len(pages))]
    combined_text = "\n\n".join([p.text for p in first_pages])
    total_chars = sum(p.char_count for p in first_pages)
    
    # Count BRSR keywords
    brsr_keywords = SECTION_KEYWORDS.get(SectionType.BRSR, [])
    keyword_count = count_brsr_keywords(combined_text, brsr_keywords)
    
    # Check for BRSR in title/header (first page especially)
    has_brsr_title = False
    brsr_title_page = None
    
    if pages:
        first_page_text = pages[0].text.lower()
        # Check for BRSR report title patterns
        title_patterns = [
            r'business\s+responsibility\s+and\s+sustainability\s+report',
            r'brsr\s+report',
            r'business\s+responsibility\s+report',
            r'sustainability\s+report'
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, first_page_text, re.IGNORECASE):
                has_brsr_title = True
                brsr_title_page = 1
                break
    
    keyword_density = keyword_count / max(total_chars / 1000, 1)  # Keywords per 1000 chars
    
    return {
        'total_chars': total_chars,
        'keyword_count': keyword_count,
        'keyword_density': keyword_density,
        'has_brsr_title': has_brsr_title,
        'brsr_title_page': brsr_title_page
    }


def detect_brsr_type(pdf_path: Path) -> Tuple[str, float, Dict]:
    """
    Detect whether BRSR is standalone or embedded in annual report.
    
    Classification logic:
    - Standalone: Document < 50 pages and BRSR-focused (keywords in title/header)
    - Embedded: Document > 100 pages and contains BRSR section markers
    - Unknown: Medium-sized documents (50-100 pages) - analyze content density
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (type: str, confidence: float, metadata: dict)
        - type: 'standalone', 'embedded', or 'unknown'
        - confidence: Confidence score (0.0 to 1.0)
        - metadata: Dictionary with detection details
    """
    logger.info(f"Detecting BRSR type for: {pdf_path.name}")
    
    try:
        # Get PDF info (page count)
        pdf_info = get_pdf_info(pdf_path)
        total_pages = pdf_info.get('pages', 0)
        
        if total_pages == 0:
            logger.warning(f"Could not determine page count for {pdf_path.name}")
            return 'unknown', 0.0, {'error': 'Could not determine page count'}
        
        logger.debug(f"PDF has {total_pages} pages")
        
        # Detect PDF type (text vs scanned)
        from pipeline.detect_pdf_type import detect_pdf_type
        pdf_type, _ = detect_pdf_type(pdf_path)
        
        # Extract text (first few pages should be enough for classification)
        pages, _ = extract_text(pdf_path, pdf_type)
        
        if not pages:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return 'unknown', 0.0, {'error': 'No text extracted'}
        
        # Analyze first pages for BRSR keywords
        first_pages_analysis = analyze_first_pages(pages, num_pages=5)
        
        # Analyze full document for keyword density (sample pages for large documents)
        if total_pages > 10:
            # Sample pages for large documents
            sample_indices = [0, len(pages) // 4, len(pages) // 2, len(pages) * 3 // 4, len(pages) - 1]
            sample_pages = [pages[i] for i in sample_indices if i < len(pages)]
        else:
            sample_pages = pages
        
        sample_text = "\n\n".join([p.text for p in sample_pages])
        brsr_keywords = SECTION_KEYWORDS.get(SectionType.BRSR, [])
        total_keyword_count = count_brsr_keywords(sample_text, brsr_keywords)
        total_sample_chars = sum(p.char_count for p in sample_pages)
        overall_keyword_density = total_keyword_count / max(total_sample_chars / 1000, 1)
        
        # Classification logic
        metadata = {
            'total_pages': total_pages,
            'total_text_pages': len(pages),
            'first_pages_analysis': first_pages_analysis,
            'overall_keyword_density': overall_keyword_density,
            'total_keyword_count': total_keyword_count
        }
        
        # Rule 1: Standalone BRSR (< 50 pages and BRSR-focused)
        if total_pages <= BRSR_STANDALONE_MAX_PAGES:
            # Check if it's BRSR-focused (keywords in title or high density)
            if first_pages_analysis['has_brsr_title'] or overall_keyword_density > BRSR_CONTENT_THRESHOLD:
                confidence = 0.9 if first_pages_analysis['has_brsr_title'] else 0.7
                logger.info(f"Classified as STANDALONE (pages: {total_pages}, confidence: {confidence:.2f})")
                metadata['classification_reason'] = f'Pages <= {BRSR_STANDALONE_MAX_PAGES} and BRSR-focused'
                return 'standalone', confidence, metadata
        
        # Rule 2: Embedded BRSR (> 100 pages with BRSR section markers)
        if total_pages >= BRSR_EMBEDDED_MIN_PAGES:
            # Check for BRSR section markers (keywords present but not dominant)
            if overall_keyword_density > BRSR_CONTENT_THRESHOLD * 0.5:  # Some BRSR keywords but not dominant
                confidence = 0.8 if first_pages_analysis['keyword_count'] > 0 else 0.6
                logger.info(f"Classified as EMBEDDED (pages: {total_pages}, confidence: {confidence:.2f})")
                metadata['classification_reason'] = f'Pages >= {BRSR_EMBEDDED_MIN_PAGES} with BRSR section markers'
                return 'embedded', confidence, metadata
        
        # Rule 3: Medium-sized documents (50-100 pages) - analyze content density
        if BRSR_STANDALONE_MAX_PAGES < total_pages < BRSR_EMBEDDED_MIN_PAGES:
            # If BRSR keywords are prominent and document is focused, likely standalone
            if first_pages_analysis['has_brsr_title'] or overall_keyword_density > BRSR_CONTENT_THRESHOLD:
                confidence = 0.75 if first_pages_analysis['has_brsr_title'] else 0.6
                logger.info(f"Classified as STANDALONE (medium-sized, pages: {total_pages}, confidence: {confidence:.2f})")
                metadata['classification_reason'] = 'Medium-sized document with BRSR focus'
                return 'standalone', confidence, metadata
            else:
                # Could be embedded or mixed
                confidence = 0.5
                logger.info(f"Classified as UNKNOWN (medium-sized, pages: {total_pages}, confidence: {confidence:.2f})")
                metadata['classification_reason'] = 'Medium-sized document with unclear BRSR focus'
                return 'unknown', confidence, metadata
        
        # Default: Unknown
        logger.info(f"Classified as UNKNOWN (pages: {total_pages}, confidence: 0.5)")
        metadata['classification_reason'] = 'Could not determine BRSR type'
        return 'unknown', 0.5, metadata
        
    except Exception as e:
        logger.error(f"Error detecting BRSR type for {pdf_path.name}: {e}", exc_info=True)
        return 'unknown', 0.0, {'error': str(e)}


class BRSRTypeDetector:
    """
    BRSR Type Detector class with additional functionality.
    """
    
    def __init__(self):
        """Initialize BRSR Type Detector."""
        pass
    
    def detect(self, pdf_path: Path) -> Tuple[str, float, Dict]:
        """
        Detect BRSR type.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (type, confidence, metadata)
        """
        return detect_brsr_type(pdf_path)
    
    def is_standalone(self, pdf_path: Path, min_confidence: float = 0.7) -> Tuple[bool, float]:
        """
        Check if BRSR is standalone.
        
        Args:
            pdf_path: Path to PDF file
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (is_standalone: bool, confidence: float)
        """
        brsr_type, confidence, _ = detect_brsr_type(pdf_path)
        is_standalone = (brsr_type == 'standalone') and (confidence >= min_confidence)
        return is_standalone, confidence
    
    def is_embedded(self, pdf_path: Path, min_confidence: float = 0.7) -> Tuple[bool, float]:
        """
        Check if BRSR is embedded in annual report.
        
        Args:
            pdf_path: Path to PDF file
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (is_embedded: bool, confidence: float)
        """
        brsr_type, confidence, _ = detect_brsr_type(pdf_path)
        is_embedded = (brsr_type == 'embedded') and (confidence >= min_confidence)
        return is_embedded, confidence


if __name__ == "__main__":
    # Test the detector
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python detect_brsr_type.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    logger.info(f"Testing BRSR type detector on: {pdf_path.name}")
    detector = BRSRTypeDetector()
    
    brsr_type, confidence, metadata = detector.detect(pdf_path)
    
    print(f"\n{'='*80}")
    print(f"BRSR Type Detection Results")
    print(f"{'='*80}")
    print(f"Type: {brsr_type.upper()}")
    print(f"Confidence: {confidence:.2f}")
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}")
    
    sys.exit(0)

