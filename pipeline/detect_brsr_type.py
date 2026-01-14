"""
BRSR Type Detector - Classifies whether BRSR is standalone or embedded in annual report.
Content-First Approach: Analyzes text content of first pages to determine document type.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import re

from pipeline.detect_pdf_type import get_pdf_info
from pipeline.extract_text import extract_text, PageText

logger = logging.getLogger(__name__)


def analyze_first_pages_content(pages: list, num_pages: int = 5) -> Dict:
    """
    Analyze first few pages for document type indicators.
    
    Args:
        pages: List of PageText objects
        num_pages: Number of pages to analyze (default: 5)
        
    Returns:
        Dictionary with analysis results
    """
    if not pages:
        return {
            'has_brsr_title': False,
            'has_annual_report_title': False,
            'has_financial_statements_toc': False,
            'combined_text': '',
            'first_page_text': ''
        }
    
    # Analyze first num_pages
    first_pages = pages[:min(num_pages, len(pages))]
    combined_text = "\n\n".join([p.text for p in first_pages])
    first_page_text = pages[0].text if pages else ""
    
    combined_lower = combined_text.lower()
    first_page_lower = first_page_text.lower()
    
    # Check for BRSR title patterns (standalone context)
    brsr_title_patterns = [
        r'business\s+responsibility\s+and\s+sustainability\s+report',
        r'brsr\s+report',
        r'business\s+responsibility\s+report',
        r'standalone\s+brsr',
        r'standalone\s+business\s+responsibility'
    ]
    
    has_brsr_title = False
    for pattern in brsr_title_patterns:
        if re.search(pattern, combined_lower, re.IGNORECASE):
            has_brsr_title = True
            break
    
    # Check for Annual Report title patterns
    annual_report_patterns = [
        r'annual\s+report\s+\d{4}',
        r'annual\s+report\s+\d{4}[-/]\d{2,4}',
        r'\d{4}[-/]\d{2,4}\s+annual\s+report',
        r'integrated\s+annual\s+report'
    ]
    
    has_annual_report_title = False
    for pattern in annual_report_patterns:
        if re.search(pattern, combined_lower, re.IGNORECASE):
            has_annual_report_title = True
            break
    
    # Check for Table of Contents referencing Financial Statements
    toc_indicators = [
        r'table\s+of\s+contents',
        r'contents',
        r'index'
    ]
    financial_statement_keywords = [
        r'financial\s+statements',
        r'balance\s+sheet',
        r'profit\s+and\s+loss',
        r'cash\s+flow',
        r'notes\s+to\s+accounts',
        r'auditor[\'s]?\s+report'
    ]
    
    has_financial_statements_toc = False
    # Check if TOC exists and contains financial statement references
    has_toc = any(re.search(pattern, combined_lower, re.IGNORECASE) for pattern in toc_indicators)
    if has_toc:
        # Look for financial statement keywords in the same context
        for keyword_pattern in financial_statement_keywords:
            if re.search(keyword_pattern, combined_lower, re.IGNORECASE):
                has_financial_statements_toc = True
                break
    
    return {
        'has_brsr_title': has_brsr_title,
        'has_annual_report_title': has_annual_report_title,
        'has_financial_statements_toc': has_financial_statements_toc,
        'combined_text': combined_text[:2000],  # First 2000 chars for debugging
        'first_page_text': first_page_text[:1000]  # First 1000 chars for debugging
    }


def detect_brsr_type(pdf_path: Path) -> Tuple[str, float, Dict]:
    """
    Detect whether BRSR is standalone or embedded in annual report.
    
    Content-First Approach:
    - Analyze text of first 5 pages
    - Standalone: "Business Responsibility and Sustainability Report" appears 
      WITHOUT "Annual Report" as main title
    - Embedded: "Annual Report 20xx" appears OR TOC references Financial Statements
    - Fallback: Use page count only if content is ambiguous
    
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
        # Detect PDF type (text vs scanned)
        from pipeline.detect_pdf_type import detect_pdf_type
        pdf_type, _ = detect_pdf_type(pdf_path)
        
        # Extract text (we'll analyze first 5 pages for classification)
        pages, _ = extract_text(pdf_path, pdf_type)
        
        if not pages:
            logger.warning(f"No text extracted from {pdf_path.name}")
            # Fallback: Try to get page count
            pdf_info = get_pdf_info(pdf_path)
            total_pages = pdf_info.get('pages', 0)
            if total_pages == 0:
                return 'unknown', 0.0, {'error': 'No text extracted and could not determine page count'}
            # Use page count as fallback
            if total_pages <= 50:
                return 'standalone', 0.5, {'error': 'No text extracted, using page count fallback', 'total_pages': total_pages}
            elif total_pages >= 100:
                return 'embedded', 0.5, {'error': 'No text extracted, using page count fallback', 'total_pages': total_pages}
            else:
                return 'unknown', 0.5, {'error': 'No text extracted, using page count fallback', 'total_pages': total_pages}
        
        # Analyze first pages content
        analysis = analyze_first_pages_content(pages, num_pages=5)
        
        # Get page count for fallback
        pdf_info = get_pdf_info(pdf_path)
        total_pages = pdf_info.get('pages', 0)
        
        metadata = {
            'total_pages': total_pages,
            'analysis': analysis
        }
        
        # Content-First Classification Logic
        
        # Rule 1: Standalone BRSR
        # BRSR title appears WITHOUT Annual Report title
        if analysis['has_brsr_title'] and not analysis['has_annual_report_title']:
            confidence = 0.9
            metadata['classification_reason'] = 'BRSR title found without Annual Report title'
            logger.info(f"Classified as STANDALONE (BRSR title without Annual Report, confidence: {confidence:.2f})")
            return 'standalone', confidence, metadata
        
        # Rule 2: Embedded BRSR
        # Annual Report title appears OR TOC references Financial Statements
        if analysis['has_annual_report_title']:
            confidence = 0.85
            metadata['classification_reason'] = 'Annual Report title found'
            logger.info(f"Classified as EMBEDDED (Annual Report title found, confidence: {confidence:.2f})")
            return 'embedded', confidence, metadata
        
        if analysis['has_financial_statements_toc']:
            confidence = 0.8
            metadata['classification_reason'] = 'TOC references Financial Statements'
            logger.info(f"Classified as EMBEDDED (TOC references Financial Statements, confidence: {confidence:.2f})")
            return 'embedded', confidence, metadata
        
        # Rule 3: Fallback - Use page count if content is ambiguous
        if total_pages > 0:
            if total_pages <= 50:
                # Small document, likely standalone
                confidence = 0.6
                metadata['classification_reason'] = 'Content ambiguous, using page count (<=50 pages)'
                logger.info(f"Classified as STANDALONE (page count fallback, confidence: {confidence:.2f})")
                return 'standalone', confidence, metadata
            elif total_pages >= 100:
                # Large document, likely embedded
                confidence = 0.6
                metadata['classification_reason'] = 'Content ambiguous, using page count (>=100 pages)'
                logger.info(f"Classified as EMBEDDED (page count fallback, confidence: {confidence:.2f})")
                return 'embedded', confidence, metadata
        
        # Default: Unknown
        confidence = 0.4
        metadata['classification_reason'] = 'Could not determine type from content or page count'
        logger.info(f"Classified as UNKNOWN (confidence: {confidence:.2f})")
        return 'unknown', confidence, metadata
        
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
