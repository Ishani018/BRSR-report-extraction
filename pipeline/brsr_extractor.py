"""
BRSR Extractor - Extract BRSR sections from embedded annual reports.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional

from pipeline.section_boundary_detector import SectionBoundaryDetector
from pipeline.section_content_extractor import SectionContentExtractor
from pipeline.section_metadata import SectionType, SectionBoundary, SectionContent, SECTION_KEYWORDS
from pipeline.extract_text import PageText
from pipeline.export_outputs import export_to_docx

logger = logging.getLogger(__name__)


def extract_brsr_from_annual_report(
    pdf_path: Path,
    pages: List[PageText],
    output_dir: Path,
    company_name: str,
    year: str
) -> Optional[Dict]:
    """
    Extract BRSR section from embedded annual report.
    
    Args:
        pdf_path: Path to PDF file
        pages: List of PageText objects from main pipeline
        output_dir: Directory for extracted section outputs
        company_name: Company name
        year: Report year
        
    Returns:
        Dictionary with extraction results or None if failed
    """
    logger.info(f"Extracting BRSR section from annual report: {pdf_path.name}")
    
    try:
        # Step 1: Detect BRSR section boundaries from PDF layout
        logger.info("Step 1: Detecting BRSR section boundaries...")
        detector = SectionBoundaryDetector(pdf_path)
        
        # Extend detector to handle BRSR sections
        # Modify detect_section_boundaries to include BRSR
        boundaries = detector.detect_section_boundaries()
        
        # Try to find BRSR boundary
        brsr_boundary = None
        
        # Method 1: Use embedded BRSR keywords
        brsr_keywords = SECTION_KEYWORDS.get(SectionType.BRSR_EMBEDDED, [])
        if not brsr_keywords:
            # Fallback to main BRSR keywords
            brsr_keywords = SECTION_KEYWORDS.get(SectionType.BRSR, [])
        
        # Find BRSR section using boundary detector logic
        # Extract layout metadata if not already done
        if not detector.text_blocks:
            detector.extract_layout_metadata()
        
        # Find BRSR section boundary
        candidates = []
        for block in detector.text_blocks:
            if not detector._is_potential_heading(block):
                continue
            
            normalized = block.normalized_text
            for keyword in brsr_keywords:
                if keyword in normalized:
                    confidence = detector._calculate_confidence(block, keyword, normalized)
                    candidates.append((block, confidence, keyword))
                    logger.debug(
                        f"BRSR candidate heading on page {block.page_number}: "
                        f"'{block.text}' (confidence={confidence:.2f})"
                    )
                    break
        
        if candidates:
            # Select best candidate
            best_block, best_confidence, matched_keyword = max(candidates, key=lambda x: x[1])
            
            # Find section end
            end_page = detector._find_section_end(best_block.page_number, SectionType.BRSR_EMBEDDED)
            
            brsr_boundary = SectionBoundary(
                section_type=SectionType.BRSR_EMBEDDED,
                start_page=best_block.page_number,
                end_page=end_page,
                confidence=best_confidence,
                start_heading=best_block.text,
                detection_method="layout_and_keywords"
            )
            
            logger.info(
                f"Found BRSR section: pages {brsr_boundary.start_page}-{brsr_boundary.end_page}, "
                f"confidence={brsr_boundary.confidence:.2f}"
            )
        
        if not brsr_boundary:
            logger.warning("Could not detect BRSR section boundaries")
            return None
        
        # Step 2: Extract BRSR content from processed text
        logger.info("Step 2: Extracting BRSR section content...")
        extractor = SectionContentExtractor(pages, output_dir)
        
        content = extractor.extract_section(brsr_boundary)
        
        if not content:
            logger.warning("Could not extract BRSR section content")
            return None
        
        # Step 3: Export extracted section to DOCX and JSON
        logger.info("Step 3: Exporting BRSR section...")
        
        # Export to DOCX
        docx_path = extractor.export_section_to_docx(content, company_name, year)
        
        # Export to JSON (if supported by extractor)
        # Note: section_content_extractor may need extension for JSON export
        # For now, we'll use the existing export methods
        
        result = {
            'success': True,
            'section_type': 'brsr_embedded',
            'boundary': brsr_boundary.to_dict(),
            'content': content.to_dict(),
            'docx_path': str(docx_path),
            'start_page': brsr_boundary.start_page,
            'end_page': brsr_boundary.end_page,
            'confidence': brsr_boundary.confidence
        }
        
        logger.info(f"Successfully extracted BRSR section: pages {brsr_boundary.start_page}-{brsr_boundary.end_page}")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting BRSR section: {e}", exc_info=True)
        return None


class BRSRExtractor:
    """
    BRSR Extractor class for extracting BRSR sections from annual reports.
    """
    
    def __init__(self):
        """Initialize BRSR Extractor."""
        pass
    
    def extract(
        self,
        pdf_path: Path,
        pages: List[PageText],
        output_dir: Path,
        company_name: str,
        year: str
    ) -> Optional[Dict]:
        """
        Extract BRSR section from embedded annual report.
        
        Args:
            pdf_path: Path to PDF file
            pages: List of PageText objects
            output_dir: Directory for outputs
            company_name: Company name
            year: Report year
            
        Returns:
            Dictionary with extraction results or None
        """
        return extract_brsr_from_annual_report(pdf_path, pages, output_dir, company_name, year)


if __name__ == "__main__":
    # Test the extractor
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python brsr_extractor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    logger.info(f"Testing BRSR extractor on: {pdf_path.name}")
    
    # Extract text first
    from pipeline.detect_pdf_type import detect_pdf_type
    from pipeline.extract_text import extract_text
    
    pdf_type, _ = detect_pdf_type(pdf_path)
    pages, _ = extract_text(pdf_path, pdf_type)
    
    if not pages:
        print("Could not extract text from PDF")
        sys.exit(1)
    
    # Extract BRSR section
    output_dir = Path(__file__).parent.parent / "outputs" / "test"
    extractor = BRSRExtractor()
    
    result = extractor.extract(
        pdf_path=pdf_path,
        pages=pages,
        output_dir=output_dir,
        company_name="Test Company",
        year="2023-24"
    )
    
    if result:
        print(f"\n✓ Successfully extracted BRSR section")
        print(f"  Pages: {result['start_page']}-{result['end_page']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  DOCX: {result['docx_path']}")
    else:
        print("\n✗ Failed to extract BRSR section")
        sys.exit(1)

