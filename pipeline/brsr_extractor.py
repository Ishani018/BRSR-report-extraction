"""
BRSR Extractor - Extract BRSR sections from embedded annual reports.
Content-First Approach: Extract all text first, then filter pages by content.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional

from pipeline.section_boundary_detector import SectionBoundaryDetector
from pipeline.extract_text import PageText
from pipeline.export_outputs import export_brsr_to_docx

logger = logging.getLogger(__name__)


def extract_brsr_from_annual_report(
    pdf_path: Path,
    pages: List[PageText],
    output_dir: Path,
    company_name: str,
    year: str
) -> Optional[Dict]:
    """
    Extract BRSR section from embedded annual report using Content-First approach.
    
    Flow: Extract All -> Filter Pages
    1. Iterate through pages to find start_index using is_brsr_start_page
    2. If found, iterate from start_index + 1 to find end_index using is_financial_end_page
    3. Slice the list: brsr_pages = pages[start_index : end_index]
    4. Pass sliced list to export functions
    
    Args:
        pdf_path: Path to PDF file (not used in new approach, kept for compatibility)
        pages: List of PageText objects from main pipeline (full document)
        output_dir: Directory for extracted section outputs
        company_name: Company name
        year: Report year
        
    Returns:
        Dictionary with extraction results or None if failed
    """
    logger.info(f"Extracting BRSR section from annual report: {pdf_path.name}")
    
    try:
        if not pages:
            logger.warning("No pages provided for extraction")
            return None
        
        # Step 1: Find BRSR start page
        logger.info("Step 1: Finding BRSR start page...")
        start_index = None
        
        for i, page in enumerate(pages):
            if SectionBoundaryDetector.is_brsr_start_page(page.text):
                start_index = i
                logger.info(f"Found BRSR start at page {page.page_number} (index {i})")
                break
        
        if start_index is None:
            logger.warning("Could not find BRSR start page")
            return None
        
        # Step 2: Find financial statements end page
        logger.info("Step 2: Finding BRSR end page (start of Financial Statements)...")
        end_index = len(pages)  # Default to end of document
        
        for i in range(start_index + 1, len(pages)):
            page = pages[i]
            if SectionBoundaryDetector.is_financial_end_page(page.text):
                end_index = i
                logger.info(f"Found BRSR end at page {page.page_number} (index {i}) - Financial Statements start")
                break
        
        # Step 3: Slice the pages list
        brsr_pages = pages[start_index:end_index]
        
        if not brsr_pages:
            logger.warning("No pages in BRSR section")
            return None
        
        start_page_number = brsr_pages[0].page_number
        end_page_number = brsr_pages[-1].page_number
        
        logger.info(f"BRSR section: pages {start_page_number}-{end_page_number} ({len(brsr_pages)} pages)")
        
        # Step 4: Export sliced pages to DOCX
        logger.info("Step 3: Exporting BRSR section to DOCX...")
        
        try:
            docx_path = export_brsr_to_docx(
                pages=brsr_pages,
                output_path=output_dir,
                company_name=company_name,
                year=year,
                is_standalone=False,  # This is embedded BRSR
                is_from_annual=True
            )
            logger.info(f"Exported BRSR section DOCX to: {docx_path}")
        except Exception as e:
            logger.error(f"Error exporting DOCX: {e}", exc_info=True)
            return None
        
        result = {
            'success': True,
            'section_type': 'brsr_embedded',
            'start_page': start_page_number,
            'end_page': end_page_number,
            'page_count': len(brsr_pages),
            'docx_path': str(docx_path),
            'confidence': 1.0  # High confidence when boundaries are found via content
        }
        
        logger.info(f"Successfully extracted BRSR section: pages {start_page_number}-{end_page_number}")
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
        print(f"  Page Count: {result['page_count']}")
        print(f"  DOCX: {result['docx_path']}")
    else:
        print("\n✗ Failed to extract BRSR section")
        sys.exit(1)
