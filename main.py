"""
Main orchestration script for the PDF processing pipeline.
"""
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm

from config import DATA_DIR, LOGS_DIR, LOG_FORMAT, LOG_LEVEL
from detect_pdf_type import detect_pdf_type, get_pdf_info
from extract_text import extract_text
from clean_text import clean_pages
from export_outputs import export_to_docx, create_output_directory


def setup_logging(log_file: Path = None) -> None:
    """
    Configure logging for the pipeline.
    
    Args:
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=handlers
    )


def parse_company_year(pdf_path: Path) -> tuple[str, str]:
    """
    Extract company name and year from directory structure or filename.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (company_name, year)
    """
    # Try to extract from directory structure
    # Expected: data/CompanyName/2022/report.pdf or data/CompanyName/report_2022.pdf
    
    parts = pdf_path.parts
    
    # Look for year in path (4-digit number between 1900-2099)
    year = None
    for part in reversed(parts):
        if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2099:
            year = part
            break
    
    # If not in path, try filename
    if not year:
        import re
        year_match = re.search(r'(19|20)\d{2}', pdf_path.stem)
        if year_match:
            year = year_match.group(0)
        else:
            year = "unknown"
    
    # Get company name from parent directory or filename
    if pdf_path.parent.name.isdigit():
        # Year is directory, company is grandparent
        company_name = pdf_path.parent.parent.name
    else:
        # Company is parent directory
        company_name = pdf_path.parent.name
    
    # Clean up company name
    company_name = company_name.replace('-', ' ').replace('_', ' ').strip()
    
    return company_name, year


def process_single_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Process a single PDF file through the entire pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with processing results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"{'='*80}")
    
    result = {
        "pdf_path": str(pdf_path),
        "status": "failed",
        "error": None
    }
    
    try:
        # Parse company and year
        company_name, year = parse_company_year(pdf_path)
        logger.info(f"Company: {company_name}, Year: {year}")
        
        result["company"] = company_name
        result["year"] = year
        
        # Step 1: Get PDF info
        logger.info("Step 1/6: Getting PDF information...")
        pdf_info = get_pdf_info(pdf_path)
        logger.info(f"  Pages: {pdf_info.get('pages', 'N/A')}, "
                   f"Size: {pdf_info.get('file_size_mb', 0):.2f} MB")
        
        # Step 2: Detect PDF type
        logger.info("Step 2/6: Detecting PDF type...")
        pdf_type, type_metadata = detect_pdf_type(pdf_path)
        logger.info(f"  Type: {pdf_type.upper()}")
        
        # Step 3: Extract text
        logger.info("Step 3/6: Extracting text...")
        pages = extract_text(pdf_path, pdf_type)
        logger.info(f"  Extracted text from {len(pages)} pages")
        
        # Step 4: Clean text (minimal cleaning to preserve layout)
        logger.info("Step 4/5: Cleaning text...")
        cleaned_pages = clean_pages(pages)
        total_chars = sum(p.char_count for p in cleaned_pages)
        logger.info(f"  Cleaned {total_chars:,} characters")
        
        # Step 5: Export to DOCX (skip tables and complex segmentation for now)
        logger.info("Step 5/5: Exporting to DOCX...")
        
        # Create output directory
        output_path = create_output_directory(company_name, year)
        
        # Export just the DOCX with page-by-page content
        docx_path = export_to_docx(cleaned_pages, [], output_path, company_name, year)
        
        # Create simple metadata
        metadata = {
            "company": company_name,
            "year": year,
            "processing_date": datetime.now().isoformat(),
            "pdf_info": pdf_info,
            "pdf_type": pdf_type,
            "statistics": {
                "total_pages": len(cleaned_pages),
                "total_characters": total_chars,
                "extraction_method": pdf_type
            }
        }
        
        json_path = output_path / "metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        result["output_directory"] = str(output_path)
        result["docx"] = str(docx_path)
        result["metadata"] = str(json_path)
        result["files_created"] = [str(docx_path), str(json_path)]
        result["status"] = "success"
        
        logger.info(f"Successfully processed {pdf_path.name}")
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        result["error"] = str(e)
        result["status"] = "failed"
    
    return result


def find_all_pdfs(data_dir: Path) -> List[Path]:
    """
    Find all PDF files in the data directory.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        List of paths to PDF files
    """
    pdf_files = list(data_dir.rglob("*.pdf"))
    return sorted(pdf_files)


def main():
    """Main entry point for the PDF processing pipeline."""
    
    # Setup logging
    log_file = LOGS_DIR / f"pipeline_{logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None), '%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting PDF Processing Pipeline")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Log file: {log_file}")
    
    # Find all PDFs
    logger.info("Scanning for PDF files...")
    pdf_files = find_all_pdfs(DATA_DIR)
    
    if not pdf_files:
        logger.error(f"No PDF files found in {DATA_DIR}")
        logger.info("Please add PDF files to the data directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF
    results = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        result = process_single_pdf(pdf_path)
        results.append(result)
    
    # Generate summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    logger.info(f"Total PDFs processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.info("\nFailed files:")
        for r in results:
            if r["status"] == "failed":
                logger.info(f"  - {r['pdf_path']}: {r.get('error', 'Unknown error')}")
    
    # Save summary report
    summary_path = create_summary_report(results)
    logger.info(f"\nSummary report saved to: {summary_path}")
    
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    main()
