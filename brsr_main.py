"""
BRSR Main Orchestrator - Master script to orchestrate entire BRSR workflow.
"""
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config.config import (
    BRSR_FINANCIAL_YEARS,
    DOWNLOAD_BASE_DIR,
    OUTPUT_BASE_DIR,
    STATUS_DIR,
    LOGS_DIR
)
from data.company_reader import read_companies, CompanyReader
from downloaders.download_manager import DownloadManager, download_brsr_report
from pipeline.detect_brsr_type import detect_brsr_type, BRSRTypeDetector
from pipeline.detect_pdf_type import detect_pdf_type, get_pdf_info
from pipeline.extract_text import extract_text
from pipeline.clean_text import clean_pages
from pipeline.export_outputs import create_brsr_output_directory, export_brsr_to_docx
from pipeline.file_naming import format_brsr_output_filename, create_output_path
from pipeline.section_hierarchy_builder import SectionHierarchyBuilder
from pipeline.brsr_extractor import extract_brsr_from_annual_report, BRSRExtractor


def setup_logging(log_file: Path = None) -> None:
    """Setup logging configuration."""
    from config.config import LOG_FORMAT, LOG_LEVEL
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=handlers
    )


def process_brsr_pdf(
    pdf_path: Path,
    company_name: str,
    year: str,
    download_tier: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process BRSR PDF through the entire pipeline.
    
    Args:
        pdf_path: Path to PDF file
        company_name: Company name
        year: Financial year
        download_tier: Tier used for download (tier1, tier2, or None)
        
    Returns:
        Dictionary with processing results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing BRSR: {pdf_path.name}")
    logger.info(f"Company: {company_name}, Year: {year}")
    logger.info(f"{'='*80}")
    
    result = {
        "pdf_path": str(pdf_path),
        "company_name": company_name,
        "year": year,
        "download_tier": download_tier,
        "status": "failed",
        "error": None,
        "brsr_type": None,
        "output_files": []
    }
    
    try:
        # Step 1: Get PDF info
        logger.info("Step 1/7: Getting PDF information...")
        pdf_info = get_pdf_info(pdf_path)
        logger.info(f"  Pages: {pdf_info.get('pages', 'N/A')}, "
                   f"Size: {pdf_info.get('file_size_mb', 0):.2f} MB")
        
        # Step 2: Detect PDF type (text vs scanned)
        logger.info("Step 2/7: Detecting PDF type...")
        pdf_type, type_metadata = detect_pdf_type(pdf_path)
        logger.info(f"  Type: {pdf_type.upper()}")
        
        # Step 3: Extract text
        logger.info("Step 3/7: Extracting text...")
        pages, extraction_stats = extract_text(pdf_path, pdf_type)
        logger.info(f"  Extracted text from {len(pages)} pages")
        
        # Step 4: Clean text
        logger.info("Step 4/7: Cleaning text...")
        cleaned_pages = clean_pages(pages)
        total_chars = sum(p.char_count for p in cleaned_pages)
        logger.info(f"  Cleaned {total_chars:,} characters")
        
        # Step 5: Detect BRSR type (standalone vs embedded)
        logger.info("Step 5/7: Detecting BRSR type...")
        brsr_type, confidence, type_metadata = detect_brsr_type(pdf_path)
        result['brsr_type'] = brsr_type
        result['brsr_confidence'] = confidence
        logger.info(f"  BRSR Type: {brsr_type.upper()} (confidence: {confidence:.2f})")
        
        # Step 6: Create output directory
        logger.info("Step 6/7: Creating output directory...")
        output_path = create_brsr_output_directory(company_name, year)
        
        # Step 7: Process and export based on BRSR type
        logger.info("Step 7/7: Processing and exporting...")
        
        if brsr_type == 'standalone':
            # Process entire document as standalone BRSR
            logger.info("  Processing as STANDALONE BRSR...")
            
            # Export to DOCX
            docx_filename = format_brsr_output_filename(company_name, year, True, False, 'docx')
            docx_path = output_path / docx_filename
            docx_path = export_brsr_to_docx(
                cleaned_pages, output_path, company_name, year,
                is_standalone=True, is_from_annual=False
            )
            result['output_files'].append(str(docx_path))
            
            # Build hierarchical structure
            logger.info("  Building hierarchical structure...")
            full_text = "\n\n".join([p.text for p in cleaned_pages])
            # Use 'mdna' as section type for hierarchy building (works for BRSR too)
            hierarchy_builder = SectionHierarchyBuilder(section_type='mdna')
            full_report_hierarchy = hierarchy_builder.build_section_hierarchy(
                text=full_text,
                company=company_name,
                year=year,
                section_name="BRSR Report",
                start_page=1,
                end_page=len(cleaned_pages),
                confidence=confidence
            )
            
            # Export JSON
            json_filename = format_brsr_output_filename(company_name, year, True, False, 'json')
            json_path = output_path / json_filename
            hierarchy_builder.export_section_json(full_report_hierarchy, json_path)
            result['output_files'].append(str(json_path))
            logger.info(f"  Exported JSON to {json_path}")
            
        elif brsr_type == 'embedded':
            # Extract BRSR section from annual report
            logger.info("  Processing as EMBEDDED BRSR...")
            
            extraction_result = extract_brsr_from_annual_report(
                pdf_path=pdf_path,
                pages=pages,  # Use original pages with layout info
                output_dir=output_path,
                company_name=company_name,
                year=year
            )
            
            if extraction_result and extraction_result.get('success'):
                logger.info(f"  Extracted BRSR section: pages {extraction_result['start_page']}-{extraction_result['end_page']}")
                result['output_files'].append(extraction_result.get('docx_path'))
                result['extraction_metadata'] = extraction_result
            else:
                # Fallback: process entire document but mark as from annual report
                logger.warning("  Section extraction failed, processing entire document as fallback...")
                docx_filename = format_brsr_output_filename(company_name, year, False, True, 'docx')
                docx_path = output_path / docx_filename
                docx_path = export_brsr_to_docx(
                    cleaned_pages, output_path, company_name, year,
                    is_standalone=False, is_from_annual=True
                )
                result['output_files'].append(str(docx_path))
        else:
            # Unknown type - process as standalone but with lower confidence
            logger.warning("  BRSR type unknown, processing as standalone with low confidence...")
            docx_filename = format_brsr_output_filename(company_name, year, True, False, 'docx')
            docx_path = output_path / docx_filename
            docx_path = export_brsr_to_docx(
                cleaned_pages, output_path, company_name, year,
                is_standalone=True, is_from_annual=False
            )
            result['output_files'].append(str(docx_path))
        
        # Create metadata JSON
        metadata_filename = format_brsr_output_filename(company_name, year, brsr_type == 'standalone', brsr_type == 'embedded', 'metadata')
        metadata_path = output_path / metadata_filename
        metadata = {
            "company": company_name,
            "year": year,
            "processing_date": datetime.now().isoformat(),
            "pdf_info": pdf_info,
            "pdf_type": pdf_type,
            "brsr_type": brsr_type,
            "brsr_confidence": confidence,
            "brsr_type_metadata": type_metadata,
            "download_tier": download_tier,
            "extraction_summary": {
                "total_pages_in_pdf": pdf_info.get('pages', len(cleaned_pages)),
                "pages_processed": len(cleaned_pages),
                "pages_with_content": extraction_stats.get('pages_with_content', len(cleaned_pages)),
                "total_characters": total_chars,
                "extraction_method": pdf_type
            },
            "output_files": result['output_files']
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        result['output_files'].append(str(metadata_path))
        result['metadata'] = str(metadata_path)
        result['status'] = "success"
        
        logger.info(f"✓ Successfully processed {pdf_path.name}")
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        result['error'] = str(e)
        result['status'] = "failed"
    
    return result


def main():
    """Main entry point for BRSR processing workflow."""
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f"brsr_pipeline_{timestamp}.log"
    setup_logging(log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("BRSR Report Processing Pipeline")
    logger.info("="*80)
    logger.info(f"Financial years: {BRSR_FINANCIAL_YEARS}")
    logger.info(f"Download directory: {DOWNLOAD_BASE_DIR}")
    logger.info(f"Output directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Log file: {log_file}")
    
    # Step 1: Load company data
    logger.info("\nStep 1: Loading company data...")
    excel_path = BASE_DIR.parent / "NIFTY 500 firms.xlsx"
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.info("Please ensure 'NIFTY 500 firms.xlsx' is in the parent directory")
        return
    
    company_reader = read_companies(excel_path)
    companies_df = company_reader.get_companies_dataframe()
    logger.info(f"Loaded {len(companies_df)} companies")
    
    # Step 2: Download BRSR reports (if not already downloaded)
    logger.info("\nStep 2: Downloading BRSR reports...")
    download_manager = DownloadManager(
        output_base_dir=DOWNLOAD_BASE_DIR,
        max_workers=8
    )
    
    # Check which downloads are needed
    years = BRSR_FINANCIAL_YEARS
    logger.info(f"Processing {len(companies_df)} companies × {len(years)} years = {len(companies_df) * len(years)} downloads")
    
    # Batch download with resume capability
    download_summary = download_manager.batch_download(companies_df, years, resume=True)
    logger.info(f"Download summary: {download_summary}")
    
    # Step 3: Process downloaded PDFs
    logger.info("\nStep 3: Processing downloaded PDFs...")
    processing_results = []
    
    # Load download status to get download information
    download_status = download_manager.status.get('downloads', [])
    
    # Process each downloaded PDF
    for download in tqdm(download_status, desc="Processing BRSR reports", unit="file"):
        if not download.get('success'):
            continue  # Skip failed downloads
        
        file_path = download.get('file_path')
        if not file_path or not Path(file_path).exists():
            continue  # Skip if file doesn't exist
        
        pdf_path = Path(file_path)
        company_name = download.get('company_name', '')
        year = download.get('year', '')
        tier = download.get('tier')
        
        if not company_name or not year:
            continue
        
        # Process PDF
        result = process_brsr_pdf(pdf_path, company_name, year, tier)
        processing_results.append(result)
    
    # Step 4: Generate summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    
    successful = sum(1 for r in processing_results if r.get('status') == 'success')
    failed = sum(1 for r in processing_results if r.get('status') == 'failed')
    
    logger.info(f"Total PDFs processed: {len(processing_results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    # BRSR type breakdown
    standalone_count = sum(1 for r in processing_results if r.get('brsr_type') == 'standalone')
    embedded_count = sum(1 for r in processing_results if r.get('brsr_type') == 'embedded')
    unknown_count = sum(1 for r in processing_results if r.get('brsr_type') == 'unknown')
    
    logger.info(f"\nBRSR Type Breakdown:")
    logger.info(f"  Standalone: {standalone_count}")
    logger.info(f"  Embedded: {embedded_count}")
    logger.info(f"  Unknown: {unknown_count}")
    
    # Tier breakdown
    tier_breakdown = {}
    for r in processing_results:
        tier = r.get('download_tier', 'unknown')
        tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1
    
    logger.info(f"\nDownload Tier Breakdown:")
    for tier, count in tier_breakdown.items():
        logger.info(f"  {tier}: {count}")
    
    # Save summary
    summary_path = STATUS_DIR / f"processing_summary_{timestamp}.json"
    summary = {
        'processing_date': datetime.now().isoformat(),
        'download_summary': download_summary,
        'processing_results': processing_results,
        'statistics': {
            'total_processed': len(processing_results),
            'successful': successful,
            'failed': failed,
            'brsr_types': {
                'standalone': standalone_count,
                'embedded': embedded_count,
                'unknown': unknown_count
            },
            'tier_breakdown': tier_breakdown
        }
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n" + "="*80)
    logger.info("BRSR Pipeline completed!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

