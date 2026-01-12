"""
Test script to extract BRSR reports for just the first company.
This is a test script - it doesn't modify the main code.
"""
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    BRSR_FINANCIAL_YEARS,
    DOWNLOAD_BASE_DIR,
    OUTPUT_BASE_DIR,
    STATUS_DIR,
    LOGS_DIR
)
from data.company_reader import read_companies
from brsr_main import process_brsr_pdf, setup_logging


def test_first_company():
    """Extract BRSR reports for the first company only."""
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f"test_first_company_{timestamp}.log"
    setup_logging(log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("TEST: BRSR Report Extraction - First Company Only")
    logger.info("="*80)
    logger.info(f"Financial years: {BRSR_FINANCIAL_YEARS}")
    logger.info(f"Download directory: {DOWNLOAD_BASE_DIR}")
    logger.info(f"Output directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Log file: {log_file}")
    
    # Step 1: Load company data and get first company
    logger.info("\nStep 1: Loading company data...")
    base_dir = Path(__file__).parent.parent
    excel_path = base_dir.parent / "NIFTY 500 firms.xlsx"
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.info("Please ensure 'NIFTY 500 firms.xlsx' is in the parent directory")
        return
    
    company_reader = read_companies(excel_path)
    companies_df = company_reader.get_companies_dataframe()
    logger.info(f"Loaded {len(companies_df)} companies")
    
    # Get first company
    if companies_df.empty:
        logger.error("No companies found in Excel file")
        return
    
    first_company_df = companies_df.iloc[[0]]  # Get first company as DataFrame
    first_company = first_company_df.iloc[0]
    company_name = first_company['company_name']
    symbol = first_company['symbol']
    
    logger.info(f"\nSelected first company: {company_name} ({symbol})")
    logger.info(f"Serial number: {first_company.get('serial_number', 'N/A')}")
    
    # Step 2: Find existing PDFs for first company (skip download)
    logger.info("\nStep 2: Finding existing PDFs for first company (skipping download)...")
    
    serial_number = first_company.get('serial_number')
    symbol = first_company['symbol']
    
    # Build company folder path: {serial_number}_{SYMBOL}
    if serial_number is not None:
        company_folder = f"{serial_number}_{symbol.upper()}"
    else:
        company_folder = symbol.upper()
    
    company_dir = DOWNLOAD_BASE_DIR / company_folder
    logger.info(f"Looking for PDFs in: {company_dir}")
    
    # Find all PDFs for this company across all years
    existing_pdfs = []
    years = BRSR_FINANCIAL_YEARS
    
    for year in years:
        year_dir = company_dir / year
        if year_dir.exists():
            pdf_files = list(year_dir.glob("*.pdf"))
            for pdf_path in pdf_files:
                existing_pdfs.append({
                    'pdf_path': pdf_path,
                    'year': year,
                    'company_name': company_name
                })
                logger.info(f"  Found: {pdf_path.name} (Year: {year})")
    
    if not existing_pdfs:
        logger.warning(f"No existing PDFs found for {company_name} in {company_dir}")
        logger.info("Skipping processing - no files to process")
        return
    
    logger.info(f"Found {len(existing_pdfs)} existing PDF(s) for {company_name}")
    
    # Step 3: Process existing PDFs for first company only
    logger.info("\nStep 3: Processing existing PDFs for first company...")
    processing_results = []
    
    # Process each existing PDF
    for pdf_info in tqdm(existing_pdfs, 
                        desc=f"Processing BRSR reports for {company_name}", 
                        unit="file"):
        pdf_path = pdf_info['pdf_path']
        year = pdf_info['year']
        
        if not pdf_path.exists():
            logger.warning(f"File not found: {pdf_path}")
            continue
        
        # Note: tier/download_tier is not stored, so using None
        tier = None
        
        # Get company information for naming convention
        naming_convention = first_company.get('naming_convention')
        symbol = first_company.get('symbol')
        serial_number = first_company.get('serial_number')
        
        # Process PDF
        logger.info(f"\nProcessing: {pdf_path.name} (Year: {year})")
        result = process_brsr_pdf(
            pdf_path, company_name, year, tier,
            naming_convention=naming_convention,
            symbol=symbol,
            serial_number=serial_number
        )
        processing_results.append(result)
    
    # Step 4: Generate summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    
    successful = sum(1 for r in processing_results if r.get('status') == 'success')
    failed = sum(1 for r in processing_results if r.get('status') == 'failed')
    
    logger.info(f"Company: {company_name} ({symbol})")
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
    
    # Show output files
    logger.info(f"\nOutput files:")
    for result in processing_results:
        if result.get('status') == 'success':
            logger.info(f"  Year {result.get('year')}:")
            for output_file in result.get('output_files', []):
                logger.info(f"    - {Path(output_file).name}")
    
    # Save summary
    summary_path = STATUS_DIR / f"test_first_company_summary_{timestamp}.json"
    
    # Convert numpy/pandas types to native Python types for JSON serialization
    serial_num = first_company.get('serial_number')
    if pd.notna(serial_num):
        serial_num = int(serial_num)  # Convert numpy int64 to Python int
    else:
        serial_num = None
    
    summary = {
        'test_date': datetime.now().isoformat(),
        'company': {
            'name': company_name,
            'symbol': symbol,
            'serial_number': serial_num
        },
        'existing_pdfs_count': len(existing_pdfs),
        'processing_results': processing_results,
        'statistics': {
            'total_processed': len(processing_results),
            'successful': successful,
            'failed': failed,
            'brsr_types': {
                'standalone': standalone_count,
                'embedded': embedded_count,
                'unknown': unknown_count
            }
        }
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETED!")
    logger.info("="*80)
    
    return summary


if __name__ == "__main__":
    test_first_company()
