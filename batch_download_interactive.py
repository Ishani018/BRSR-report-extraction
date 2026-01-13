"""
Interactive Batch Download Script - Allows control over batch processing.
Process companies starting from a specific company number.
"""
import sys
import logging
from pathlib import Path
from typing import Optional

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config.config import DOWNLOAD_BASE_DIR, BRSR_FINANCIAL_YEARS, STATUS_DIR
from data.company_reader import read_companies
from downloaders.download_manager import DownloadManager


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_next_company_number(status_csv_path: Path, companies_df) -> int:
    """Get the next company number to process based on status CSV."""
    import pandas as pd
    
    if not status_csv_path.exists():
        return 0
    
    try:
        df = pd.read_csv(status_csv_path)
        if df.empty:
            return 0
        
        # Get the highest serial number that has at least one successful download
        downloaded = df[df['status'] == 'Downloaded']
        if downloaded.empty:
            return 0
        
        # Get unique serial numbers with downloads
        if 'serial_number' in downloaded.columns:
            serials = downloaded['serial_number'].dropna()
            if not serials.empty:
                max_serial = serials.astype(int).max()
                # Find the index of this company in the DataFrame
                if 'serial_number' in companies_df.columns:
                    matching_idx = companies_df[companies_df['serial_number'] == max_serial].index
                    if len(matching_idx) > 0:
                        # Return the next company index
                        return int(matching_idx[0]) + 1
        
        # Fallback: count unique companies
        unique_companies = downloaded.groupby(['symbol', 'serial_number']).size()
        return min(len(unique_companies), len(companies_df))
    except Exception as e:
        logging.warning(f"Could not read status CSV: {e}")
        return 0


def interactive_batch_download():
    """Interactive batch download with user control using company numbers."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("Interactive Batch Download - BRSR Reports")
    logger.info("="*80)
    logger.info("")
    
    # Step 1: Load company data
    logger.info("Step 1: Loading company data...")
    excel_path = BASE_DIR.parent / "NIFTY 500 firms.xlsx"
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.info("Please ensure 'NIFTY 500 firms.xlsx' is in the parent directory")
        return
    
    company_reader = read_companies(excel_path)
    companies_df = company_reader.get_companies_dataframe()
    logger.info(f"Loaded {len(companies_df)} companies")
    logger.info("")
    
    # Step 2: Check status and suggest starting point
    status_csv_path = STATUS_DIR / "download_status.csv"
    next_company_num = get_next_company_number(status_csv_path, companies_df)
    logger.info(f"Suggested starting company number: {next_company_num}")
    logger.info("")
    
    # Step 3: Initialize download manager
    download_manager = DownloadManager(
        output_base_dir=DOWNLOAD_BASE_DIR,
        max_workers=8
    )
    
    # Step 4: Interactive configuration using company numbers
    print("\n" + "="*80)
    print("Download Configuration")
    print("="*80)
    print("\nCompanies are numbered from 0 to " + str(len(companies_df) - 1))
    print("  - Company 0 = first company in the list")
    print(f"  - Company {len(companies_df) - 1} = last company in the list")
    print("")
    
    # Show some example companies
    print("Example companies:")
    for i in range(min(5, len(companies_df))):
        row = companies_df.iloc[i]
        company_name = row.get('company_name', 'N/A')
        symbol = row.get('symbol', 'N/A')
        serial = row.get('serial_number', i)
        print(f"  Company {i}: {company_name} ({symbol}, Serial: {serial})")
    if len(companies_df) > 5:
        print("  ...")
    print("")
    
    # Get starting company number
    print("1. START FROM COMPANY: Which company number do you want to start from?")
    print(f"   (0 = first company, {next_company_num} = suggested resume point, {len(companies_df) - 1} = last company)")
    start_input = input(f"   Enter starting company number [default: {next_company_num}]: ").strip()
    
    try:
        start_from_company = int(start_input) if start_input else next_company_num
    except ValueError:
        print(f"   Invalid input, using default: {next_company_num}")
        start_from_company = next_company_num
    
    if start_from_company < 0:
        start_from_company = 0
    if start_from_company >= len(companies_df):
        print(f"   ⚠ Warning: Company number {start_from_company} exceeds total companies ({len(companies_df)}), using {len(companies_df) - 1}")
        start_from_company = len(companies_df) - 1
    
    # Show which company we're starting from
    start_row = companies_df.iloc[start_from_company]
    start_company_name = start_row.get('company_name', 'N/A')
    start_symbol = start_row.get('symbol', 'N/A')
    start_serial = start_row.get('serial_number', start_from_company)
    print(f"   → Starting from: Company {start_from_company} = {start_company_name} ({start_symbol}, Serial: {start_serial})")
    print("")
    
    # Get number of companies to process
    remaining_companies = len(companies_df) - start_from_company
    print("2. NUMBER OF COMPANIES: How many companies do you want to process?")
    print(f"   (Enter a number like '10' to process 10 companies, or 'all' to process all remaining)")
    print(f"   (Maximum available from company {start_from_company}: {remaining_companies} companies)")
    num_companies_input = input(f"   Enter number of companies [default: 10, 'all' for all remaining]: ").strip().lower()
    
    if num_companies_input == 'all' or num_companies_input == '':
        num_companies = None
        print(f"   → Will process all {remaining_companies} remaining companies")
    elif num_companies_input.isdigit():
        num_companies = int(num_companies_input)
        if num_companies > remaining_companies:
            print(f"   ⚠ Warning: Only {remaining_companies} companies available, limiting to that")
            num_companies = remaining_companies
        else:
            print(f"   → Will process {num_companies} companies")
    else:
        num_companies = 10
        print(f"   → Will process 10 companies (default)")
    
    print("")
    
    # Get batch size (for internal processing)
    print("3. BATCH SIZE (Internal): How many companies to process in each internal batch?")
    print("   (This is for internal processing - smaller batches = more control, larger = faster)")
    batch_size_input = input(f"   Enter batch size [default: 10]: ").strip()
    batch_size = int(batch_size_input) if batch_size_input.isdigit() else 10
    print(f"   → Using batch size: {batch_size}")
    print("")
    
    # Calculate which companies will be processed
    if num_companies:
        end_company = min(start_from_company + num_companies, len(companies_df))
        companies_to_process = num_companies
    else:
        end_company = len(companies_df)
        companies_to_process = remaining_companies
    
    # Show end company info
    if end_company < len(companies_df):
        end_row = companies_df.iloc[end_company - 1]
        end_company_name = end_row.get('company_name', 'N/A')
        end_symbol = end_row.get('symbol', 'N/A')
        print("="*80)
        print("Configuration Summary:")
        print(f"  🏢 Starting from: Company {start_from_company} = {start_company_name} ({start_symbol})")
        if end_company < len(companies_df):
            print(f"  🏢 Ending at: Company {end_company - 1} = {end_company_name} ({end_symbol})")
        print(f"  📊 Processing: {companies_to_process} companies")
        print(f"  📦 Internal batch size: {batch_size} companies per batch")
        print(f"  📅 Financial years: {', '.join(BRSR_FINANCIAL_YEARS)}")
        print(f"  📄 Total download tasks: ~{companies_to_process * len(BRSR_FINANCIAL_YEARS)} PDFs")
    else:
        print("="*80)
        print("Configuration Summary:")
        print(f"  🏢 Starting from: Company {start_from_company} = {start_company_name} ({start_symbol})")
        print(f"  🏢 Ending at: Last company (Company {len(companies_df) - 1})")
        print(f"  📊 Processing: {companies_to_process} companies")
        print(f"  📦 Internal batch size: {batch_size} companies per batch")
        print(f"  📅 Financial years: {', '.join(BRSR_FINANCIAL_YEARS)}")
        print(f"  📄 Total download tasks: ~{companies_to_process * len(BRSR_FINANCIAL_YEARS)} PDFs")
    print("="*80)
    
    # Ask about force reload
    print("")
    print("4. FORCE RELOAD: Re-download all files even if already downloaded?")
    print("   (This will delete existing files and download fresh copies)")
    force_reload_input = input("   Force reload? [y/N]: ").strip().lower()
    force_reload = force_reload_input == 'y'
    if force_reload:
        print("   → Will re-download ALL files (existing files will be replaced)")
    else:
        print("   → Will skip files that are already downloaded (default)")
    
    confirm = input("\nProceed with download? [y/N]: ").strip().lower()
    if confirm != 'y':
        logger.info("Download cancelled by user.")
        return
    
    # Step 5: Convert company numbers to batch parameters for internal processing
    # We need to slice the DataFrame and convert to batch-based processing
    # Create a subset DataFrame starting from start_from_company
    companies_subset = companies_df.iloc[start_from_company:end_company].copy()
    
    # Calculate batch parameters
    # start_from_batch = 0 (we're starting from the beginning of our subset)
    # num_batches = calculate based on num_companies and batch_size
    if num_companies:
        total_companies_in_subset = min(num_companies, len(companies_subset))
        num_batches = (total_companies_in_subset + batch_size - 1) // batch_size
    else:
        num_batches = None  # Process all
    
    logger.info("")
    logger.info("Starting download...")
    logger.info(f"Processing companies {start_from_company} to {end_company - 1 if end_company <= len(companies_df) else len(companies_df) - 1}")
    if force_reload:
        logger.info("FORCE RELOAD MODE: All existing files will be re-downloaded")
    logger.info("")
    
    # Step 6: Run batch download with the subset
    summary = download_manager.batch_download(
        companies_df=companies_subset,
        years=BRSR_FINANCIAL_YEARS,
        resume=True,
        force_reload=force_reload,
        batch_size=batch_size,
        num_batches=num_batches,
        start_from_batch=0  # Always start from beginning of subset
    )
    
    # Step 7: Show results
    logger.info("")
    logger.info("="*80)
    logger.info("Download Complete!")
    logger.info("="*80)
    logger.info(f"Companies processed: {summary.get('companies_processed_in_run', 0)}")
    logger.info(f"Successful downloads: {summary['successful']}")
    logger.info(f"Failed downloads: {summary['failed']}")
    logger.info(f"Success rate: {summary['success_rate']:.1f}%")
    logger.info("")
    
    # Calculate next company number for resuming
    if num_companies:
        next_company_num = start_from_company + companies_to_process
    else:
        next_company_num = len(companies_df)
    
    if next_company_num < len(companies_df):
        logger.info(f"To continue, run again with starting company number: {next_company_num}")
        logger.info(f"  (This will start from: {companies_df.iloc[next_company_num].get('company_name', 'N/A')})")
    else:
        logger.info("All companies processed!")
    
    logger.info(f"Status CSV: {summary['status_csv_path']}")


if __name__ == "__main__":
    try:
        interactive_batch_download()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
