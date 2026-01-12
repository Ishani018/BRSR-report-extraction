"""
Interactive Batch Download Script - Allows control over batch processing.
Process companies in batches of 10 with ability to resume and control progress.
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


def get_last_batch_info(status_csv_path: Path) -> dict:
    """Get information about last batch processed from status CSV."""
    import pandas as pd
    
    if not status_csv_path.exists():
        return {'last_batch': 0, 'companies_processed': 0}
    
    try:
        df = pd.read_csv(status_csv_path)
        if df.empty:
            return {'last_batch': 0, 'companies_processed': 0}
        
        # Count unique companies with successful downloads
        downloaded_companies = df[df['status'] == 'Downloaded'].groupby(['symbol', 'serial_number']).size()
        companies_processed = len(downloaded_companies)
        
        # Estimate batch number (assuming batch_size=10)
        last_batch = companies_processed // 10
        
        return {
            'last_batch': last_batch,
            'companies_processed': companies_processed
        }
    except Exception as e:
        logging.warning(f"Could not read status CSV: {e}")
        return {'last_batch': 0, 'companies_processed': 0}


def interactive_batch_download():
    """Interactive batch download with user control."""
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
    
    # Step 2: Check last batch info
    status_csv_path = STATUS_DIR / "download_status.csv"
    last_batch_info = get_last_batch_info(status_csv_path)
    logger.info(f"Last batch processed: {last_batch_info['last_batch']}")
    logger.info(f"Companies processed: {last_batch_info['companies_processed']}")
    logger.info("")
    
    # Step 3: Initialize download manager
    download_manager = DownloadManager(
        output_base_dir=DOWNLOAD_BASE_DIR,
        max_workers=8
    )
    
    # Step 4: Interactive batch selection
    print("\n" + "="*80)
    print("Batch Download Configuration")
    print("="*80)
    print("\nHow batch processing works:")
    print("  - Companies are processed in groups (batches)")
    print("  - Each company needs reports for 3 years (2022-23, 2023-24, 2024-25)")
    print("  - Batch 0 = first group, Batch 1 = second group, etc.")
    print("")
    
    # Calculate total companies and show example
    total_companies = len(companies_df)
    estimated_batches = (total_companies + 9) // 10  # Estimate with batch_size=10
    
    print(f"Total companies to process: {total_companies}")
    print(f"Example: With batch size 10, you'll have ~{estimated_batches} batches")
    print("")
    
    # Get batch size
    print("1. BATCH SIZE: How many companies to process in each batch?")
    print("   (e.g., 10 = process 10 companies at a time)")
    batch_size_input = input(f"   Enter batch size [default: 10]: ").strip()
    batch_size = int(batch_size_input) if batch_size_input.isdigit() else 10
    
    # Calculate total batches with the chosen batch size
    actual_total_batches = (total_companies + batch_size - 1) // batch_size
    print(f"   → This will create approximately {actual_total_batches} batches")
    print("")
    
    # Get starting batch
    suggested_start = last_batch_info['last_batch']
    print("2. START FROM BATCH: Which batch number do you want to start from?")
    print(f"   (0 = first batch, {suggested_start} = suggested resume point, {actual_total_batches-1} = last batch)")
    print(f"   Last processed: batch {suggested_start}")
    start_input = input(f"   Enter starting batch number [default: {suggested_start}]: ").strip()
    start_from_batch = int(start_input) if start_input.isdigit() else suggested_start
    
    if start_from_batch >= actual_total_batches:
        print(f"   ⚠ Warning: Starting batch {start_from_batch} exceeds total batches ({actual_total_batches-1})")
        max_batches = 0
    else:
        remaining_batches = actual_total_batches - start_from_batch
        max_batches = remaining_batches
        print(f"   → Will start from batch {start_from_batch}, {remaining_batches} batch(es) remaining")
    print("")
    
    # Get number of batches to process
    print("3. NUMBER OF BATCHES: How many batches do you want to process?")
    print("   (Enter a number like '5' to process 5 batches, or 'all' to process all remaining)")
    if max_batches > 0:
        print(f"   (Maximum available from batch {start_from_batch}: {max_batches} batch(es))")
    num_batches_input = input(f"   Enter number of batches [default: 1, 'all' for all remaining]: ").strip().lower()
    if num_batches_input == 'all' or num_batches_input == '':
        num_batches = None
        print(f"   → Will process all remaining batches")
    elif num_batches_input.isdigit():
        num_batches = int(num_batches_input)
        if num_batches > max_batches and max_batches > 0:
            print(f"   ⚠ Warning: Only {max_batches} batches available, limiting to that")
            num_batches = max_batches
        else:
            print(f"   → Will process {num_batches} batch(es)")
    else:
        num_batches = 1
        print(f"   → Will process 1 batch (default)")
    
    print("")
    print("="*80)
    print("Configuration Summary:")
    print(f"  📦 Batch size: {batch_size} companies per batch")
    print(f"  🎯 Starting from: Batch {start_from_batch}")
    if num_batches:
        total_companies_in_batches = min(num_batches * batch_size, total_companies - (start_from_batch * batch_size))
        print(f"  📊 Processing: {num_batches} batch(es) = ~{total_companies_in_batches} companies")
    else:
        total_companies_in_batches = total_companies - (start_from_batch * batch_size)
        print(f"  📊 Processing: All remaining batches = ~{total_companies_in_batches} companies")
    print(f"  📅 Financial years: {', '.join(BRSR_FINANCIAL_YEARS)}")
    print(f"  📄 Total download tasks: ~{total_companies_in_batches * len(BRSR_FINANCIAL_YEARS)} PDFs")
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
    
    # Step 5: Run batch download
    logger.info("")
    logger.info("Starting batch download...")
    if force_reload:
        logger.info("FORCE RELOAD MODE: All existing files will be re-downloaded")
    logger.info("")
    
    summary = download_manager.batch_download(
        companies_df=companies_df,
        years=BRSR_FINANCIAL_YEARS,
        resume=True,
        force_reload=force_reload,
        batch_size=batch_size,
        num_batches=num_batches,
        start_from_batch=start_from_batch
    )
    
    # Step 6: Show results
    logger.info("")
    logger.info("="*80)
    logger.info("Download Complete!")
    logger.info("="*80)
    logger.info(f"Batches processed: {summary['batches_processed']}")
    logger.info(f"Successful downloads: {summary['successful']}")
    logger.info(f"Failed downloads: {summary['failed']}")
    logger.info(f"Success rate: {summary['success_rate']:.1f}%")
    logger.info("")
    
    if summary.get('companies_processed_in_run', 0) < summary.get('total_companies_in_run', 0):
        remaining = summary.get('total_companies_in_run', 0) - summary.get('companies_processed_in_run', 0)
        logger.info(f"Remaining companies: {remaining}")
        logger.info(f"To continue, run again with start_from_batch={summary.get('next_batch_start', 0)}")
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

