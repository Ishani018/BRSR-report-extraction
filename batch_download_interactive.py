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
    
    # Get batch size
    batch_size_input = input(f"Batch size (companies per batch) [default: 10]: ").strip()
    batch_size = int(batch_size_input) if batch_size_input.isdigit() else 10
    
    # Get starting batch
    suggested_start = last_batch_info['last_batch']
    start_input = input(f"Start from batch [default: {suggested_start}]: ").strip()
    start_from_batch = int(start_input) if start_input.isdigit() else suggested_start
    
    # Get number of batches to process
    num_batches_input = input(f"Number of batches to process [default: 1, 'all' for all remaining]: ").strip().lower()
    if num_batches_input == 'all' or num_batches_input == '':
        num_batches = None
    elif num_batches_input.isdigit():
        num_batches = int(num_batches_input)
    else:
        num_batches = 1
    
    print("")
    print("="*80)
    print("Configuration Summary:")
    print(f"  Batch size: {batch_size} companies per batch")
    print(f"  Start from batch: {start_from_batch}")
    print(f"  Number of batches: {num_batches if num_batches else 'All remaining'}")
    print(f"  Financial years: {', '.join(BRSR_FINANCIAL_YEARS)}")
    print("="*80)
    
    confirm = input("\nProceed with download? [y/N]: ").strip().lower()
    if confirm != 'y':
        logger.info("Download cancelled by user.")
        return
    
    # Step 5: Run batch download
    logger.info("")
    logger.info("Starting batch download...")
    logger.info("")
    
    summary = download_manager.batch_download(
        companies_df=companies_df,
        years=BRSR_FINANCIAL_YEARS,
        resume=True,
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
    
    if summary['companies_processed'] < summary['total_companies']:
        logger.info(f"Remaining companies: {summary['total_companies'] - summary['companies_processed']}")
        logger.info(f"To continue, run again with start_from_batch={summary['next_batch_start']}")
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

