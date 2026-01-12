"""
Script to delete all downloaded PDFs and reset download status.

This script deletes all PDFs from the downloads directory and resets the status CSV,
since old downloads were annual reports (not BRSR reports).

WARNING: This is a destructive operation. All downloaded PDFs will be deleted.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import config
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

import logging
import pandas as pd
import shutil

from config.config import DOWNLOAD_BASE_DIR, STATUS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Status CSV filename (actual file is download_status.csv, not .json)
DOWNLOAD_STATUS_FILE = "download_status.csv"


def delete_all_downloads():
    """
    Delete all downloaded PDFs and reset status CSV.
    """
    download_dir = Path(DOWNLOAD_BASE_DIR)
    status_csv_path = Path(STATUS_DIR) / DOWNLOAD_STATUS_FILE
    
    if not download_dir.exists():
        logger.warning(f"Download directory does not exist: {download_dir}")
        return
    
    # Count files before deletion
    pdf_count = 0
    total_size = 0
    
    for pdf_file in download_dir.rglob("*.pdf"):
        if pdf_file.is_file():
            pdf_count += 1
            total_size += pdf_file.stat().st_size
    
    if pdf_count == 0:
        logger.info("No PDF files found to delete")
    else:
        logger.info(f"Found {pdf_count} PDF files ({total_size / (1024*1024):.2f} MB)")
        logger.warning("DELETING ALL DOWNLOADED PDFS...")
        
        # Delete all PDF files
        deleted_count = 0
        for pdf_file in download_dir.rglob("*.pdf"):
            if pdf_file.is_file():
                try:
                    pdf_file.unlink()
                    deleted_count += 1
                    if deleted_count % 100 == 0:
                        logger.info(f"Deleted {deleted_count} files...")
                except Exception as e:
                    logger.error(f"Error deleting {pdf_file}: {e}")
        
        logger.info(f"Deleted {deleted_count} PDF files")
        
        # Also delete any temporary files
        tmp_count = 0
        for tmp_file in download_dir.rglob("*.tmp"):
            if tmp_file.is_file():
                try:
                    tmp_file.unlink()
                    tmp_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {tmp_file}: {e}")
        
        if tmp_count > 0:
            logger.info(f"Deleted {tmp_count} temporary files")
    
    # Reset status CSV
    if status_csv_path.exists():
        logger.info(f"Resetting status CSV: {status_csv_path}")
        try:
            # Backup existing status CSV
            backup_path = status_csv_path.with_suffix('.csv.backup')
            shutil.copy2(status_csv_path, backup_path)
            logger.info(f"Backed up status CSV to: {backup_path}")
            
            # Create empty status CSV with correct columns (matching actual CSV structure)
            empty_df = pd.DataFrame(columns=[
                'serial_number', 'company_name', 'symbol', 'year',
                'status', 'error', 'file_path', 'timestamp', 'report_type'
            ])
            empty_df.to_csv(status_csv_path, index=False)
            logger.info("Status CSV reset to empty")
        except Exception as e:
            logger.error(f"Error resetting status CSV: {e}")
    else:
        logger.info("Status CSV does not exist, skipping reset")
    
    logger.info("Cleanup complete!")


if __name__ == "__main__":
    print("=" * 60)
    print("WARNING: This will delete ALL downloaded PDFs")
    print("=" * 60)
    response = input("Are you sure you want to continue? (yes/no): ")
    
    if response.lower() == 'yes':
        delete_all_downloads()
    else:
        print("Cancelled.")
