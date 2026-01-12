"""
Script to validate all downloaded PDFs, delete corrupted files, and update status CSV.
Checks if PDFs can actually be opened/parsed (not just file size/header).
Deletes invalid/corrupted PDFs so they can be re-downloaded.
"""
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import DOWNLOAD_BASE_DIR, STATUS_DIR
from downloaders.download_manager import DownloadManager

# Try to import PDF libraries
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_pdf_file(pdf_path: Path) -> tuple:
    """
    Validate a PDF file by actually trying to open/parse it.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not pdf_path.exists():
        return False, "File does not exist"
    
    # Check file size (must be at least 1KB)
    file_size = pdf_path.stat().st_size
    if file_size < 1000:
        return False, f"File too small ({file_size} bytes)"
    
    # Check PDF header
    try:
        with open(pdf_path, 'rb') as f:
            first_bytes = f.read(4)
            if first_bytes != b'%PDF':
                return False, f"Invalid PDF header: {first_bytes}"
    except Exception as e:
        return False, f"Error reading file: {e}"
    
    # Try to actually open and parse the PDF using available libraries
    # Try pdfplumber first (more commonly used in this codebase)
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                if page_count == 0:
                    return False, "PDF has no pages"
                # Try to read first page to ensure it's parseable
                first_page = pdf.pages[0]
                _ = first_page.extract_text()  # Try to extract text
            return True, f"Valid PDF ({page_count} pages, {file_size / (1024*1024):.2f} MB)"
        except Exception as e:
            error_msg = str(e)
            # Check for common corruption errors
            if "Unexpected EOF" in error_msg or "invalid" in error_msg.lower():
                return False, f"Corrupted PDF: {error_msg}"
            return False, f"Error parsing PDF: {error_msg}"
    
    # Fallback to PyMuPDF if pdfplumber not available
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            if page_count == 0:
                doc.close()
                return False, "PDF has no pages"
            # Try to read first page
            first_page = doc[0]
            _ = first_page.get_text()  # Try to extract text
            doc.close()
            return True, f"Valid PDF ({page_count} pages, {file_size / (1024*1024):.2f} MB)"
        except Exception as e:
            error_msg = str(e)
            if "Unexpected EOF" in error_msg or "invalid" in error_msg.lower():
                return False, f"Corrupted PDF: {error_msg}"
            return False, f"Error parsing PDF: {error_msg}"
    
    # If no PDF library available, just check header (not ideal)
    return True, f"Header valid but not fully validated (no PDF library available)"


def validate_all_downloads():
    """Validate all downloaded PDFs and update status CSV."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("PDF Validation Script")
    logger.info("="*80)
    
    if not HAS_PDFPLUMBER and not HAS_PYMUPDF:
        logger.error("ERROR: No PDF library available (pdfplumber or PyMuPDF required)")
        logger.error("Please install: pip install pdfplumber")
        return
    
    # Load download manager to access status CSV
    download_manager = DownloadManager(output_base_dir=DOWNLOAD_BASE_DIR)
    status_csv_path = download_manager.status_csv_path
    
    if not status_csv_path.exists():
        logger.error(f"Status CSV not found: {status_csv_path}")
        logger.info("No downloads to validate")
        return
    
    # Load status DataFrame
    status_df = download_manager.load_status_csv()
    
    if status_df.empty:
        logger.info("Status CSV is empty - no downloads to validate")
        return
    
    logger.info(f"Loaded status CSV with {len(status_df)} entries")
    
    # Filter for "Downloaded" status
    downloaded_df = status_df[status_df['status'] == 'Downloaded'].copy()
    logger.info(f"Found {len(downloaded_df)} files marked as 'Downloaded'")
    
    if downloaded_df.empty:
        logger.info("No files marked as 'Downloaded' to validate")
        return
    
    # Validate each downloaded file
    logger.info("\nValidating PDF files...")
    validation_results = []
    valid_count = 0
    invalid_count = 0
    
    for idx, row in downloaded_df.iterrows():
        file_path_str = row.get('file_path', '')
        company_name = row.get('company_name', '')
        symbol = row.get('symbol', '')
        year = row.get('year', '')
        
        if not file_path_str or pd.isna(file_path_str):
            continue
        
        file_path = Path(str(file_path_str).strip())
        
        logger.info(f"Validating: {file_path.name} ({company_name} - {year})")
        
        is_valid, message = validate_pdf_file(file_path)
        
        validation_results.append({
            'index': idx,
            'file_path': str(file_path),
            'company_name': company_name,
            'symbol': symbol,
            'year': year,
            'is_valid': is_valid,
            'message': message
        })
        
        if is_valid:
            valid_count += 1
            logger.info(f"  ✓ Valid: {message}")
        else:
            invalid_count += 1
            logger.warning(f"  ✗ Invalid: {message}")
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total files validated: {len(validation_results)}")
    logger.info(f"Valid: {valid_count}")
    logger.info(f"Invalid: {invalid_count}")
    
    # Update status CSV and delete invalid files
    if invalid_count > 0:
        logger.info("\nUpdating status CSV and deleting invalid files...")
        updated_count = 0
        deleted_count = 0
        
        for result in validation_results:
            if not result['is_valid']:
                idx = result['index']
                file_path = Path(result['file_path'])
                
                # Delete the corrupted/invalid file
                try:
                    if file_path.exists():
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"  Deleted: {file_path.name}")
                    else:
                        logger.info(f"  File already missing: {file_path.name}")
                except Exception as e:
                    logger.warning(f"  Error deleting {file_path.name}: {e}")
                
                # Update status to 'Failed' with validation error
                download_manager.status_df.at[idx, 'status'] = 'Failed'
                error_msg = f"Validation failed: {result['message']}"
                download_manager.status_df.at[idx, 'error'] = error_msg
                # Clear file_path since file was deleted
                download_manager.status_df.at[idx, 'file_path'] = ''
                updated_count += 1
                logger.info(f"  Updated CSV: {file_path.name} -> Failed")
        
        # Save updated status CSV
        download_manager.save_status_csv()
        logger.info(f"\nDeleted {deleted_count} invalid/corrupted PDF files")
        logger.info(f"Updated {updated_count} entries in status CSV")
        logger.info(f"Status CSV saved to: {status_csv_path}")
        logger.info(f"\nNote: Deleted files can now be re-downloaded by running the download process again")
    else:
        logger.info("\nAll files are valid - no updates needed")
    
    # Save validation report
    report_path = STATUS_DIR / f"pdf_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    validation_df = pd.DataFrame(validation_results)
    validation_df.to_csv(report_path, index=False)
    logger.info(f"\nValidation report saved to: {report_path}")
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    validate_all_downloads()
