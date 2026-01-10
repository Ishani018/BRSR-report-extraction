"""
Test NSE API Connection - Test script to verify NSE API download works.
Downloads BRSR report for RELIANCE symbol to test the connection.
"""
import sys
import logging
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config.config import DOWNLOAD_BASE_DIR
from downloaders.nse_downloader import get_nse_report, NSEDownloader


def setup_logging():
    """Setup logging configuration for test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def test_nse_connection():
    """Test NSE API connection by downloading RELIANCE BRSR report."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("NSE API Connection Test")
    logger.info("="*80)
    logger.info(f"Testing download for symbol: RELIANCE")
    logger.info("")
    
    # Create test output directory
    test_output_dir = DOWNLOAD_BASE_DIR / "test"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output path for the downloaded PDF
    output_path = test_output_dir / "RELIANCE_BRSR_test.pdf"
    
    logger.info(f"Output directory: {test_output_dir}")
    logger.info(f"Output file: {output_path.name}")
    logger.info("")
    
    # Test 1: Using get_nse_report function directly
    logger.info("Test 1: Using get_nse_report() function...")
    logger.info("-" * 80)
    
    success, error = get_nse_report(
        symbol="RELIANCE",
        output_path=output_path,
        year=None  # Get latest report
    )
    
    if success:
        logger.info("")
        logger.info("✓ SUCCESS: PDF downloaded successfully!")
        logger.info(f"  File saved to: {output_path}")
        
        # Check file size
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  File size: {file_size_mb:.2f} MB")
        
        # Verify it's a valid PDF by checking first bytes
        with open(output_path, 'rb') as f:
            first_bytes = f.read(4)
            if first_bytes == b'%PDF':
                logger.info("  ✓ File is a valid PDF")
            else:
                logger.warning(f"  ⚠ Warning: File may not be a valid PDF (first bytes: {first_bytes})")
        
        logger.info("")
        logger.info("="*80)
        logger.info("✓ NSE API Connection Test PASSED")
        logger.info("="*80)
        return 0
    else:
        logger.error("")
        logger.error("✗ FAILED: Could not download PDF")
        logger.error(f"  Error: {error}")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Check your internet connection")
        logger.error("  2. Verify NSE website is accessible: https://www.nseindia.com")
        logger.error("  3. Check if RELIANCE symbol exists and has BRSR reports")
        logger.error("  4. Verify the API endpoint is still valid")
        logger.error("")
        logger.error("="*80)
        logger.error("✗ NSE API Connection Test FAILED")
        logger.error("="*80)
        return 1


def test_nse_downloader_class():
    """Test using NSEDownloader class."""
    logger = logging.getLogger(__name__)
    
    logger.info("")
    logger.info("Test 2: Using NSEDownloader class...")
    logger.info("-" * 80)
    
    downloader = NSEDownloader(rate_limit_delay=1.5)
    output_path = DOWNLOAD_BASE_DIR / "test" / "RELIANCE_BRSR_class_test.pdf"
    
    success, error = downloader.download(
        symbol="RELIANCE",
        output_path=output_path,
        year=None
    )
    
    if success:
        logger.info("✓ SUCCESS: Downloaded using NSEDownloader class")
        logger.info(f"  File: {output_path}")
        return True
    else:
        logger.warning(f"✗ FAILED using class method: {error}")
        return False


if __name__ == "__main__":
    # Run main test
    exit_code = test_nse_connection()
    
    # Optionally test the class method too
    if exit_code == 0:
        test_nse_downloader_class()
    
    sys.exit(exit_code)

