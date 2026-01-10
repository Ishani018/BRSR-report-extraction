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
from downloaders.download_manager import format_filename


def setup_logging():
    """Setup logging configuration for test."""
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG to see full API responses
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
    
    # Test with DEBUG level to see full API responses
    logging.getLogger('downloaders.nse_downloader').setLevel(logging.DEBUG)
    
    # Test direct API call to see what endpoint format works
    logger.info("\nDEBUG: Testing API endpoint directly...")
    try:
        import requests
        from config.config import NSE_BASE_URL, NSE_API_ENDPOINT, NSE_HEADERS, NSE_API_TIMEOUT
        
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        
        # Visit annual reports page first
        annual_reports_page = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
        session.get(annual_reports_page, timeout=NSE_API_TIMEOUT)
        import time
        time.sleep(1)
        
        # Try the API call
        test_symbol = "RELIANCE"
        api_url = f"{NSE_API_ENDPOINT}?index=equities&symbol={test_symbol}"
        logger.info(f"Testing API URL: {api_url}")
        
        response = session.get(api_url, timeout=NSE_API_TIMEOUT)
        logger.info(f"API Response Status: {response.status_code}")
        logger.info(f"API Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                logger.info(f"API Response Type: {type(data)}")
                logger.info(f"API Response: {str(data)[:1000]}")
            except Exception as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.info(f"Response text (first 1000 chars): {response.text[:1000]}")
    except Exception as e:
        logger.error(f"Error in debug API test: {e}", exc_info=True)
    
    logger.info("\n" + "-" * 80)
    logger.info("Now testing with get_nse_report() function...")
    logger.info("-" * 80 + "\n")
    
    # Test with specific years to verify year filtering works
    test_years = ["2024-25", "2023-24", "2022-23", None]  # Try specific years, then latest
    
    success = False
    error = None
    downloaded_file = None
    
    for test_year in test_years:
        if test_year:
            # Use symbol-based naming convention with serial number at front (matches CSV/Excel format)
            company_name = "Reliance Industries Limited"
            symbol = "RELIANCE"  # Use symbol from Excel/CSV
            serial_number = 1  # Example serial number (from Excel row)
            # Use format_filename to get proper naming: {SerialNumber}_{SYMBOL}_BRSR_{Year}.pdf
            filename = format_filename(company_name, test_year, is_standalone=True, symbol=symbol, serial_number=serial_number)
            output_path = test_output_dir / filename.replace('.pdf', '_test.pdf')  # Add _test suffix
            logger.info(f"\nTesting with year: {test_year}")
            logger.info(f"Output file: {output_path.name} (serial: {serial_number}, symbol: {symbol})")
        else:
            company_name = "Reliance Industries Limited"
            symbol = "RELIANCE"
            serial_number = 1
            filename = format_filename(company_name, "latest", is_standalone=True, symbol=symbol, serial_number=serial_number)
            output_path = test_output_dir / filename.replace('.pdf', '_test.pdf')
            logger.info(f"\nTesting with latest report (year=None)")
            logger.info(f"Output file: {output_path.name} (serial: {serial_number}, symbol: {symbol})")
        
        success, error = get_nse_report(
            symbol="RELIANCE",
            output_path=output_path,
            year=test_year  # Test year filtering
        )
        
        if success:
            downloaded_file = output_path
            logger.info(f"✓ SUCCESS: PDF downloaded successfully for year: {test_year or 'latest'}")
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
            
            # Verify naming convention
            if test_year and test_year not in output_path.name:
                logger.warning(f"⚠ Warning: Year {test_year} not found in filename: {output_path.name}")
            
            return 0
        else:
            logger.warning(f"✗ Failed for year {test_year}: {error}")
            if test_year != test_years[-1]:  # Not the last one (None)
                logger.info("  Trying next year...")
                continue
    
    # If all failed
    logger.error("")
    logger.error("✗ FAILED: Could not download PDF for any year")
    logger.error(f"  Last error: {error}")
    logger.error("")
    logger.error("Troubleshooting:")
    logger.error("  1. Check your internet connection")
    logger.error("  2. Verify NSE website is accessible: https://www.nseindia.com")
    logger.error("  3. Check if RELIANCE symbol exists and has annual reports")
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

