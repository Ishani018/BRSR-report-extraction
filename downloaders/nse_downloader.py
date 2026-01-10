"""
NSE API Downloader - Tier 1 (Primary) download source using NSE corporate filings API.
"""
import logging
import requests
import time
from pathlib import Path
from typing import Optional, Tuple

from config.config import (
    NSE_BASE_URL,
    NSE_API_ENDPOINT,
    NSE_ARCHIVES_BASE_URL,
    NSE_API_TIMEOUT,
    NSE_DOWNLOAD_TIMEOUT,
    NSE_HEADERS
)

logger = logging.getLogger(__name__)


def get_nse_report(symbol: str, output_path: Path, year: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Download BRSR report from NSE API for a given symbol.
    
    Implementation based on reference code provided. This function:
    1. Mimics a browser to avoid 403 Forbidden
    2. Initializes session by visiting NSE homepage to get cookies (essential)
    3. Calls Corporate Filings API with preserved typos ('bussiness', 'sustainabilitiy')
    4. Extracts fileName from response
    5. Downloads PDF using same session (preserves cookies)
    
    Args:
        symbol: NSE symbol (e.g., 'RELIANCE', 'TCS')
        output_path: Path where PDF should be saved
        year: Optional financial year to filter (format: '2022-23'). If None, gets latest report.
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
        - (True, None) on success
        - (False, error_message) on failure
    """
    # Mimic a browser to avoid 403 Forbidden
    headers = NSE_HEADERS.copy()
    
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        # 1. Initialize Session (Essential: Visit homepage to get cookies)
        # Without this, API returns 403 Forbidden
        logger.debug(f"Initializing NSE session for symbol: {symbol}")
        session.get(NSE_BASE_URL, timeout=NSE_API_TIMEOUT)
        time.sleep(0.5)  # Small delay to ensure cookies are set
        
        # 2. Call Corporate Filings API
        # Note: 'bussiness' and 'sustainabilitiy' typos are native to NSE API - must be preserved
        api_url = f"{NSE_API_ENDPOINT}?index=equities&symbol={symbol}"
        logger.debug(f"Calling NSE API: {api_url}")
        
        response = session.get(api_url, timeout=NSE_API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                # Get the latest report (or filter by year if provided)
                # For now, we get the first result (latest)
                # TODO: Add year filtering if API response includes date metadata
                file_name = data[0].get('fileName')
                
                if file_name:
                    # 3. Construct download URL and download PDF
                    download_url = f"{NSE_ARCHIVES_BASE_URL}/{file_name}"
                    logger.info(f"Downloading PDF from: {download_url}")
                    
                    # Use same session (preserves cookies)
                    pdf_response = session.get(download_url, timeout=NSE_DOWNLOAD_TIMEOUT)
                    
                    if pdf_response.status_code == 200:
                        # Save PDF to output path
                        output_path = Path(output_path)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_path, 'wb') as f:
                            f.write(pdf_response.content)
                        
                        file_size = len(pdf_response.content) / (1024 * 1024)  # MB
                        logger.info(f"Successfully downloaded {output_path.name} ({file_size:.2f} MB)")
                        return True, None
                    else:
                        error_msg = f"Failed to download PDF: HTTP {pdf_response.status_code}"
                        logger.warning(f"{symbol}: {error_msg}")
                        return False, error_msg
                else:
                    error_msg = "No fileName found in API response"
                    logger.warning(f"{symbol}: {error_msg}")
                    return False, error_msg
            else:
                error_msg = f"No reports found for symbol {symbol}"
                logger.info(f"{symbol}: {error_msg}")
                return False, error_msg
        elif response.status_code == 403:
            error_msg = "403 Forbidden - Session may have expired or rate limited"
            logger.warning(f"{symbol}: {error_msg}")
            return False, error_msg
        else:
            error_msg = f"API returned status code {response.status_code}"
            logger.warning(f"{symbol}: {error_msg}")
            return False, error_msg
            
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout while accessing NSE API: {e}"
        logger.error(f"{symbol}: {error_msg}")
        return False, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error: {e}"
        logger.error(f"{symbol}: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(f"{symbol}: {error_msg}", exc_info=True)
        return False, error_msg


class NSEDownloader:
    """
    NSE API Downloader class with additional functionality.
    """
    
    def __init__(self, rate_limit_delay: float = 1.5):
        """
        Initialize NSE Downloader.
        
        Args:
            rate_limit_delay: Delay in seconds between requests to avoid rate limiting
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def download(self, symbol: str, output_path: Path, year: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Download BRSR report with rate limiting.
        
        Args:
            symbol: NSE symbol
            output_path: Path where PDF should be saved
            year: Optional financial year filter
        
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        # Rate limiting: ensure minimum delay between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        
        return get_nse_report(symbol, output_path, year)
    
    def download_with_retry(
        self,
        symbol: str,
        output_path: Path,
        year: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Download BRSR report with retry logic.
        
        Args:
            symbol: NSE symbol
            output_path: Path where PDF should be saved
            year: Optional financial year filter
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
        
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        for attempt in range(max_retries):
            success, error = self.download(symbol, output_path, year)
            
            if success:
                return True, None
            
            # If 403 Forbidden, session may have expired - retry after delay
            if attempt < max_retries - 1 and "403" in (error or ""):
                logger.info(f"Retry {attempt + 1}/{max_retries} for {symbol} after {retry_delay}s delay")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                break
        
        return False, error or "Max retries exceeded"


if __name__ == "__main__":
    # Test the downloader
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with a known symbol (e.g., TCS, RELIANCE)
    test_symbol = sys.argv[1] if len(sys.argv) > 1 else "TCS"
    test_output = Path(__file__).parent.parent / "downloads" / "test" / f"{test_symbol}_BRSR_test.pdf"
    
    logger.info(f"Testing NSE downloader with symbol: {test_symbol}")
    success, error = get_nse_report(test_symbol, test_output)
    
    if success:
        logger.info(f"✓ Successfully downloaded to: {test_output}")
        sys.exit(0)
    else:
        logger.error(f"✗ Download failed: {error}")
        sys.exit(1)

