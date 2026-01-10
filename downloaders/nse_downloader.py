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
    NSE_HEADERS,
    NSE_RATE_LIMIT_DELAY
)

logger = logging.getLogger(__name__)


def get_nse_report(symbol: str, output_path: Path, year: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Download Annual Report (containing BRSR) from NSE API for a given symbol.
    
    Uses the stable Annual Reports API endpoint. This function:
    1. Mimics a browser to avoid 403 Forbidden
    2. Initializes session by visiting NSE homepage to get cookies (essential)
    3. Calls Annual Reports API endpoint
    4. Extracts fileName from response (handles both full URLs and filenames)
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
        # 1. Initialize Session (Essential: Visit annual reports page to get proper cookies)
        # Visit the actual annual reports page first to match the Referer header
        # Without this, API may return 403 Forbidden or empty results
        logger.debug(f"Initializing NSE session for symbol: {symbol}")
        annual_reports_page = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
        session.get(annual_reports_page, timeout=NSE_API_TIMEOUT)
        time.sleep(0.5)  # Small delay to ensure cookies are set
        # Also visit homepage to ensure all cookies are set
        session.get(NSE_BASE_URL, timeout=NSE_API_TIMEOUT)
        time.sleep(0.5)  # Additional delay
        
        # 2. Call Annual Reports API
        # Try different API parameter formats - NSE API may need specific format
        # Format 1: index=equities&symbol={symbol} (standard format)
        api_url = f"{NSE_API_ENDPOINT}?index=equities&symbol={symbol}"
        logger.debug(f"Calling NSE Annual Reports API: {api_url}")
        
        response = session.get(api_url, timeout=NSE_API_TIMEOUT)
        
        # If empty response, try alternative parameter formats
        if response.status_code == 200:
            try:
                test_data = response.json()
                if isinstance(test_data, list) and len(test_data) == 0:
                    # Try Format 2: symbol only (without index parameter)
                    logger.info(f"Empty response, trying alternative API format...")
                    api_url_alt1 = f"{NSE_API_ENDPOINT}?symbol={symbol}"
                    response_alt1 = session.get(api_url_alt1, timeout=NSE_API_TIMEOUT)
                    logger.debug(f"Alternative format 1: {api_url_alt1}, status: {response_alt1.status_code}")
                    
                    if response_alt1.status_code == 200:
                        test_data_alt = response_alt1.json()
                        if isinstance(test_data_alt, list) and len(test_data_alt) > 0:
                            logger.info(f"Alternative format worked! Using Format 2")
                            response = response_alt1
                        else:
                            # Try Format 3: different parameter structure
                            logger.debug(f"Trying alternative format 2...")
                            api_url_alt2 = f"{NSE_API_ENDPOINT}?index=equities&symbol={symbol.upper()}"
                            response_alt2 = session.get(api_url_alt2, timeout=NSE_API_TIMEOUT)
                            if response_alt2.status_code == 200:
                                test_data_alt2 = response_alt2.json()
                                if isinstance(test_data_alt2, list) and len(test_data_alt2) > 0:
                                    logger.info(f"Alternative format 2 worked! Using uppercase symbol")
                                    response = response_alt2
            except Exception as e:
                logger.debug(f"Error trying alternative formats: {e}")
                pass  # Continue with original response
        
        # Debug: Log response details before processing
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response text (first 1000 chars): {response.text[:1000]}")
                return False, f"Invalid JSON response: {e}"
            
            # Debug: Log the response structure
            logger.debug(f"API Response type: {type(data)}")
            if isinstance(data, list):
                logger.debug(f"API Response is list with {len(data)} items")
                if len(data) > 0:
                    logger.debug(f"First item type: {type(data[0])}")
                    if isinstance(data[0], dict):
                        logger.debug(f"First item keys: {list(data[0].keys())}")
            elif isinstance(data, dict):
                logger.debug(f"API Response is dict with keys: {list(data.keys())}")
                logger.debug(f"API Response dict content (first 500 chars): {str(data)[:500]}")
                # Check if data is nested in a key
                if 'data' in data:
                    data = data['data']
                    logger.debug(f"Found 'data' key, using nested data: {len(data) if isinstance(data, list) else 'N/A'}")
                elif 'result' in data:
                    data = data['result']
                    logger.debug(f"Found 'result' key, using nested data: {len(data) if isinstance(data, list) else 'N/A'}")
                elif 'records' in data:
                    data = data['records']
                    logger.debug(f"Found 'records' key, using nested data: {len(data) if isinstance(data, list) else 'N/A'}")
            
            if isinstance(data, list) and len(data) > 0:
                # Filter by year if provided, otherwise get latest (first item)
                report_data = None
                
                # Log available reports for debugging
                logger.info(f"Found {len(data)} annual report(s) for {symbol}")
                if len(data) > 0 and isinstance(data[0], dict):
                    # Log all available reports' structure
                    logger.info(f"Report fields available: {list(data[0].keys())}")
                    
                    # Log date/year fields from first few reports
                    for idx, report in enumerate(data[:min(3, len(data))]):
                        date_fields = {k: v for k, v in report.items() 
                                     if any(keyword in k.lower() for keyword in ['date', 'year', 'fy', 'filing', 'period', 'financial'])}
                        if date_fields:
                            logger.info(f"Report #{idx} date/year fields: {date_fields}")
                        else:
                            logger.info(f"Report #{idx} keys (no date fields found): {list(report.keys())}")
                            # Log all fields for first report to see structure
                            if idx == 0:
                                logger.info(f"Report #{idx} full structure: {report}")
                
                if year:
                    # Try to find report matching the requested financial year
                    # Financial year format: '2022-23' means FY 2022-23 (April 2022 - March 2023)
                    # Parse the year to extract start and end years
                    try:
                        year_start = int(year.split('-')[0])
                        year_end_str = year.split('-')[1] if '-' in year and len(year.split('-')) > 1 else str(year_start + 1)
                        # Convert end year string to full year (e.g., '23' -> 2023, '25' -> 2025)
                        if len(year_end_str) == 2:
                            year_end = int(f"20{year_end_str}")
                        else:
                            year_end = int(year_end_str)
                        
                        logger.info(f"Filtering reports for financial year {year} (FY {year_start}-{year_end})")
                        
                        # First, try to match using fromYr and toYr fields directly (most reliable)
                        candidates = []
                        for idx, report in enumerate(data):
                            from_yr = report.get('fromYr')
                            to_yr = report.get('toYr')
                            
                            # Direct match: fromYr and toYr match exactly
                            if from_yr and to_yr:
                                try:
                                    from_yr_int = int(str(from_yr).strip())
                                    to_yr_int = int(str(to_yr).strip())
                                    
                                    # Match if fromYr matches start year and toYr matches end year
                                    if from_yr_int == year_start and to_yr_int == year_end:
                                        candidates.append((report, idx, f"fromYr={from_yr},toYr={to_yr}"))
                                        logger.debug(f"Candidate #{idx}: Exact match - fromYr={from_yr}, toYr={to_yr}")
                                except (ValueError, AttributeError):
                                    pass
                            
                            # Fallback: check fileName for year pattern (e.g., "AR_2022_2023.pdf" or "2022-23")
                            if not candidates or idx == 0:  # Always check first few for debugging
                                file_name_check = report.get('fileName', '')
                                # Check if fileName contains both years
                                if (str(year_start) in file_name_check and 
                                    (str(year_end) in file_name_check or year_end_str in file_name_check or str(year) in file_name_check)):
                                    if not any(c[0] == report for c in candidates):  # Avoid duplicates
                                        candidates.append((report, idx, f"fileName={file_name_check[:80]}"))
                                        logger.debug(f"Candidate #{idx}: matches year in fileName")
                        
                        # Select best candidate (exact fromYr/toYr match preferred)
                        if candidates:
                            # Prefer candidates with exact fromYr/toYr match
                            exact_matches = [c for c in candidates if 'fromYr' in c[2]]
                            if exact_matches:
                                report_data = exact_matches[0][0]
                                logger.info(f"Selected report for {year}: fromYr={report_data.get('fromYr')}, toYr={report_data.get('toYr')}")
                            else:
                                report_data = candidates[0][0]
                                logger.info(f"Selected report for {year}: {candidates[0][2]}")
                        else:
                            logger.warning(f"No report found matching year {year} in {len(data)} available reports")
                            # Log all available years for debugging
                            available_fy = []
                            for idx, report in enumerate(data[:min(10, len(data))]):  # Check first 10 reports
                                from_yr = report.get('fromYr')
                                to_yr = report.get('toYr')
                                if from_yr and to_yr:
                                    try:
                                        available_fy.append(f"{from_yr}-{str(to_yr)[-2:]}")
                                    except:
                                        pass
                            if available_fy:
                                logger.info(f"Available financial years found: {', '.join(available_fy)}")
                            
                    except Exception as e:
                        logger.warning(f"Error filtering by year {year}: {e}, falling back to latest report")
                        import traceback
                        logger.debug(traceback.format_exc())
                
                # If no year filtering or no match found, use the latest (first) report
                if report_data is None:
                    report_data = data[0]
                    if year:
                        logger.warning(f"No report found for year {year}, using latest available report (index 0)")
                    else:
                        logger.debug(f"No year specified, using latest report")
                
                file_name = report_data.get('fileName')
                
                if file_name:
                    # 3. Construct download URL - handle both full URLs and filenames
                    # The fileName field can be inconsistent:
                    # - Sometimes it's a full URL (starting with 'http')
                    # - Sometimes it's just a filename
                    if file_name.startswith('http'):
                        # fileName is already a full URL, use it directly
                        download_url = file_name
                        logger.debug(f"Using full URL from API: {download_url}")
                    else:
                        # fileName is just a filename, append to archive base URL
                        download_url = f"{NSE_ARCHIVES_BASE_URL}/{file_name}"
                        logger.debug(f"Constructed URL from filename: {download_url}")
                    
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
                    elif pdf_response.status_code == 404:
                        # Try fallback URL construction if 404
                        # Only try fallback if we haven't already tried the annual_reports path
                        if not download_url.startswith('http') or 'annual_reports' not in download_url:
                            logger.warning(f"Got 404 for {download_url}, trying fallback URL construction...")
                            # Extract just the filename if it's a full URL
                            fallback_filename = file_name
                            if '/' in file_name and file_name.startswith('http'):
                                fallback_filename = file_name.split('/')[-1]
                            
                            fallback_url = f"https://nsearchives.nseindia.com/annual_reports/{fallback_filename}"
                            logger.debug(f"Trying fallback URL: {fallback_url}")
                            
                            pdf_response = session.get(fallback_url, timeout=NSE_DOWNLOAD_TIMEOUT)
                            if pdf_response.status_code == 200:
                                output_path = Path(output_path)
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(output_path, 'wb') as f:
                                    f.write(pdf_response.content)
                                file_size = len(pdf_response.content) / (1024 * 1024)
                                logger.info(f"Successfully downloaded via fallback URL: {output_path.name} ({file_size:.2f} MB)")
                                return True, None
                        
                        error_msg = f"Failed to download PDF: HTTP 404 Not Found (tried: {download_url})"
                        logger.warning(f"{symbol}: {error_msg}")
                        return False, error_msg
                    else:
                        error_msg = f"Failed to download PDF: HTTP {pdf_response.status_code}"
                        logger.warning(f"{symbol}: {error_msg}")
                        return False, error_msg
                else:
                    error_msg = "No fileName found in API response"
                    logger.warning(f"{symbol}: {error_msg}")
                    return False, error_msg
            else:
                # Log the actual response for debugging - IMPORTANT for troubleshooting
                error_details = []
                if isinstance(data, list):
                    logger.warning(f"API returned empty list for {symbol} - no annual reports found")
                    error_details.append("empty list (0 items)")
                    # Log response content for debugging
                    logger.info(f"API URL tried: {api_url}")
                    logger.info(f"Response text (first 1500 chars): {response.text[:1500]}")
                elif isinstance(data, dict):
                    error_details.append(f"dict with keys: {list(data.keys())}")
                    # Check if there's a message or error in the response
                    if 'message' in data:
                        msg = data.get('message')
                        error_details.append(f"message: {msg}")
                        logger.info(f"API message: {msg}")
                    if 'error' in data:
                        err = data.get('error')
                        error_details.append(f"error: {err}")
                        logger.info(f"API error: {err}")
                    if 'status' in data:
                        status = data.get('status')
                        error_details.append(f"status: {status}")
                        logger.info(f"API status: {status}")
                    # Log full response for debugging (limit to avoid huge logs)
                    logger.info(f"Full response dict (first 1500 chars): {str(data)[:1500]}")
                else:
                    error_details.append(f"unexpected type: {type(data).__name__}")
                    logger.info(f"API returned unexpected type: {type(data)}, value: {str(data)[:500]}")
                
                error_msg = f"No annual reports found for symbol {symbol} (API returned {', '.join(error_details) if error_details else type(data).__name__})"
                logger.info(f"{symbol}: {error_msg}")
                logger.info(f"API endpoint used: {api_url}")
                return False, error_msg
        elif response.status_code == 403:
            error_msg = "403 Forbidden - Session may have expired or rate limited. Try increasing delay."
            logger.warning(f"{symbol}: {error_msg}")
            return False, error_msg
        elif response.status_code == 404:
            error_msg = f"404 Not Found - API endpoint may have changed or symbol {symbol} not found"
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
    
    def __init__(self, rate_limit_delay: Optional[float] = None):
        """
        Initialize NSE Downloader.
        
        Args:
            rate_limit_delay: Delay in seconds between requests to avoid rate limiting
                             (defaults to config value if None)
        """
        self.rate_limit_delay = rate_limit_delay if rate_limit_delay is not None else NSE_RATE_LIMIT_DELAY
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

