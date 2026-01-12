"""
NSE API Downloader - Tier 1 (Primary) download source using NSE corporate filings API.
"""
import logging
import requests
import time
import zipfile
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


def extract_pdf_from_zip(zip_path: Path, output_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Extract PDF file from ZIP archive.
    
    Args:
        zip_path: Path to ZIP file
        output_path: Path where extracted PDF should be saved
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find PDF files in the ZIP
            pdf_files = [f for f in zip_ref.namelist() if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                return False, "No PDF files found in ZIP archive"
            
            # Use the first PDF file found (usually there's only one)
            pdf_file = pdf_files[0]
            if len(pdf_files) > 1:
                logger.info(f"Multiple PDFs in ZIP, extracting first one: {pdf_file}")
            
            # Extract the PDF
            with zip_ref.open(pdf_file) as pdf_content:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as out_file:
                    out_file.write(pdf_content.read())
            
            logger.info(f"Extracted PDF from ZIP: {pdf_file}")
            return True, None
            
    except zipfile.BadZipFile:
        return False, "Invalid ZIP file"
    except Exception as e:
        return False, f"Error extracting ZIP: {e}"


def get_nse_report(symbol: str, output_path: Path, year: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
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
                return False, f"Invalid JSON response: {e}", None
            
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
            
            # The Annual Reports API returns a list of reports
            # Use data[0] as the latest report (or filter by year if provided)
            if isinstance(data, list) and len(data) > 0:
                # Filter by year if provided, otherwise get latest (first item - data[0])
                report_data = None
                report_type = None  # Will be set to "BRSR" or "Annual Report"
                
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
                        
                        # Helper function to check if a report is a BRSR report
                        def is_brsr_report(report: dict) -> bool:
                            """Check if report is a standalone BRSR report."""
                            submission_type = str(report.get('submission_type', '')).upper()
                            file_name = str(report.get('fileName', '')).upper()
                            
                            # Check submission_type for BRSR indicators
                            brsr_keywords = ['BRSR', 'BUSINESS RESPONSIBILITY', 'SUSTAINABILITY']
                            if any(keyword in submission_type for keyword in brsr_keywords):
                                return True
                            
                            # Check fileName for BRSR indicators (but exclude "AR_" prefix which indicates Annual Report)
                            if not file_name.startswith('AR_') and any(keyword in file_name for keyword in brsr_keywords):
                                return True
                            
                            return False
                        
                        # First, try to match using fromYr and toYr fields directly (most reliable)
                        # Separate BRSR and annual report candidates
                        brsr_candidates = []
                        annual_candidates = []
                        
                        for idx, report in enumerate(data):
                            from_yr = report.get('fromYr')
                            to_yr = report.get('toYr')
                            
                            # Check if year matches
                            year_matches = False
                            match_reason = ""
                            
                            # Direct match: fromYr and toYr match exactly
                            if from_yr and to_yr:
                                try:
                                    from_yr_int = int(str(from_yr).strip())
                                    to_yr_int = int(str(to_yr).strip())
                                    
                                    # Match if fromYr matches start year and toYr matches end year
                                    if from_yr_int == year_start and to_yr_int == year_end:
                                        year_matches = True
                                        match_reason = f"fromYr={from_yr},toYr={to_yr}"
                                        logger.debug(f"Report #{idx}: Year match - {match_reason}")
                                    # NSE API quirk: sometimes toYr equals fromYr (e.g., 2024-2024 for FY 2024-25)
                                    # In this case, match if fromYr matches start year and toYr equals fromYr
                                    elif from_yr_int == year_start and to_yr_int == from_yr_int:
                                        # This is likely the correct FY (NSE API quirk)
                                        year_matches = True
                                        match_reason = f"fromYr={from_yr},toYr={to_yr} (NSE API quirk: toYr=fromYr)"
                                        logger.debug(f"Report #{idx}: Year match (NSE API quirk) - {match_reason}")
                                except (ValueError, AttributeError):
                                    pass
                            
                            # Fallback: check fileName for year pattern (e.g., "AR_2022_2023.pdf" or "2022-23")
                            if not year_matches:
                                file_name_check = report.get('fileName', '')
                                # Check if fileName contains both years
                                if (str(year_start) in file_name_check and 
                                    (str(year_end) in file_name_check or year_end_str in file_name_check or str(year) in file_name_check)):
                                    year_matches = True
                                    match_reason = f"fileName={file_name_check[:80]}"
                                    logger.debug(f"Report #{idx}: Year match in fileName")
                            
                            # If year matches, categorize as BRSR or Annual Report
                            if year_matches:
                                if is_brsr_report(report):
                                    brsr_candidates.append((report, idx, match_reason))
                                    logger.debug(f"BRSR candidate #{idx}: {match_reason}")
                                else:
                                    annual_candidates.append((report, idx, match_reason))
                                    logger.debug(f"Annual Report candidate #{idx}: {match_reason}")
                        
                        # ONLY select BRSR reports - reject annual reports
                        candidates = brsr_candidates  # Only use BRSR candidates, no annual report fallback
                        report_type = "BRSR"
                        
                        if candidates:
                            # Prefer candidates with exact fromYr/toYr match
                            exact_matches = [c for c in candidates if 'fromYr' in c[2]]
                            if exact_matches:
                                report_data = exact_matches[0][0]
                                logger.info(f"Selected BRSR report for {year}: {exact_matches[0][2]}")
                            else:
                                report_data = candidates[0][0]
                                logger.info(f"Selected BRSR report for {year}: {candidates[0][2]}")
                        else:
                            # No BRSR report found - mark as failed (don't use annual report)
                            logger.warning(f"No BRSR report found matching year {year} in {len(data)} available reports")
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
                            # Don't set report_data - will return failure below
                            
                    except Exception as e:
                        logger.warning(f"Error filtering by year {year}: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        # Don't fall back - will return failure below if report_data is None
                
                # If no BRSR report found, return failure (don't use annual report fallback)
                if report_data is None:
                    if year:
                        error_msg = f"No BRSR report found for year {year} (only BRSR reports are accepted, annual reports are rejected)"
                        logger.warning(f"{symbol}: {error_msg}")
                    else:
                        error_msg = "No BRSR report found (only BRSR reports are accepted, annual reports are rejected)"
                        logger.warning(f"{symbol}: {error_msg}")
                    return False, error_msg, None
                
                file_name = report_data.get('fileName')
                
                if file_name:
                    # 3. Construct download URL - handle both full URLs and filenames
                    # URL Construction Logic as specified:
                    # - Check if fileName starts with 'http'. If yes, use it as download_url directly.
                    # - If no, append it to NSE_ARCHIVES_BASE_URL.
                    # - Edge Case: If the constructed URL doesn't look like a valid NSE archive link,
                    #   allow a fallback to https://nsearchives.nseindia.com/annual_reports/{fileName}
                    
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
                        
                        # Validate downloaded PDF
                        file_size_bytes = output_path.stat().st_size
                        file_size_mb = file_size_bytes / (1024 * 1024)  # MB
                        
                        # Check file size (must be at least 1KB)
                        if file_size_bytes < 1000:
                            error_msg = f"Downloaded file is too small ({file_size_bytes} bytes) - likely corrupted or incomplete"
                            logger.warning(f"{symbol}: {error_msg}")
                            output_path.unlink()  # Delete invalid file
                            return False, error_msg, None
                        
                        # Check if file is actually a PDF or ZIP (check first bytes)
                        try:
                            with open(output_path, 'rb') as f:
                                first_bytes = f.read(4)
                                
                                if first_bytes == b'%PDF':
                                    # Valid PDF - proceed
                                    pass
                                elif first_bytes == b'PK\x03\x04' or first_bytes == b'PK\x05\x06':
                                    # ZIP file - extract PDF from it
                                    logger.info(f"{symbol}: Downloaded file is a ZIP archive, extracting PDF...")
                                    zip_path = output_path
                                    # Create temporary path for extracted PDF
                                    extracted_pdf_path = output_path.with_suffix('.pdf.tmp')
                                    
                                    success_extract, extract_error = extract_pdf_from_zip(zip_path, extracted_pdf_path)
                                    
                                    if not success_extract:
                                        error_msg = f"Failed to extract PDF from ZIP: {extract_error}"
                                        logger.warning(f"{symbol}: {error_msg}")
                                        output_path.unlink()  # Delete ZIP file
                                        return False, error_msg, None
                                    
                                    # Replace ZIP with extracted PDF
                                    zip_path.unlink()  # Delete ZIP file
                                    extracted_pdf_path.replace(output_path)  # Rename extracted PDF to final path
                                    logger.info(f"{symbol}: Successfully extracted PDF from ZIP")
                                else:
                                    error_msg = f"Downloaded file is not a valid PDF or ZIP (first bytes: {first_bytes})"
                                    logger.warning(f"{symbol}: {error_msg}")
                                    output_path.unlink()  # Delete invalid file
                                    return False, error_msg, None
                        except Exception as e:
                            error_msg = f"Error validating PDF/ZIP file: {e}"
                            logger.warning(f"{symbol}: {error_msg}")
                            if output_path.exists():
                                output_path.unlink()  # Delete invalid file
                            return False, error_msg, None
                        
                        logger.info(f"Successfully downloaded and validated {output_path.name} ({file_size_mb:.2f} MB)")
                        return True, None, report_type
                    elif pdf_response.status_code == 404:
                        # Edge Case: If we got a 404, try the fallback to https://nsearchives.nseindia.com/annual_reports/{fileName}
                        # Check if the URL looks like a valid NSE archive link
                        is_valid_nse_url = 'nsearchives.nseindia.com/annual_reports' in download_url
                        
                        # Always try fallback on 404 (Edge Case: If the constructed URL doesn't look like a standard NSE archive link)
                        logger.warning(f"Got 404 for {download_url}, trying fallback URL construction...")
                        # Extract just the filename for fallback
                        fallback_filename = file_name
                        if '/' in file_name:
                            # Extract filename from full URL or path
                            fallback_filename = file_name.split('/')[-1]
                        
                        # Edge Case: Fallback to standard NSE archive URL format
                        fallback_url = f"https://nsearchives.nseindia.com/annual_reports/{fallback_filename}"
                        logger.debug(f"Trying fallback URL: {fallback_url}")
                        
                        pdf_response_fallback = session.get(fallback_url, timeout=NSE_DOWNLOAD_TIMEOUT)
                        if pdf_response_fallback.status_code == 200:
                            output_path = Path(output_path)
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, 'wb') as f:
                                f.write(pdf_response_fallback.content)
                            
                            # Validate downloaded PDF
                            file_size_bytes = output_path.stat().st_size
                            file_size_mb = file_size_bytes / (1024 * 1024)  # MB
                            
                            # Check file size (must be at least 1KB)
                            if file_size_bytes < 1000:
                                error_msg = f"Downloaded file is too small ({file_size_bytes} bytes) - likely corrupted or incomplete"
                                logger.warning(f"{symbol}: {error_msg}")
                                output_path.unlink()  # Delete invalid file
                                return False, error_msg, None
                            
                            # Check if file is actually a PDF or ZIP (check first bytes)
                            try:
                                with open(output_path, 'rb') as f:
                                    first_bytes = f.read(4)
                                    
                                    if first_bytes == b'%PDF':
                                        # Valid PDF - proceed
                                        pass
                                    elif first_bytes == b'PK\x03\x04' or first_bytes == b'PK\x05\x06':
                                        # ZIP file - extract PDF from it
                                        logger.info(f"{symbol}: Downloaded file is a ZIP archive, extracting PDF...")
                                        zip_path = output_path
                                        # Create temporary path for extracted PDF
                                        extracted_pdf_path = output_path.with_suffix('.pdf.tmp')
                                        
                                        success_extract, extract_error = extract_pdf_from_zip(zip_path, extracted_pdf_path)
                                        
                                        if not success_extract:
                                            error_msg = f"Failed to extract PDF from ZIP: {extract_error}"
                                            logger.warning(f"{symbol}: {error_msg}")
                                            output_path.unlink()  # Delete ZIP file
                                            return False, error_msg, None
                                        
                                        # Replace ZIP with extracted PDF
                                        zip_path.unlink()  # Delete ZIP file
                                        extracted_pdf_path.replace(output_path)  # Rename extracted PDF to final path
                                        logger.info(f"{symbol}: Successfully extracted PDF from ZIP")
                                    else:
                                        error_msg = f"Downloaded file is not a valid PDF or ZIP (first bytes: {first_bytes})"
                                        logger.warning(f"{symbol}: {error_msg}")
                                        output_path.unlink()  # Delete invalid file
                                        return False, error_msg, None
                            except Exception as e:
                                error_msg = f"Error validating PDF/ZIP file: {e}"
                                logger.warning(f"{symbol}: {error_msg}")
                                if output_path.exists():
                                    output_path.unlink()  # Delete invalid file
                                return False, error_msg, None
                            
                            logger.info(f"Successfully downloaded and validated via fallback URL: {output_path.name} ({file_size_mb:.2f} MB)")
                            return True, None, report_type
                        
                        # If still 404 after fallback attempt, return error
                        error_msg = f"Failed to download PDF: HTTP 404 Not Found (tried: {download_url} and fallback: {fallback_url})"
                        logger.warning(f"{symbol}: {error_msg}")
                        return False, error_msg, None
                    else:
                        error_msg = f"Failed to download PDF: HTTP {pdf_response.status_code}"
                        logger.warning(f"{symbol}: {error_msg}")
                        return False, error_msg, None
                else:
                    error_msg = "No fileName found in API response"
                    logger.warning(f"{symbol}: {error_msg}")
                    return False, error_msg, None
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
                return False, error_msg, None
        elif response.status_code == 403:
            error_msg = "403 Forbidden - Session may have expired or rate limited. Try increasing delay."
            logger.warning(f"{symbol}: {error_msg}")
            return False, error_msg, None
        elif response.status_code == 404:
            error_msg = f"404 Not Found - API endpoint may have changed or symbol {symbol} not found"
            logger.warning(f"{symbol}: {error_msg}")
            return False, error_msg, None
        else:
            error_msg = f"API returned status code {response.status_code}"
            logger.warning(f"{symbol}: {error_msg}")
            return False, error_msg, None
            
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout while accessing NSE API: {e}"
        logger.error(f"{symbol}: {error_msg}")
        return False, error_msg, None
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error: {e}"
        logger.error(f"{symbol}: {error_msg}")
        return False, error_msg, None
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(f"{symbol}: {error_msg}", exc_info=True)
        return False, error_msg, None


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
        
    def download(self, symbol: str, output_path: Path, year: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Download BRSR report with rate limiting.
        
        Args:
            symbol: NSE symbol
            output_path: Path where PDF should be saved
            year: Optional financial year filter
        
        Returns:
            Tuple of (success: bool, error_message: Optional[str], report_type: Optional[str])
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

