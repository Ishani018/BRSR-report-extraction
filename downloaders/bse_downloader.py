"""
BSE Downloader - Downloads BRSR reports from Bombay Stock Exchange.
Since BSE doesn't have a public API like NSE, this uses Google Search with site restriction.
"""
import logging
import requests
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_bse_report_via_google(
    company_name: str,
    symbol: str,
    year: str,
    output_path: Path
) -> Tuple[bool, Optional[str]]:
    """
    Download BRSR report from BSE using Google Search with site restriction.
    
    Since BSE doesn't have a public API, we use Google Search restricted to bseindia.com.
    
    Args:
        company_name: Company name
        symbol: BSE symbol (optional, for search refinement)
        year: Financial year (e.g., '2022-23')
        output_path: Path where PDF should be saved
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        from .google_search_downloader import build_search_query, search_with_web_scraping, is_pdf_url, download_pdf
        
        # Build search query restricted to BSE website
        # Prioritize standalone BRSR reports - exclude annual reports
        search_terms = f'"{company_name}"'
        if symbol and symbol.strip():
            search_terms += f' "{symbol}"'
        search_terms += f' "BRSR" {year} -"annual report"'
        
        query = f'site:bseindia.com {search_terms} filetype:pdf'
        
        logger.info(f"Searching BSE via Google: {query}")
        
        # Use web scraping (free, no API key needed)
        results = search_with_web_scraping(query, num_results=10)
        
        if not results:
            error_msg = "No BSE search results found"
            logger.warning(f"{company_name} ({year}): {error_msg}")
            return False, error_msg
        
        # Filter for PDF links
        pdf_results = [r for r in results if is_pdf_url(r.get('link', ''))]
        
        if not pdf_results:
            error_msg = "No PDF links found in BSE search results"
            logger.warning(f"{company_name} ({year}): {error_msg}")
            return False, error_msg
        
        # Try to download first valid PDF
        for i, result in enumerate(pdf_results[:5], 1):  # Try up to 5 results
            pdf_url = result['link']
            logger.info(f"Attempting to download BSE PDF {i}/{len(pdf_results)}: {pdf_url}")
            
            success, error = download_pdf(pdf_url, output_path)
            
            if success:
                # Verify it's actually a valid PDF
                output_path_obj = Path(output_path)
                if output_path_obj.exists() and output_path_obj.stat().st_size > 1000:  # At least 1KB
                    # Check first bytes are PDF
                    with open(output_path_obj, 'rb') as f:
                        first_bytes = f.read(4)
                        if first_bytes == b'%PDF':
                            logger.info(f"✓ Valid PDF downloaded from BSE")
                            return True, None
                        else:
                            logger.warning(f"Downloaded file is not a valid PDF, trying next result...")
                            output_path_obj.unlink()  # Delete invalid file
                else:
                    logger.warning(f"Downloaded file is too small, trying next result...")
            
            logger.debug(f"Download failed: {error}, trying next result...")
        
        error_msg = "All BSE PDF download attempts failed"
        logger.warning(f"{company_name} ({year}): {error_msg}")
        return False, error_msg
        
    except ImportError as e:
        error_msg = f"Required module not available: {e}"
        logger.error(f"{company_name} ({year}): {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Error downloading from BSE: {e}"
        logger.error(f"{company_name} ({year}): {error_msg}", exc_info=True)
        return False, error_msg


class BSEDownloader:
    """
    BSE Downloader class.
    """
    
    def __init__(self, rate_limit_delay: Optional[float] = None):
        """
        Initialize BSE Downloader.
        
        Args:
            rate_limit_delay: Delay in seconds between requests (defaults to 2.0 if None)
        """
        self.rate_limit_delay = rate_limit_delay if rate_limit_delay is not None else 2.0
        self.last_request_time = 0
    
    def download(
        self,
        company_name: str,
        symbol: str,
        year: str,
        output_path: Path
    ) -> Tuple[bool, Optional[str]]:
        """
        Download BRSR report from BSE with rate limiting.
        
        Args:
            company_name: Company name
            symbol: BSE symbol
            year: Financial year
            output_path: Path where PDF should be saved
        
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        
        return get_bse_report_via_google(company_name, symbol, year, output_path)


def get_bse_report(company_name: str, symbol: str, year: str, output_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to download BRSR report from BSE.
    
    Args:
        company_name: Company name
        symbol: BSE symbol
        year: Financial year
        output_path: Path where PDF should be saved
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    return get_bse_report_via_google(company_name, symbol, year, output_path)


if __name__ == "__main__":
    # Test the downloader
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_company = "Reliance Industries"
    test_symbol = "RELIANCE"
    test_year = "2023-24"
    test_output = Path(__file__).parent.parent / "downloads" / "test" / f"{test_company}_BRSR_bse_test.pdf"
    
    logger.info(f"Testing BSE downloader for: {test_company}")
    success, error = get_bse_report(test_company, test_symbol, test_year, test_output)
    
    if success:
        logger.info(f"✓ Successfully downloaded to: {test_output}")
        sys.exit(0)
    else:
        logger.error(f"✗ Download failed: {error}")
        sys.exit(1)
