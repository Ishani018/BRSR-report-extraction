"""
Google Search Downloader - Tier 2 (Fallback) download source using Google Search.
"""
import logging
import requests
import time
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.parse import quote_plus, urlparse
import re

from config.config import (
    GOOGLE_SEARCH_API_KEY,
    GOOGLE_SEARCH_ENGINE_ID,
    GOOGLE_SEARCH_TIMEOUT,
    GOOGLE_SEARCH_MAX_RESULTS
)

logger = logging.getLogger(__name__)


def extract_domain(website: str) -> str:
    """
    Extract domain from website URL.
    
    Args:
        website: Website URL (e.g., 'https://www.company.com' or 'company.com')
        
    Returns:
        Domain name (e.g., 'company.com')
    """
    if not website:
        return ""
    
    # Remove protocol if present
    website = website.replace('https://', '').replace('http://', '').strip()
    
    # Remove www. if present
    if website.startswith('www.'):
        website = website[4:]
    
    # Remove trailing slash
    website = website.rstrip('/')
    
    # Get domain only (remove path)
    parts = website.split('/')
    domain = parts[0]
    
    return domain


def build_search_query(company_name: str, website: str, year: str) -> str:
    """
    Build Google search query for BRSR report.
    
    Args:
        company_name: Company name
        website: Company website (optional)
        year: Financial year (e.g., '2022-23')
        
    Returns:
        Google search query string
    """
    domain = extract_domain(website) if website else ""
    
    if domain:
        # Use site: search to limit to company website
        query = f'site:{domain} filetype:pdf "BRSR" {year}'
    else:
        # Fallback to company name search
        query = f'"{company_name}" "Business Responsibility and Sustainability Report" "BRSR" {year} filetype:pdf'
    
    return query


def search_with_google_api(query: str, num_results: int = 10) -> List[dict]:
    """
    Search using Google Custom Search API.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of search result dictionaries with 'link' and 'title' keys
    """
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        logger.warning("Google Search API key or Engine ID not configured. Skipping API search.")
        return []
    
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_SEARCH_API_KEY,
            'cx': GOOGLE_SEARCH_ENGINE_ID,
            'q': query,
            'num': min(num_results, 10)  # Google API max is 10 per request
        }
        
        logger.debug(f"Searching Google API with query: {query}")
        response = requests.get(url, params=params, timeout=GOOGLE_SEARCH_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            results = [{'link': item.get('link'), 'title': item.get('title')} for item in items]
            logger.info(f"Google API returned {len(results)} results")
            return results
        else:
            logger.warning(f"Google API returned status {response.status_code}: {response.text[:200]}")
            return []
            
    except Exception as e:
        logger.error(f"Error searching with Google API: {e}")
        return []


def search_with_web_scraping(query: str, num_results: int = 10) -> List[dict]:
    """
    Search using web scraping (BeautifulSoup) - fallback if API not available.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of search result dictionaries with 'link' and 'title' keys
    """
    try:
        from bs4 import BeautifulSoup
        
        # Build Google search URL
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
        
        # Headers to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
        }
        
        logger.debug(f"Searching Google with web scraping: {query}")
        response = requests.get(search_url, headers=headers, timeout=GOOGLE_SEARCH_TIMEOUT)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse Google search results (structure may change)
            for result in soup.select('div.g')[:num_results]:
                link_elem = result.select_one('a[href^="http"]')
                title_elem = result.select_one('h3')
                
                if link_elem and title_elem:
                    # Extract actual URL (Google wraps links)
                    href = link_elem.get('href', '')
                    # Remove Google redirect wrapper if present
                    if href.startswith('/url?q='):
                        href = href.split('&')[0][7:]  # Extract actual URL
                    
                    results.append({
                        'link': href,
                        'title': title_elem.get_text(strip=True)
                    })
            
            logger.info(f"Web scraping returned {len(results)} results")
            return results
        else:
            logger.warning(f"Google search returned status {response.status_code}")
            return []
            
    except ImportError:
        logger.warning("BeautifulSoup not available. Install with: pip install beautifulsoup4")
        return []
    except Exception as e:
        logger.error(f"Error scraping Google search: {e}")
        return []


def is_pdf_url(url: str) -> bool:
    """
    Check if URL points to a PDF file.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL appears to be a PDF
    """
    if not url:
        return False
    
    # Check extension
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    if path.endswith('.pdf'):
        return True
    
    # Check if URL contains pdf keyword
    if 'pdf' in path or 'pdf' in parsed.query.lower():
        return True
    
    return False


def download_pdf(url: str, output_path: Path, timeout: int = 30) -> Tuple[bool, Optional[str]]:
    """
    Download PDF from URL.
    
    Args:
        url: PDF URL
        output_path: Path where PDF should be saved
        timeout: Download timeout in seconds
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
        }
        
        logger.debug(f"Downloading PDF from: {url}")
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        
        if response.status_code == 200:
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                # Try to verify it's actually a PDF by checking first bytes
                first_bytes = response.content[:4]
                if first_bytes != b'%PDF':
                    error_msg = f"URL does not appear to be a PDF (Content-Type: {content_type})"
                    logger.warning(error_msg)
                    return False, error_msg
            
            # Save PDF
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Successfully downloaded {output_path.name} ({file_size:.2f} MB)")
            return True, None
        else:
            error_msg = f"Failed to download PDF: HTTP {response.status_code}"
            logger.warning(error_msg)
            return False, error_msg
            
    except requests.exceptions.Timeout:
        error_msg = "Download timeout"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error downloading PDF: {e}"
        logger.error(error_msg)
        return False, error_msg


def get_google_search_report(
    company_name: str,
    website: str,
    year: str,
    output_path: Path
) -> Tuple[bool, Optional[str]]:
    """
    Download BRSR report using Google Search (Tier 2 fallback).
    
    Args:
        company_name: Company name
        website: Company website URL
        year: Financial year (e.g., '2022-23')
        output_path: Path where PDF should be saved
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    # Build search query
    query = build_search_query(company_name, website, year)
    logger.info(f"Searching Google for: {query}")
    
    # Try Google Custom Search API first
    results = search_with_google_api(query, num_results=GOOGLE_SEARCH_MAX_RESULTS)
    
    # Fall back to web scraping if API not available or no results
    if not results:
        logger.info("Google API not available or no results, trying web scraping...")
        results = search_with_web_scraping(query, num_results=GOOGLE_SEARCH_MAX_RESULTS)
    
    if not results:
        error_msg = "No search results found"
        logger.warning(f"{company_name} ({year}): {error_msg}")
        return False, error_msg
    
    # Filter for PDF links
    pdf_results = [r for r in results if is_pdf_url(r.get('link', ''))]
    
    if not pdf_results:
        error_msg = "No PDF links found in search results"
        logger.warning(f"{company_name} ({year}): {error_msg}")
        return False, error_msg
    
    # Try to download first valid PDF
    for i, result in enumerate(pdf_results[:5], 1):  # Try up to 5 results
        pdf_url = result['link']
        logger.info(f"Attempting to download PDF {i}/{len(pdf_results)}: {pdf_url}")
        
        success, error = download_pdf(pdf_url, output_path)
        
        if success:
            # Verify it's actually a valid PDF by checking file size and content
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and output_path_obj.stat().st_size > 1000:  # At least 1KB
                # Check first bytes are PDF
                with open(output_path_obj, 'rb') as f:
                    first_bytes = f.read(4)
                    if first_bytes == b'%PDF':
                        logger.info(f"✓ Valid PDF downloaded from Google Search")
                        return True, None
                    else:
                        logger.warning(f"Downloaded file is not a valid PDF, trying next result...")
                        output_path_obj.unlink()  # Delete invalid file
            else:
                logger.warning(f"Downloaded file is too small, trying next result...")
        
        logger.debug(f"Download failed: {error}, trying next result...")
    
    error_msg = "All PDF download attempts failed"
    logger.warning(f"{company_name} ({year}): {error_msg}")
    return False, error_msg


class GoogleSearchDownloader:
    """
    Google Search Downloader class.
    """
    
    def __init__(self):
        """Initialize Google Search Downloader."""
        pass
    
    def download(
        self,
        company_name: str,
        website: str,
        year: str,
        output_path: Path
    ) -> Tuple[bool, Optional[str]]:
        """
        Download BRSR report using Google Search.
        
        Args:
            company_name: Company name
            website: Company website URL
            year: Financial year
            output_path: Path where PDF should be saved
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        return get_google_search_report(company_name, website, year, output_path)


if __name__ == "__main__":
    # Test the downloader
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_company = "Reliance Industries"
    test_website = "https://www.ril.com"
    test_year = "2023-24"
    test_output = Path(__file__).parent.parent / "downloads" / "test" / f"{test_company}_BRSR_google_test.pdf"
    
    logger.info(f"Testing Google Search downloader for: {test_company}")
    success, error = get_google_search_report(test_company, test_website, test_year, test_output)
    
    if success:
        logger.info(f"✓ Successfully downloaded to: {test_output}")
        sys.exit(0)
    else:
        logger.error(f"✗ Download failed: {error}")
        sys.exit(1)

