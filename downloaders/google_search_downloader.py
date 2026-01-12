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
        # Prioritize standalone BRSR reports - exclude "annual report" in title
        query = f'site:{domain} filetype:pdf "BRSR" "{year}" -"annual report"'
    else:
        # Prioritize standalone BRSR reports - exclude annual reports from results
        # Use quotes around BRSR to require it, and exclude "annual report"
        query = f'"{company_name}" "BRSR" "{year}" filetype:pdf -"annual report"'
    
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


def validate_pdf_is_brsr(pdf_path: Path) -> bool:
    """
    Quick validation to check if a PDF is likely a BRSR report.
    Extracts text from first few pages and checks for BRSR keywords.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if PDF appears to be a BRSR report, False otherwise
    """
    try:
        # Try pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # Check first 3 pages for BRSR keywords
                text_sample = ""
                for page_num in range(min(3, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    text_sample += page.extract_text() or ""
                
                text_lower = text_sample.lower()
                # Check for BRSR indicators
                brsr_keywords = ['brsr', 'business responsibility', 'sustainability report', 
                               'business responsibility and sustainability']
                has_brsr_keywords = any(keyword in text_lower for keyword in brsr_keywords)
                
                if has_brsr_keywords:
                    return True
                return False
        except ImportError:
            # Fallback to PyMuPDF if pdfplumber not available
            try:
                import fitz
                doc = fitz.open(pdf_path)
                text_sample = ""
                for page_num in range(min(3, len(doc))):
                    page = doc[page_num]
                    text_sample += page.get_text().lower()
                doc.close()
                
                brsr_keywords = ['brsr', 'business responsibility', 'sustainability report']
                has_brsr_keywords = any(keyword in text_sample for keyword in brsr_keywords)
                return has_brsr_keywords
            except ImportError:
                # If no PDF library available, assume it's valid (can't validate)
                return True
    except Exception as e:
        logger.debug(f"Error validating PDF content: {e}")
        # If validation fails, assume it might be valid (don't reject on validation error)
        return True


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
    
    # Prioritize results - Google Search query already includes "BRSR" so results should be relevant
    # Use smart prioritization to pick the best result, rely on PDF validation for final check
    # Be less strict with filtering since PDF validation will verify content
    brsr_results = []
    annual_results = []
    
    # Extract year components for matching
    year_parts = year.split('-') if '-' in year else [year]
    year_start = year_parts[0] if year_parts else ""
    year_end_short = year_parts[1] if len(year_parts) > 1 else ""
    
    for idx, result in enumerate(pdf_results):
        title = result.get('title', '').lower()
        link = result.get('link', '').lower()
        combined_text = f"{title} {link}"
        
        # Check if it's clearly an annual report (has "annual report" but no BRSR indicators)
        has_annual = 'annual report' in combined_text or 'annual-report' in combined_text or 'annualreport' in combined_text
        has_brsr_indicators = 'brsr' in combined_text or 'business responsibility' in combined_text or 'sustainability report' in combined_text
        
        # Only skip if it's clearly an annual report WITHOUT any BRSR indicators
        if has_annual and not has_brsr_indicators:
            annual_results.append(result)
            continue
        
        # Accept all other results (query includes "BRSR" so they should be relevant)
        # Calculate comprehensive priority score - higher is better
        priority = 0
        
        # Base priority: Google API returns results in relevance order, so earlier results are better
        # Give higher priority to results that appear earlier in Google's ranking
        priority += (len(pdf_results) - idx) * 2  # Earlier results get more points
        
        # Strong indicators of BRSR report
        if 'brsr' in title:
            priority += 15  # Very strong signal
        if 'brsr' in link:
            priority += 10  # Strong signal in URL
        
        # Standalone BRSR indicators
        if 'standalone' in combined_text:
            priority += 8
        if 'standalone brsr' in combined_text:
            priority += 12  # Even stronger
        
        # Business Responsibility / Sustainability indicators
        if 'business responsibility' in title:
            priority += 8
        if 'sustainability report' in title or 'sustainability-report' in combined_text:
            priority += 6
        if 'sustainability' in title:
            priority += 3
        
        # Year matching - higher priority if year appears in title/URL
        if year_start in combined_text:
            priority += 5
        if year_end_short in combined_text:
            priority += 3
        if year.replace('-', '_') in combined_text or year.replace('-', '/') in combined_text:
            priority += 6  # Year in various formats
        
        # Company name matching (partial match)
        company_words = company_name.lower().split()
        for word in company_words:
            if len(word) > 3 and word in combined_text:  # Only meaningful words
                priority += 2
        
        # URL patterns that suggest BRSR reports
        if '/brsr/' in link or '/sustainability/' in link or '/csr/' in link:
            priority += 7
        if 'brsr' in link.split('/')[-1]:  # BRSR in filename
            priority += 10
        
        # Negative signals (lower priority but don't exclude)
        if has_annual:
            priority -= 8  # Annual reports are lower priority but still considered
        if 'annual' in title and 'brsr' not in combined_text:
            priority -= 5
        
        # Store priority with result and log for debugging
        result['_priority'] = priority
        result['_index'] = idx  # Store original position
        brsr_results.append(result)
    
    # Sort by priority (highest first)
    brsr_results.sort(key=lambda x: x.get('_priority', 0), reverse=True)
    
    # Log top results for debugging
    if brsr_results:
        logger.info(f"Found {len(brsr_results)} result(s) (filtered {len(annual_results)} annual reports)")
        logger.debug(f"Top 3 results by priority:")
        for i, result in enumerate(brsr_results[:3], 1):
            logger.debug(f"  {i}. Priority: {result.get('_priority', 0)}, Title: {result.get('title', 'N/A')[:60]}")
    
    # Use prioritized results (PDF validation will verify they're BRSR)
    if brsr_results:
        pdf_results = brsr_results
        logger.info(f"Sorted {len(brsr_results)} result(s) by relevance, will try top results")
    else:
        error_msg = "No search results found after filtering"
        logger.warning(f"{company_name} ({year}): {error_msg}")
        return False, error_msg
    
    # Try to download and validate PDFs, starting with highest priority results
    for i, result in enumerate(pdf_results[:5], 1):  # Try up to 5 results
        pdf_url = result['link']
        priority = result.get('_priority', 0)
        logger.info(f"Attempting to download PDF {i}/{len(pdf_results)} (priority: {priority}): {pdf_url}")
        
        success, error = download_pdf(pdf_url, output_path)
        
        if success:
            # Verify it's actually a valid PDF by checking file size and content
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and output_path_obj.stat().st_size > 1000:  # At least 1KB
                # Check first bytes are PDF
                with open(output_path_obj, 'rb') as f:
                    first_bytes = f.read(4)
                    if first_bytes == b'%PDF':
                        # Quick validation: Try to extract some text and check for BRSR keywords
                        is_likely_brsr = validate_pdf_is_brsr(output_path_obj)
                        if is_likely_brsr:
                            logger.info(f"✓ Valid BRSR PDF downloaded from Google Search (result {i})")
                            return True, None
                        else:
                            logger.warning(f"Downloaded PDF doesn't appear to be a BRSR report (result {i}), trying next result...")
                            output_path_obj.unlink()  # Delete incorrect PDF
                    else:
                        logger.warning(f"Downloaded file is not a valid PDF (result {i}), trying next result...")
                        output_path_obj.unlink()  # Delete invalid file
            else:
                logger.warning(f"Downloaded file is too small (result {i}), trying next result...")
                if output_path_obj.exists():
                    output_path_obj.unlink()
        
        logger.debug(f"Download failed (result {i}): {error}, trying next result...")
    
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

