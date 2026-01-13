"""
Google Search Downloader - Tier 2 (Fallback) download source using Google Search.
"""
import logging
import requests
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from urllib.parse import quote_plus, urlparse
import re

from config.config import (
    GOOGLE_SEARCH_API_KEY,
    GOOGLE_SEARCH_ENGINE_ID,
    GOOGLE_SEARCH_TIMEOUT,
    GOOGLE_SEARCH_MAX_RESULTS,
    NEGATIVE_KEYWORDS
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
        # Include both "BRSR" and "Business Responsibility and Sustainability Report" as search terms
        query = f'site:{domain} filetype:pdf ("BRSR" OR "Business Responsibility and Sustainability Report") {year}'
    else:
        # Search with company name, include both BRSR variants
        query = f'"{company_name}" filetype:pdf ("BRSR" OR "Business Responsibility and Sustainability Report") {year}'
    
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


def score_search_result(result: Dict) -> int:
    """
    Score a search result based on title and link.
    Prioritizes title content over filename/URL (which may be gibberish).
    
    Args:
        result: Dictionary with 'title' and 'link' keys
        
    Returns:
        Score (higher is better). Negative scores indicate junk documents (-1000 = kill immediately).
    """
    title = result.get('title', '').lower()
    link = result.get('link', '').lower()
    
    score = 0
    
    # CRITICAL: Junk filter - kill immediately if found (-1000 points)
    # Check for negative keywords from config (multi-word phrases)
    for keyword in NEGATIVE_KEYWORDS:
        keyword_lower = keyword.lower()
        if keyword_lower in title or keyword_lower in link:
            logger.debug(f"Junk detected (config keyword) in '{result.get('title', 'N/A')[:50]}': {keyword}")
            return -1000  # Kill immediately
    
    # Also check for individual junk words (single-word patterns)
    # These catch cases like "Presentation", "Investor", "Earnings", etc. even if not in config
    junk_words = ['presentation', 'investor', 'earnings', 'call', 'transcript', 'release', 'brief']
    for word in junk_words:
        if word in title or word in link:
            logger.debug(f"Junk detected (word pattern) in '{result.get('title', 'N/A')[:50]}': {word}")
            return -1000  # Kill immediately
    
    # Trust the Title: Strong positive signals (title is more reliable than URL)
    if 'business responsibility' in title or 'brsr' in title or 'sustainability report' in title:
        score += 100
    
    if 'integrated report' in title:
        score += 50
    
    if 'annual report' in title:
        score += 30
    
    # Link scoring: Only trust specific keywords, ignore gibberish filenames
    # Do NOT give points for keywords in link unless it's strictly "brsr" or "sustainability"
    if 'brsr' in link:
        score += 20  # BRSR in URL is a good signal
    
    if 'sustainability' in link:
        score += 10  # Sustainability in URL is a moderate signal
    
    # Don't penalize gibberish filenames - we rely on title and content validation
    
    return score


def validate_pdf_is_brsr(pdf_path: Path, company_name: str, year: str) -> bool:
    """
    Strict 3-point validation to verify PDF is the correct BRSR report for the company and year.
    
    Args:
        pdf_path: Path to PDF file
        company_name: Company name to verify (e.g., "Reliance Industries Limited")
        year: Financial year to verify (e.g., "2023-24")
        
    Returns:
        True only if ALL checks pass:
        - Check 1: Contains BRSR content keywords
        - Check 2: Contains company name (at least one significant word)
        - Check 3: Contains the year (full year or start/end years)
    """
    try:
        # Extract text from first 5 pages
        text_sample = ""
        
        # Try pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # Extract text from first 5 pages
                for page_num in range(min(5, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    text_sample += page.extract_text() or ""
        except ImportError:
            # Fallback to PyMuPDF if pdfplumber not available
            try:
                import fitz
                doc = fitz.open(pdf_path)
                for page_num in range(min(5, len(doc))):
                    page = doc[page_num]
                    text_sample += page.get_text() or ""
                doc.close()
            except ImportError:
                logger.warning("No PDF library available, cannot validate PDF content")
                return False
        
        if not text_sample:
            logger.debug("Could not extract text from PDF")
            return False
        
        text_lower = text_sample.lower()
        
        # Check 1 (Content): Must contain BRSR-related keywords
        content_keywords = [
            'business responsibility',
            'brsr',
            'sustainability report',
            'section a'
        ]
        has_content = any(keyword in text_lower for keyword in content_keywords)
        
        if not has_content:
            logger.debug("Check 1 failed: PDF does not contain BRSR content keywords")
            return False
        
        # Check 2 (Company): Must contain at least one significant word from company name
        # Ignore common words: Limited, Ltd, India, Private, Public, Corporation, Corp, etc.
        common_words = {
            'limited', 'ltd', 'ltd.', 'india', 'private', 'public', 'corporation', 
            'corp', 'corp.', 'inc', 'inc.', 'incorporated', 'company', 'co', 'co.',
            'industries', 'group', 'enterprises', 'solutions', 'services', 'systems'
        }
        
        # Extract significant words from company name
        company_words = company_name.lower().split()
        significant_words = [word for word in company_words if len(word) > 2 and word not in common_words]
        
        # Remove common suffixes/prefixes
        significant_words = [
            word.rstrip('.,;:') for word in significant_words 
            if word.rstrip('.,;:') not in common_words
        ]
        
        # Check if at least one significant word appears in the text
        has_company = False
        if significant_words:
            has_company = any(word in text_lower for word in significant_words)
        else:
            # If no significant words (edge case), check if any company word appears
            has_company = any(word in text_lower for word in company_words if len(word) > 2)
        
        if not has_company:
            logger.debug(f"Check 2 failed: PDF does not contain company name (checked: {significant_words[:3]})")
            return False
        
        # Check 3 (Year): Must contain the year (e.g., "2023-24") OR start/end years (e.g., "2023" and "2024")
        year_parts = year.split('-') if '-' in year else [year]
        year_start = year_parts[0] if year_parts else ""
        year_end_short = year_parts[1] if len(year_parts) > 1 else ""
        
        # Calculate end year from start year (e.g., 2023 -> 2024)
        year_end = ""
        if year_start and year_end_short:
            # Extract last two digits of start year
            try:
                start_year_int = int(year_start)
                # End year is next year (e.g., 2023-24 means 2023 to 2024)
                year_end = str(start_year_int + 1)
            except ValueError:
                pass
        
        # Check for year in various formats
        has_year = False
        if year in text_sample or year.replace('-', '_') in text_sample or year.replace('-', '/') in text_sample:
            has_year = True
        elif year_start and year_end and year_start in text_sample and year_end in text_sample:
            # Both start and end years present (e.g., "2023" and "2024")
            has_year = True
        elif year_start and year_start in text_sample:
            # At least start year present
            has_year = True
        
        if not has_year:
            logger.debug(f"Check 3 failed: PDF does not contain year {year} (checked for: {year}, {year_start}, {year_end})")
            return False
        
        # All checks passed
        logger.debug("All 3 validation checks passed")
        return True
        
    except Exception as e:
        logger.debug(f"Error validating PDF content: {e}")
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
    
    # Score & Sort: Apply scoring function to all results
    logger.info(f"Scoring {len(pdf_results)} PDF results...")
    scored_results = []
    junk_count = 0
    
    for result in pdf_results:
        score = score_search_result(result)
        result['_score'] = score
        
        if score < 0:
            junk_count += 1
            logger.debug(f"Junk document (score: {score}): {result.get('title', 'N/A')[:60]}")
        else:
            scored_results.append(result)
    
    if not scored_results:
        error_msg = f"All {len(pdf_results)} results were filtered as junk"
        logger.warning(f"{company_name} ({year}): {error_msg}")
        return False, error_msg
    
    # Sort by score (descending - highest score first)
    scored_results.sort(key=lambda x: x.get('_score', 0), reverse=True)
    
    logger.info(f"Scored {len(scored_results)} valid candidates (filtered {junk_count} junk documents)")
    logger.debug(f"Top 3 results by score:")
    for i, result in enumerate(scored_results[:3], 1):
        logger.debug(f"  {i}. Score: {result.get('_score', 0)}, Title: {result.get('title', 'N/A')[:60]}")
    
    # Loop & Validate: Download to temp path, validate, then move to final location
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Try top 5 candidates (strict validation - download and verify each)
    max_attempts = min(5, len(scored_results))
    
    for i, result in enumerate(scored_results[:max_attempts], 1):
        pdf_url = result['link']
        score = result.get('_score', 0)
        title = result.get('title', 'N/A')[:60]
        logger.info(f"Attempting candidate {i}/{max_attempts} (score: {score}): {title}")
        
        # Download to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=output_path_obj.parent) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Download to temp path
            success, error = download_pdf(pdf_url, temp_path)
            
            if not success:
                logger.debug(f"Download failed: {error}, trying next candidate...")
                if temp_path.exists():
                    temp_path.unlink()
                continue
            
            # Verify it's a valid PDF file
            if not temp_path.exists() or temp_path.stat().st_size < 1000:
                logger.debug(f"Downloaded file is too small or missing, trying next candidate...")
                if temp_path.exists():
                    temp_path.unlink()
                continue
            
            # Check first bytes are PDF
            with open(temp_path, 'rb') as f:
                first_bytes = f.read(4)
                if first_bytes != b'%PDF':
                    logger.debug(f"Downloaded file is not a valid PDF, trying next candidate...")
                    temp_path.unlink()
                    continue
            
            # CRITICAL: Strict 3-point validation (Content, Company, Year)
            is_valid_brsr = validate_pdf_is_brsr(temp_path, company_name, year)
            
            if is_valid_brsr:
                # Valid BRSR report for correct company and year - move to final location
                if output_path_obj.exists():
                    output_path_obj.unlink()  # Remove existing file if any
                temp_path.rename(output_path_obj)
                logger.info(f"✓ Success: Valid BRSR PDF downloaded and validated (candidate {i}, score: {score})")
                return True, None
            else:
                # Validation failed - wrong company/year or not a BRSR report
                logger.warning(f"Validation Failed (Wrong Company/Year): Candidate {i} failed 3-point check, trying next candidate...")
                temp_path.unlink()
                continue
                
        except Exception as e:
            logger.error(f"Error processing candidate {i}: {e}, trying next candidate...")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            continue
    
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

