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
    """Extract domain from website URL."""
    if not website:
        return ""
    website = website.replace('https://', '').replace('http://', '').strip()
    if website.startswith('www.'):
        website = website[4:]
    website = website.rstrip('/')
    parts = website.split('/')
    return parts[0]


def build_search_query(company_name: str, website: str, year: str) -> str:
    """
    Build Google search query with NEGATIVE keywords to filter junk at source.
    Uses a more flexible approach to avoid over-filtering.
    """
    # Reduced negative keywords - only the most critical ones
    # Too many negatives can cause 0 results
    negatives = '-presentation -transcript -earnings'
    
    domain = extract_domain(website) if website else ""
    
    # Extract year start for more flexible matching (e.g., "2023" from "2023-24")
    year_start = year.split('-')[0] if '-' in year else year
    
    if domain:
        # Use site: search with simplified query
        # Try BRSR first, then fallback to broader search
        query = f'site:{domain} filetype:pdf (BRSR OR "Business Responsibility") {year_start} {negatives}'
    else:
        # Search with company name - use simpler OR condition
        # Remove quotes from company name to allow partial matches
        query = f'{company_name} filetype:pdf (BRSR OR "Business Responsibility" OR "Sustainability Report") {year_start} {negatives}'
    
    return query


def search_with_google_api(query: str, num_results: int = 10) -> List[dict]:
    """Search using Google Custom Search API."""
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        logger.warning("Google Search API key or Engine ID not configured. Skipping API search.")
        return []
    
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_SEARCH_API_KEY,
            'cx': GOOGLE_SEARCH_ENGINE_ID,
            'q': query,
            'num': min(num_results, 10)
        }
        
        logger.debug(f"Searching Google API with query: {query}")
        response = requests.get(url, params=params, timeout=GOOGLE_SEARCH_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            results = [{'link': item.get('link'), 'title': item.get('title')} for item in items]
            logger.info(f"Google API returned {len(results)} results")
            
            # Debug: Log if searchInfo shows total results
            search_info = data.get('searchInformation', {})
            total_results = search_info.get('totalResults', '0')
            logger.debug(f"Google API searchInfo: totalResults={total_results}, queryTime={search_info.get('searchTime', 'N/A')}")
            
            return results
        else:
            error_text = response.text[:500]  # Show more of the error
            logger.warning(f"Google API returned status {response.status_code}: {error_text}")
            return []
            
    except Exception as e:
        logger.error(f"Error searching with Google API: {e}")
        return []


def search_with_web_scraping(query: str, num_results: int = 10) -> List[dict]:
    """Search using web scraping (fallback)."""
    try:
        from bs4 import BeautifulSoup
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
        
        logger.debug(f"Searching Google with web scraping: {query}")
        response = requests.get(search_url, headers=headers, timeout=GOOGLE_SEARCH_TIMEOUT)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.select('div.g')[:num_results]:
                link_elem = result.select_one('a[href^="http"]')
                title_elem = result.select_one('h3')
                if link_elem and title_elem:
                    href = link_elem.get('href', '')
                    if href.startswith('/url?q='):
                        href = href.split('&')[0][7:]
                    results.append({'link': href, 'title': title_elem.get_text(strip=True)})
            logger.info(f"Web scraping returned {len(results)} results")
            return results
        else:
            logger.warning(f"Google search returned status {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error scraping Google search: {e}")
        return []


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    if not url: return False
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith('.pdf') or 'pdf' in path or 'pdf' in parsed.query.lower()


def score_search_result(result: Dict) -> int:
    """
    Score a search result based on title and link.
    Returns: Score (higher is better). Negative scores indicate junk.
    """
    title = result.get('title', '').lower()
    link = result.get('link', '').lower()
    score = 0
    
    # 1. KILL JUNK IMMEDIATELY (-1000)
    # Check config keywords
    for keyword in NEGATIVE_KEYWORDS:
        if keyword.lower() in title or keyword.lower() in link:
            return -1000
    
    # Check hardcoded junk words
    junk_words = ['presentation', 'investor', 'earnings', 'call', 'transcript', 'release', 'brief', 'outcome', 'results']
    for word in junk_words:
        if word in title or word in link:
            return -1000
    
    # Check for Annual Reports (if strict standalone policy is on)
    if 'integrated report' in title or 'integrated annual report' in title:
        return -1000
    if 'annual report' in title:
        return -1000

    # 2. REWARD GOOD TITLES
    if 'business responsibility' in title: score += 100
    if 'brsr' in title: score += 100
    if 'sustainability report' in title: score += 100
    if 'esg report' in title: score += 80
    
    # 3. REWARD URL SIGNALS
    if 'brsr' in link: score += 20
    if 'sustainability' in link: score += 10
    
    return score


def validate_pdf_is_brsr(pdf_path: Path, company_name: str, year: str, title_score: int = 0) -> Tuple[bool, str]:
    """
    Forensic Validation with "High Confidence Bypass".
    Returns: (Success: bool, Reason: str)
    """
    try:
        # --- SETUP: EXTRACT TEXT ---
        pages_text = []
        full_text_15 = ""
        page_count = 0
        
        # Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                # Standalone BRSRs are usually < 150 pages. 
                # If > 200, it's almost certainly an Annual Report (unless specifically whitelisted).
                if page_count > 200:
                    return False, f"Page count {page_count} too high for standalone BRSR (>200)"
                
                for page_num in range(min(15, len(pdf.pages))):
                    text = pdf.pages[page_num].extract_text() or ""
                    pages_text.append(text)
                    full_text_15 += text
        except:
            # Fallback to PyMuPDF
            try:
                import fitz
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                if page_count > 200:
                    return False, f"Page count {page_count} too high (>200)"
                for page_num in range(min(15, len(doc))):
                    text = doc[page_num].get_text() or ""
                    pages_text.append(text)
                    full_text_15 += text
                doc.close()
            except:
                return False, "No PDF library available"

        full_text_lower = full_text_15.lower()
        page1_lower = pages_text[0].lower() if pages_text else ""
        
        # --- POISON PILL CHECK (Always Enforced) ---
        # Even if title is perfect, reject if it contains Annual Report sections
        poison_pills = [
            "independent auditor's report", 
            "standalone financial statements", 
            "consolidated financial statements", 
            "profit and loss account"
        ]
        for pill in poison_pills:
            if pill in full_text_lower:
                return False, f"Found Poison Pill: '{pill}' (Looks like Annual Report)"

        # --- HIGH CONFIDENCE BYPASS ---
        # If title strongly indicates BRSR (Score >= 100), we relax the strict checks.
        is_high_confidence = title_score >= 100
        
        if is_high_confidence:
            logger.info(f"High Confidence Title (Score {title_score}). Relaxing validation...")
            
            # Scenario A: Scanned PDF (No text extracted)
            if len(full_text_lower.strip()) < 50:
                # If title says "BRSR" and file is an image, we TRUST the title.
                return True, "Success: High confidence title + Scanned PDF (Bypass text check)"
            
            # Scenario B: Text exists, check Year and Company (but still require proper company match)
            
            # Check Year (Loose)
            year_parts = year.split('-')
            y_start = year_parts[0]
            if y_start not in full_text_lower:
                # If even the year start (e.g. 2023) is missing, it's definitely wrong
                return False, f"High Confidence Failed: Year {y_start} not found anywhere in 15 pages"
            
            # Check Company (STRICT - even for high confidence, we must verify correct company)
            # Extract significant words from company name
            common_words = {'limited', 'ltd', 'ltd.', 'india', 'private', 'public', 'corporation', 
                          'corp', 'corp.', 'inc', 'inc.', 'incorporated', 'company', 'co', 'co.',
                          'industries', 'group', 'enterprises', 'solutions', 'services', 'systems'}
            
            company_words = [w.rstrip('.,;:').lower() for w in company_name.split() 
                           if len(w.rstrip('.,;:')) > 2 and w.rstrip('.,;:').lower() not in common_words]
            
            # Check if company name appears on cover page (pages 1-2) - this is critical
            pages_1_2 = (pages_text[0] if len(pages_text) > 0 else "") + (pages_text[1] if len(pages_text) > 1 else "")
            pages_1_2_lower = pages_1_2.lower()
            
            # Strategy 1: Check if full company name (or most of it) appears on cover page
            company_name_lower = company_name.lower()
            has_full_name_on_cover = company_name_lower in pages_1_2_lower
            
            # Strategy 2: If not full name, require at least 2 significant words on cover page
            # OR all significant words anywhere in first 15 pages
            if not has_full_name_on_cover:
                if len(company_words) >= 2:
                    # For multi-word names, require at least 2 words on cover page
                    found_on_cover = sum(1 for word in company_words if word in pages_1_2_lower)
                    if found_on_cover < 2:
                        # Fallback: Check if all words exist anywhere in first 15 pages
                        found_anywhere = sum(1 for word in company_words if word in full_text_lower)
                        if found_anywhere < len(company_words):
                            return False, f"High Confidence Failed: Company name mismatch (required: {company_words}, found on cover: {found_on_cover}, found anywhere: {found_anywhere})"
                elif len(company_words) == 1:
                    # Single significant word - must appear on cover page
                    if company_words[0] not in pages_1_2_lower:
                        return False, f"High Confidence Failed: Company word '{company_words[0]}' not found on cover page"
                else:
                    # No significant words (all common words) - require full name
                    if company_name_lower not in full_text_lower:
                        return False, f"High Confidence Failed: Full company name '{company_name}' not found"
            
            return True, "Success: High confidence title + Company verified"

        # --- LOW CONFIDENCE: STRICT FORENSIC CHECK ---
        # If title is generic (e.g. "Report.pdf"), we must be strict.
        
        pages_1_2 = (pages_text[0] if len(pages_text)>0 else "") + (pages_text[1] if len(pages_text)>1 else "")
        pages_1_2_lower = pages_1_2.lower()
        
        # 1. Year on Cover (Page 1-2)
        if year.split('-')[0] not in pages_1_2_lower:
             return False, "Strict Check Failed: Year not found on Cover (Pages 1-2)"
        
        # 2. Company Name on Cover (Page 1-2) - CRITICAL to prevent wrong company downloads
        company_name_lower = company_name.lower()
        common_words = {'limited', 'ltd', 'ltd.', 'india', 'private', 'public', 'corporation', 
                       'corp', 'corp.', 'inc', 'inc.', 'incorporated', 'company', 'co', 'co.',
                       'industries', 'group', 'enterprises', 'solutions', 'services', 'systems'}
        
        company_words = [w.rstrip('.,;:').lower() for w in company_name.split() 
                        if len(w.rstrip('.,;:')) > 2 and w.rstrip('.,;:').lower() not in common_words]
        
        # Must have full company name OR at least 2 significant words on cover page
        has_full_name = company_name_lower in pages_1_2_lower
        if not has_full_name:
            if len(company_words) >= 2:
                found_on_cover = sum(1 for word in company_words if word in pages_1_2_lower)
                if found_on_cover < 2:
                    return False, f"Strict Check Failed: Company name not found on cover (required: {company_words}, found: {found_on_cover})"
            elif len(company_words) == 1:
                if company_words[0] not in pages_1_2_lower:
                    return False, f"Strict Check Failed: Company word '{company_words[0]}' not found on cover"
            else:
                # All common words - require full name
                if company_name_lower not in pages_1_2_lower:
                    return False, f"Strict Check Failed: Full company name '{company_name}' not found on cover"
             
        # 3. Report DNA (Must have 'Section A' and 'Principle')
        if 'section a' not in full_text_lower or 'principle' not in full_text_lower:
            return False, "Strict Check Failed: Missing BRSR DNA (Section A / Principle)"
            
        return True, "Success: Passed Strict Forensic Check"

    except Exception as e:
        return False, f"Validation Exception: {str(e)}"


def download_pdf(url: str, output_path: Path, timeout: int = 30) -> Tuple[bool, Optional[str]]:
    """Download PDF from URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                if response.content[:4] != b'%PDF':
                    return False, "URL is not a PDF"
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True, None
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


def get_google_search_report(
    company_name: str,
    website: str,
    year: str,
    output_path: Path
) -> Tuple[bool, Optional[str]]:
    """
    Download BRSR report using Google Search.
    """
    query = build_search_query(company_name, website, year)
    logger.info(f"Searching Google: {query}")
    
    # Try API first, then scraping
    results = search_with_google_api(query, num_results=GOOGLE_SEARCH_MAX_RESULTS)
    if not results:
        results = search_with_web_scraping(query, num_results=GOOGLE_SEARCH_MAX_RESULTS)
    
    # If still no results, try a simpler fallback query (less restrictive)
    if not results:
        logger.info(f"Primary query returned 0 results, trying simpler fallback query...")
        year_start = year.split('-')[0] if '-' in year else year
        fallback_query = f'{company_name} BRSR {year_start} filetype:pdf'
        logger.info(f"Fallback query: {fallback_query}")
        
        results = search_with_google_api(fallback_query, num_results=GOOGLE_SEARCH_MAX_RESULTS)
        if not results:
            results = search_with_web_scraping(fallback_query, num_results=GOOGLE_SEARCH_MAX_RESULTS)
    
    if not results:
        return False, "No results found"
    
    # Filter PDFs
    pdf_results = [r for r in results if is_pdf_url(r.get('link', ''))]
    if not pdf_results:
        return False, "No PDF links found"
    
    # Score Results
    scored_results = []
    for result in pdf_results:
        score = score_search_result(result)
        result['_score'] = score
        if score > -1000: # Only keep non-junk
            scored_results.append(result)
            
    scored_results.sort(key=lambda x: x.get('_score', 0), reverse=True)
    
    if not scored_results:
        return False, "All results were junk (Annual Reports/Presentations)"

    # Try ALL valid candidates (increased from 5 to ensure we find it)
    max_attempts = len(scored_results)
    
    for i, result in enumerate(scored_results, 1):
        score = result.get('_score', 0)
        title = result.get('title', 'N/A')[:60]
        logger.info(f"Candidate {i}/{max_attempts} (Score: {score}): {title}")
        
        # Temp download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            temp_path = Path(tmp.name)
            
        try:
            success, error = download_pdf(result['link'], temp_path)
            if not success:
                logger.debug(f"  -> Download failed: {error}")
                continue
                
            # Validate with detailed feedback
            is_valid, reason = validate_pdf_is_brsr(temp_path, company_name, year, title_score=score)
            
            if is_valid:
                logger.info(f"  -> VALID: {reason}")
                if Path(output_path).exists(): Path(output_path).unlink()
                temp_path.rename(output_path)
                return True, None
            else:
                logger.warning(f"  -> INVALID: {reason}")
                temp_path.unlink()
                
        except Exception as e:
            logger.error(f"Error checking candidate {i}: {e}")
            if temp_path.exists(): temp_path.unlink()
            
    return False, "All candidates failed validation"


class GoogleSearchDownloader:
    def __init__(self): pass
    def download(self, company_name, website, year, output_path):
        return get_google_search_report(company_name, website, year, output_path)

if __name__ == "__main__":
    # Test block
    pass
