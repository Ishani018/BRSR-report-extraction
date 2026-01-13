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
    
    # STRICT POLICY: Standalone BRSR Only - Penalize Annual Reports and Integrated Reports
    # Only reward standalone BRSR-related keywords
    if 'business responsibility' in title or 'brsr' in title or 'sustainability report' in title or 'esg report' in title:
        score += 100
    
    # Penalize Annual Reports and Integrated Reports (treat as junk)
    if 'integrated report' in title or 'integrated annual report' in title:
        logger.debug(f"Penalizing Integrated Report in '{result.get('title', 'N/A')[:50]}'")
        return -1000  # Kill immediately - not a standalone BRSR
    
    if 'annual report' in title:
        logger.debug(f"Penalizing Annual Report in '{result.get('title', 'N/A')[:50]}'")
        return -1000  # Kill immediately - not a standalone BRSR
    
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
    Extremely thorough 4-layer forensic validation to verify PDF is the correct standalone BRSR report.
    
    Args:
        pdf_path: Path to PDF file
        company_name: Company name to verify (e.g., "Reliance Industries Limited")
        year: Financial year to verify (e.g., "2023-24")
        
    Returns:
        True only if ALL 4 layers pass:
        - Layer 1: Cover Page Check (year on pages 1-2)
        - Layer 2: Sector-Aware Company Match (on pages 1-3)
        - Layer 3: Report Type DNA Check (BRSR structure, no Annual Report)
        - Layer 4: Year Dominance Check (full 15-page scan)
    """
    try:
        # Setup & Extraction: Read first 15 pages
        pages_text = []  # List to store each page's text separately
        page1_text = ""
        page2_text = ""
        pages_1_2_text = ""  # Combined text from pages 1-2 for cover page check
        pages_1_3_text = ""  # Combined text from pages 1-3 for company check
        full_text_15 = ""  # Full text from first 15 pages for year dominance
        
        pdf_metadata = None
        page_count = 0
        
        # Try pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # Page count check: Standalone BRSRs are usually < 100 pages, reject > 150 pages
                page_count = len(pdf.pages)
                if page_count > 150:
                    logger.debug(f"Setup failed: PDF has {page_count} pages (Standalone BRSRs are typically < 100 pages, > 150 suggests Annual Report)")
                    return False
                
                # Read PDF metadata (Title field as fallback)
                pdf_metadata = pdf.metadata if hasattr(pdf, 'metadata') else None
                
                # Extract text from first 15 pages, keeping pages separate
                for page_num in range(min(15, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
                    full_text_15 += page_text
                    
                    if page_num == 0:
                        page1_text = page_text
                    elif page_num == 1:
                        page2_text = page_text
                
                # Combine pages 1-2 for cover page check
                pages_1_2_text = page1_text + page2_text
                
                # Combine pages 1-3 for company check
                pages_1_3_text = page1_text + page2_text
                if len(pages_text) > 2:
                    pages_1_3_text += pages_text[2]
                    
        except ImportError:
            # Fallback to PyMuPDF if pdfplumber not available
            try:
                import fitz
                doc = fitz.open(pdf_path)
                
                # Page count check
                page_count = len(doc)
                if page_count > 150:
                    logger.debug(f"Setup failed: PDF has {page_count} pages (Standalone BRSRs are typically < 100 pages, > 150 suggests Annual Report)")
                    doc.close()
                    return False
                
                # Read PDF metadata
                pdf_metadata = doc.metadata if hasattr(doc, 'metadata') else None
                
                # Extract text from first 15 pages
                for page_num in range(min(15, len(doc))):
                    page = doc[page_num]
                    page_text = page.get_text() or ""
                    pages_text.append(page_text)
                    full_text_15 += page_text
                    
                    if page_num == 0:
                        page1_text = page_text
                    elif page_num == 1:
                        page2_text = page_text
                
                # Combine pages 1-2 and 1-3
                pages_1_2_text = page1_text + page2_text
                pages_1_3_text = page1_text + page2_text
                if len(pages_text) > 2:
                    pages_1_3_text += pages_text[2]
                
                doc.close()
            except ImportError:
                logger.warning("No PDF library available, cannot validate PDF content")
                return False
        
        if not full_text_15:
            logger.debug("Setup failed: Could not extract text from PDF")
            return False
        
        # Convert to lowercase for matching
        page1_lower = page1_text.lower()
        pages_1_2_lower = pages_1_2_text.lower()
        pages_1_3_lower = pages_1_3_text.lower()
        full_text_lower = full_text_15.lower()
        
        # ====================================================================
        # LAYER 1: The "Cover Page" Check (Pages 1-2 Only)
        # ====================================================================
        # Build year patterns for cover page check
        year_parts = year.split('-') if '-' in year else [year]
        year_start = year_parts[0] if year_parts else ""
        year_end_short = year_parts[1] if len(year_parts) > 1 else ""
        
        target_year_patterns = []
        if year_end_short:
            target_year_patterns.extend([year, year.replace('-', '_'), year.replace('-', '/')])
            target_year_patterns.extend([f"fy{year_end_short}", f"fy {year_end_short}", f"fy-{year_end_short}"])
        
        if year_start and year_end_short:
            try:
                start_int = int(year_start)
                end_int = start_int + 1
                full_year_format = f"{start_int}-{end_int}"
                target_year_patterns.extend([full_year_format, full_year_format.replace('-', '_'), full_year_format.replace('-', '/')])
            except ValueError:
                pass
        
        # Check if target year appears on pages 1-2
        year_found_on_cover = False
        for pattern in target_year_patterns:
            if pattern.lower() in pages_1_2_lower:
                year_found_on_cover = True
                break
        
        if not year_found_on_cover:
            logger.warning(f"Layer 1 failed: Target year {year} not found on cover pages (pages 1-2)")
            return False
        
        # ====================================================================
        # LAYER 2: Strict "Sector-Aware" Company Match
        # ====================================================================
        # Clean company name (remove common words)
        common_words = {
            'limited', 'ltd', 'ltd.', 'india', 'private', 'public', 'corporation', 
            'corp', 'corp.', 'inc', 'inc.', 'incorporated', 'company', 'co', 'co.',
            'industries', 'group', 'enterprises', 'solutions', 'services', 'systems'
        }
        
        company_words = company_name.lower().split()
        cleaned_words = [word.rstrip('.,;:') for word in company_words 
                        if len(word.rstrip('.,;:')) > 2 and word.rstrip('.,;:') not in common_words]
        
        # Blacklist generic sector names
        generic_names = {'bank', 'power', 'infra', 'finance', 'capital', 'global', 'infrastructure', 
                        'financial', 'insurance', 'cement', 'pharma', 'pharmaceuticals', 'energy', 
                        'realty', 'housing', 'holdings', 'investment'}
        
        # Check if cleaned name is generic
        is_generic = len(cleaned_words) == 1 and cleaned_words[0] in generic_names
        
        if is_generic:
            # For generic names, require full original company name on pages 1-3
            company_name_lower = company_name.lower()
            has_full_name = company_name_lower in pages_1_3_lower
            if not has_full_name:
                logger.warning(f"Layer 2 failed: Generic company name '{cleaned_words[0]}' requires full name '{company_name}' on pages 1-3, not found")
                return False
        else:
            # For non-generic names, require all cleaned words on pages 1-3
            found_words = []
            for word in cleaned_words:
                if word in pages_1_3_lower:
                    found_words.append(word)
            
            if len(found_words) < len(cleaned_words):
                logger.warning(f"Layer 2 failed: Company name match incomplete (required: {cleaned_words}, found: {found_words} on pages 1-3)")
                return False
        
        # ====================================================================
        # LAYER 3: The "Report Type" DNA Check
        # ====================================================================
        # Must-Have 1: "Section A" AND ("General Information" OR "Details of the Listed Entity")
        has_section_a = 'section a' in full_text_lower
        has_general_info = 'general information' in full_text_lower
        has_listed_entity = 'details of the listed entity' in full_text_lower
        
        if not has_section_a:
            logger.warning("Layer 3 failed: PDF does not contain 'Section A'")
            return False
        
        if not (has_general_info or has_listed_entity):
            logger.warning("Layer 3 failed: PDF does not contain 'General Information' or 'Details of the Listed Entity'")
            return False
        
        # Must-Have 2: "Principle" (e.g., "Principle 1", "Principle 9")
        has_principle = 'principle' in full_text_lower
        if not has_principle:
            logger.warning("Layer 3 failed: PDF does not contain 'Principle' (e.g., Principle 1, Principle 9)")
            return False
        
        # Rejection: Junk documents
        junk_indicators = ['investor presentation', 'earnings call', 'transcript', 'press release']
        for indicator in junk_indicators:
            if indicator in full_text_lower:
                logger.warning(f"Layer 3 failed: PDF contains junk indicator '{indicator}'")
                return False
        
        # Strict "No Annual Report" Rule: Check Page 1
        if page1_lower:
            # Check for "Integrated Annual Report" or "Annual Report 20..." (but allow "Extract" or "Annexure")
            if 'integrated annual report' in page1_lower:
                logger.warning("Layer 3 failed: Page 1 contains 'Integrated Annual Report'")
                return False
            
            # Check for "Annual Report 20..." but exclude if it's an extract/annexure
            if 'annual report 20' in page1_lower:
                # Allow if it's clearly an extract or annexure
                if 'extract' not in page1_lower and 'annexure' not in page1_lower:
                    logger.warning("Layer 3 failed: Page 1 contains 'Annual Report 20...' (not an extract/annexure)")
                    return False
        
        # Also check full text for "Integrated Annual Report"
        if 'integrated annual report' in full_text_lower:
            logger.warning("Layer 3 failed: PDF contains 'Integrated Annual Report'")
            return False
        
        # ====================================================================
        # LAYER 4: Year Dominance (Full Text - 15 Pages)
        # ====================================================================
        # Calculate previous year
        previous_year = ""
        if year_start:
            try:
                start_year_int = int(year_start)
                prev_start = str(start_year_int - 1)
                if year_end_short:
                    prev_end_short = str(int(year_end_short) - 1)
                    if len(prev_end_short) == 1:
                        prev_end_short = '0' + prev_end_short
                    previous_year = f"{prev_start}-{prev_end_short}"
            except ValueError:
                pass
        
        # Build year patterns for full text scan
        target_year_strings = []
        previous_year_strings = []
        
        if year_end_short:
            target_year_strings.extend([year, year.replace('-', '_'), year.replace('-', '/')])
            target_year_strings.extend([f"fy{year_end_short}", f"fy {year_end_short}", f"fy-{year_end_short}"])
            if previous_year:
                previous_year_strings.extend([previous_year, previous_year.replace('-', '_'), previous_year.replace('-', '/')])
                prev_end = previous_year.split('-')[1] if '-' in previous_year else ""
                if prev_end:
                    previous_year_strings.extend([f"fy{prev_end}", f"fy {prev_end}", f"fy-{prev_end}"])
        
        if year_start and year_end_short:
            try:
                start_int = int(year_start)
                end_int = start_int + 1
                full_year_format = f"{start_int}-{end_int}"
                target_year_strings.extend([full_year_format, full_year_format.replace('-', '_'), full_year_format.replace('-', '/')])
                
                if previous_year:
                    prev_start_int = start_int - 1
                    prev_end_int = prev_start_int + 1
                    prev_full_format = f"{prev_start_int}-{prev_end_int}"
                    previous_year_strings.extend([prev_full_format, prev_full_format.replace('-', '_'), prev_full_format.replace('-', '/')])
            except ValueError:
                pass
        
        # Count occurrences across all 15 pages
        target_year_count = 0
        previous_year_count = 0
        
        for year_str in target_year_strings:
            target_year_count += full_text_lower.count(year_str.lower())
        
        for year_str in previous_year_strings:
            previous_year_count += full_text_lower.count(year_str.lower())
        
        # Year Dominance Check: Reject if previous year appears significantly more
        if previous_year_count > target_year_count:
            logger.warning(f"Layer 4 failed: Year dominance check failed (target {year}: {target_year_count}, previous {previous_year}: {previous_year_count})")
            return False
        
        # Must have at least one occurrence of target year
        if target_year_count == 0:
            logger.warning(f"Layer 4 failed: Target year {year} not found in full 15-page scan")
            return False
        
        # All 4 layers passed
        logger.debug(f"All 4 validation layers passed (company: {cleaned_words if not is_generic else 'full name'}, year: {year} found {target_year_count}x, pages: {page_count})")
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
            
            # CRITICAL: Strict 4-layer forensic validation (Cover Page, Company, Report Type DNA, Year Dominance)
            is_valid_brsr = validate_pdf_is_brsr(temp_path, company_name, year)
            
            if is_valid_brsr:
                # Valid standalone BRSR report for correct company and year - move to final location
                if output_path_obj.exists():
                    output_path_obj.unlink()  # Remove existing file if any
                temp_path.rename(output_path_obj)
                logger.info(f"✓ Success: Valid standalone BRSR PDF downloaded and validated (candidate {i}, score: {score})")
                return True, None
            else:
                # Validation failed - wrong company/year, not a standalone BRSR, or failed one of the 4 layers
                logger.warning(f"Validation Failed: Candidate {i} failed 4-layer forensic check (Cover Page/Company/Report Type/Year), trying next candidate...")
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

