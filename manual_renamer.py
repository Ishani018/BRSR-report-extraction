"""
Manual PDF Renamer - Identify and rename manually downloaded BRSR PDFs.

This script processes PDFs from a manual_downloads/ folder, identifies the company
and year, then renames them according to the Excel naming convention.
"""
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.company_reader import CompanyReader, read_companies
from config.config import BRSR_FINANCIAL_YEARS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ManualPDFRenamer:
    """Identify and rename manually downloaded PDFs."""
    
    def __init__(self, excel_path: Path, input_folder: Path, output_folder: Path):
        """
        Initialize the renamer.
        
        Args:
            excel_path: Path to NIFTY 500 firms.xlsx
            input_folder: Folder containing manually downloaded PDFs
            output_folder: Folder to move renamed PDFs to
        """
        self.excel_path = Path(excel_path)
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Create folders if missing
        self.input_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Load company data
        logger.info(f"Loading company data from {excel_path}")
        self.company_reader = read_companies(excel_path)
        self.companies_df = self.company_reader.get_companies_dataframe()
        
        # Build symbol lookup (symbol -> company row)
        self.symbol_lookup = {}
        for _, row in self.companies_df.iterrows():
            symbol = str(row['symbol']).strip().upper()
            if symbol:
                self.symbol_lookup[symbol] = row.to_dict()
        
        logger.info(f"Loaded {len(self.companies_df)} companies")
    
    def extract_text_from_pdf(self, pdf_path: Path, max_pages: int = 5) -> str:
        """
        Extract text from first N pages of PDF.
        Tries pdfplumber first, then PyMuPDF, then OCR as fallback.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to extract
            
        Returns:
            Extracted text
        """
        # Try pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page_num in range(min(max_pages, len(pdf.pages))):
                    page_text = pdf.pages[page_num].extract_text() or ""
                    text_parts.append(page_text)
                extracted_text = "\n".join(text_parts)
                
                # If pdfplumber returned very little text, try PyMuPDF as fallback
                if len(extracted_text.strip()) < 50:
                    logger.debug(f"pdfplumber extracted only {len(extracted_text)} chars, trying PyMuPDF...")
                    try:
                        import fitz
                        doc = fitz.open(pdf_path)
                        text_parts_pymupdf = []
                        for page_num in range(min(max_pages, len(doc))):
                            page_text = doc[page_num].get_text() or ""
                            text_parts_pymupdf.append(page_text)
                        doc.close()
                        extracted_text_pymupdf = "\n".join(text_parts_pymupdf)
                        if len(extracted_text_pymupdf.strip()) > len(extracted_text.strip()):
                            logger.debug(f"PyMuPDF extracted more text ({len(extracted_text_pymupdf)} chars), using it")
                            extracted_text = extracted_text_pymupdf
                    except Exception:
                        pass  # Continue with pdfplumber result
                
                # If still insufficient text, try OCR
                if len(extracted_text.strip()) < 50:
                    logger.info("Text extraction methods failed, trying OCR (this may take longer)...")
                    ocr_text = self._extract_text_with_ocr(pdf_path, max_pages)
                    if ocr_text and len(ocr_text.strip()) > len(extracted_text.strip()):
                        logger.info(f"OCR extracted {len(ocr_text)} characters successfully")
                        return ocr_text
                
                return extracted_text
        except ImportError:
            # pdfplumber not available, try PyMuPDF
            pass
        except Exception as e:
            logger.debug(f"pdfplumber failed for {pdf_path.name}: {e}, trying PyMuPDF...")
        
        # Fallback to PyMuPDF
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text_parts = []
            for page_num in range(min(max_pages, len(doc))):
                page_text = doc[page_num].get_text() or ""
                text_parts.append(page_text)
            doc.close()
            extracted_text = "\n".join(text_parts)
            
            # If PyMuPDF also failed, try OCR
            if len(extracted_text.strip()) < 50:
                logger.info("PyMuPDF extraction insufficient, trying OCR (this may take longer)...")
                ocr_text = self._extract_text_with_ocr(pdf_path, max_pages)
                if ocr_text and len(ocr_text.strip()) > len(extracted_text.strip()):
                    logger.info(f"OCR extracted {len(ocr_text)} characters successfully")
                    return ocr_text
            
            return extracted_text
        except ImportError:
            logger.error("No PDF library available (pdfplumber or PyMuPDF), trying OCR...")
            # Try OCR as last resort
            ocr_text = self._extract_text_with_ocr(pdf_path, max_pages)
            if ocr_text:
                return ocr_text
            return ""
        except Exception as e:
            logger.warning(f"Error extracting text from {pdf_path.name} with PyMuPDF: {e}")
            # Try OCR as last resort
            logger.info("Trying OCR as fallback...")
            ocr_text = self._extract_text_with_ocr(pdf_path, max_pages)
            if ocr_text:
                return ocr_text
            return ""
    
    def _extract_text_with_ocr(self, pdf_path: Path, max_pages: int = 5) -> str:
        """
        Extract text from PDF using OCR (Optical Character Recognition).
        This is used as a fallback for scanned/image-based PDFs.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process
            
        Returns:
            Extracted text, or empty string if OCR fails or is not available
        """
        try:
            import pytesseract
            from PIL import Image
            
            # Try to use PyMuPDF to convert PDF pages to images
            try:
                import fitz
            except ImportError:
                logger.warning("PyMuPDF not available, cannot perform OCR on PDF")
                return ""
            
            doc = fitz.open(pdf_path)
            text_parts = []
            
            try:
                for page_num in range(min(max_pages, len(doc))):
                    page = doc[page_num]
                    
                    # Convert PDF page to image (300 DPI for better OCR accuracy)
                    mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    from io import BytesIO
                    img = Image.open(BytesIO(img_data))
                    
                    # Perform OCR
                    page_text = pytesseract.image_to_string(img, lang='eng')
                    text_parts.append(page_text)
                    
                    logger.debug(f"OCR processed page {page_num + 1}/{min(max_pages, len(doc))}")
            
            finally:
                doc.close()
            
            return "\n".join(text_parts)
        
        except ImportError:
            logger.warning("OCR libraries not available (pytesseract/Pillow). Install with: pip install pytesseract pillow")
            logger.warning("Also install Tesseract OCR engine: https://github.com/tesseract-ocr/tesseract")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed for {pdf_path.name}: {e}")
            return ""
    
    def match_company_by_symbol(self, filename: str) -> Optional[Dict]:
        """
        Try to match company by symbol in filename (STRICT matching).
        
        Args:
            filename: PDF filename
            
        Returns:
            Company dictionary if matched, None otherwise
        """
        filename_upper = filename.upper()
        filename_no_ext = Path(filename).stem.upper()
        
        # Priority 1: Exact symbol match at start of filename (highest confidence)
        for symbol, company_data in self.symbol_lookup.items():
            # Check if filename starts with symbol followed by separator
            if (filename_upper.startswith(symbol + '_') or 
                filename_upper.startswith(symbol + '-') or 
                filename_upper.startswith(symbol + '.') or 
                filename_upper.startswith(symbol + ' ') or
                filename_no_ext == symbol):  # Exact match (no extension)
                logger.info(f"Matched by symbol (exact start): {symbol} from filename '{filename}'")
                return company_data
        
        # Priority 2: Symbol as whole word (word boundary) in filename
        for symbol, company_data in self.symbol_lookup.items():
            # Use word boundary to ensure it's not part of a larger word
            # e.g., "TCS" should match "TCS_report.pdf" but not "ATCS_report.pdf"
            pattern = rf'\b{symbol}\b'
            if re.search(pattern, filename_no_ext):
                logger.info(f"Matched by symbol (word boundary): {symbol} from filename '{filename}'")
                return company_data
        
        return None
    
    def match_company_by_symbol_in_content(self, pdf_text: str) -> Optional[Dict]:
        """
        Try to match company by symbol in PDF content (HIGH PRIORITY, STRICT MATCHING).
        Requires symbol to appear in context with company name to avoid false matches.
        
        Args:
            pdf_text: Extracted text from PDF
            
        Returns:
            Company dictionary if matched, None otherwise
        """
        if not pdf_text:
            return None
        
        pdf_text_lower = pdf_text.lower()
        pdf_text_upper = pdf_text.upper()
        
        # Short/ambiguous symbols that might appear as exchanges or references
        # These need stronger context (company name nearby)
        ambiguous_symbols = {'BSE', 'NSE', 'SEBI', 'NIFTY', 'SENSEX', 'MCX'}
        
        # Search for symbols in PDF content (first 3 pages priority for cover page)
        first_part = pdf_text_lower[:3000]  # First ~3000 chars (roughly first 2-3 pages, cover area)
        first_part_upper = pdf_text_upper[:3000]
        
        best_match = None
        best_score = 0
        
        # Priority: Check for symbols in first part of document (higher confidence)
        for symbol, company_data in self.symbol_lookup.items():
            symbol_upper = symbol.upper()
            
            # CRITICAL: Skip ambiguous symbols entirely when matching from PDF content
            # These symbols (BSE, NSE, etc.) appear in many PDFs as regulatory references
            # and cause false matches. Only allow them to match from filenames.
            is_ambiguous = symbol_upper in ambiguous_symbols
            if is_ambiguous:
                continue  # Skip ambiguous symbols completely in PDF content matching
            
            company_name = company_data['company_name'].lower()
            company_words = [w.strip().rstrip('.,;:') for w in company_name.split() if len(w.strip()) > 2]
            
            # Use word boundary to ensure it's not part of a larger word
            pattern = rf'\b{re.escape(symbol_upper)}\b'
            
            # Check if symbol appears in first part
            symbol_match = re.search(pattern, first_part_upper)
            if not symbol_match:
                continue
            
            score = 100  # Base score for symbol in first part
            
            # Prefer if company name appears
            for word in company_words:
                if word in first_part:
                    score += 100
                    break
            
            if score > best_score:
                best_match = company_data
                best_score = score
                logger.info(f"Found symbol '{symbol}' in PDF content (first pages, score: {score})")
        
        if best_match:
            return best_match
        
        # Fallback: Check entire text if not found in first part (with even stricter rules)
        for symbol, company_data in self.symbol_lookup.items():
            symbol_upper = symbol.upper()
            company_name = company_data['company_name'].lower()
            company_words = [w.strip().rstrip('.,;:') for w in company_name.split() if len(w.strip()) > 2]
            
            is_ambiguous = symbol_upper in ambiguous_symbols
            
            # Skip ambiguous symbols in fallback (too risky)
            if is_ambiguous:
                continue
            
            pattern = rf'\b{re.escape(symbol_upper)}\b'
            if re.search(pattern, pdf_text_upper):
                # Still prefer if company name appears
                company_found = any(word in pdf_text_lower for word in company_words)
                if company_found:
                    logger.info(f"Matched by symbol in PDF content (anywhere): {symbol}")
                    return company_data
        
        return None
    
    def verify_company_match(self, pdf_text: str, company_data: Dict) -> bool:
        """
        Verify that a matched company actually appears in the PDF.
        This is critical to prevent false matches (e.g., "BSE" matching when it's just a reference to the exchange).
        
        Args:
            pdf_text: Extracted text from PDF
            company_data: Matched company dictionary
            
        Returns:
            True if company name appears prominently in PDF, False otherwise
        """
        if not pdf_text or len(pdf_text.strip()) < 50:
            return False
        
        company_name = company_data['company_name']
        symbol = company_data.get('symbol', '').upper()
        company_name_lower = company_name.lower()
        pdf_text_lower = pdf_text.lower()
        
        # Ambiguous symbols that commonly appear as references (exchanges, regulators)
        ambiguous_symbols = {'BSE', 'NSE', 'SEBI', 'NIFTY', 'SENSEX', 'MCX'}
        
        # For ambiguous symbols, STRICTLY require full company name (not just symbol)
        if symbol in ambiguous_symbols:
            # Must have full company name in first 3000 chars (cover page)
            first_part = pdf_text_lower[:3000]
            if company_name_lower not in first_part:
                return False
            return True
        
        # Check first 3000 chars (cover page area) for company name
        first_part = pdf_text_lower[:3000]
        
        # Common words to ignore
        common_words = {
            'limited', 'ltd', 'ltd.', 'india', 'private', 'public', 'corporation',
            'corp', 'corp.', 'inc', 'inc.', 'incorporated', 'company', 'co', 'co.'
        }
        
        # Extract significant words from company name
        company_words = [
            w.rstrip('.,;:').lower() 
            for w in company_name.split() 
            if len(w.rstrip('.,;:')) > 2 and w.rstrip('.,;:').lower() not in common_words
        ]
        
        # Priority 1: Full company name appears in first part (highest confidence)
        if company_name_lower in first_part:
            return True
        
        # Priority 2: For multi-word names, check if at least 2 significant words appear together
        if len(company_words) >= 2:
            found_words = sum(1 for word in company_words if word in first_part)
            if found_words >= 2:
                return True
        
        # Priority 3: For single-word names, must appear in first part
        if len(company_words) == 1:
            return company_words[0] in first_part
        
        # Priority 4: Fallback - check full text if not in first part
        if company_name_lower in pdf_text_lower:
            return True
        
        # If all checks fail, it's likely a false match
        return False
    
    def match_company_by_content(self, pdf_text: str) -> Optional[Dict]:
        """
        Match company by content using STRICT sector-aware matching.
        Requires company name to appear on first page or as consecutive words.
        
        Args:
            pdf_text: Extracted text from PDF
            
        Returns:
            Company dictionary if matched, None otherwise
        """
        # Split text into pages (by looking for page breaks or using first portion)
        # For now, use first 2000 chars as "first page" approximation
        first_page_text = pdf_text[:2000].lower()
        pdf_text_lower = pdf_text.lower()
        
        # BLACKLIST: Companies that appear frequently in regulatory text and cause false matches
        # These should NOT be matched through content matching (only from filename)
        blacklisted_companies = {
            'bse limited', 'bse ltd',
            'nse limited', 'nse ltd',
            'bombay stock exchange limited',
            'national stock exchange of india limited'
        }
        
        # Common words to ignore when matching
        common_words = {
            'limited', 'ltd', 'ltd.', 'india', 'private', 'public', 'corporation',
            'corp', 'corp.', 'inc', 'inc.', 'incorporated', 'company', 'co', 'co.',
            'industries', 'group', 'enterprises', 'solutions', 'services', 'systems'
        }
        
        # Generic sector names that should not match alone
        generic_names = {
            'bank', 'power', 'infra', 'finance', 'capital', 'global', 'infrastructure',
            'financial', 'insurance', 'cement', 'pharma', 'pharmaceuticals', 'energy',
            'realty', 'housing', 'holdings', 'investment', 'clean', 'electrical', 'electrics'
        }
        
        best_match = None
        best_score = 0
        
        for _, row in self.companies_df.iterrows():
            company_name = str(row['company_name']).strip()
            if not company_name:
                continue
            
            company_name_lower = company_name.lower()
            
            # Skip blacklisted companies (they appear too frequently as regulatory references)
            if company_name_lower in blacklisted_companies:
                continue
            
            # Strategy 1: Check if full company name appears on first page (highest confidence)
            if company_name_lower in first_page_text:
                score = 1000 + len(company_name)  # High score for full name on first page
                if score > best_score:
                    best_match = row.to_dict()
                    best_score = score
                continue
            
            # Extract significant words
            company_words = [
                w.rstrip('.,;:').lower() 
                for w in company_name.split() 
                if len(w.rstrip('.,;:')) > 2 and w.rstrip('.,;:').lower() not in common_words
            ]
            
            if not company_words:
                # All common words - require full name anywhere
                if company_name_lower in pdf_text_lower:
                    score = len(company_name)
                    if score > best_score:
                        best_match = row.to_dict()
                        best_score = score
                continue
            
            # Check if generic name
            is_generic = len(company_words) == 1 and company_words[0] in generic_names
            
            if is_generic:
                # For generic names, require full company name anywhere in PDF
                if company_name_lower in pdf_text_lower:
                    score = len(company_name)
                    if score > best_score:
                        best_match = row.to_dict()
                        best_score = score
            else:
                # Strategy 2: Check for consecutive word pairs on first page (medium confidence)
                # This ensures words appear together, not scattered
                if len(company_words) >= 2:
                    # Check for consecutive pairs (e.g., "ABB India" not just "ABB" and "India" separately)
                    found_consecutive_pair = False
                    for i in range(len(company_words) - 1):
                        pair = f"{company_words[i]} {company_words[i+1]}"
                        if pair in first_page_text:
                            found_consecutive_pair = True
                            score = 500 + len(pair)  # Medium-high score for consecutive pair on first page
                            if score > best_score:
                                best_match = row.to_dict()
                                best_score = score
                            break
                    
                    # Strategy 3: If no consecutive pair, require ALL words on first page
                    if not found_consecutive_pair:
                        matched_words_on_first_page = sum(1 for word in company_words if word in first_page_text)
                        if matched_words_on_first_page == len(company_words):
                            score = 400 + len(company_words) * 10
                            if score > best_score:
                                best_match = row.to_dict()
                                best_score = score
                        # Fallback: Require ALL words anywhere in PDF (lowest confidence)
                        elif matched_words_on_first_page >= 2:
                            matched_words_anywhere = sum(1 for word in company_words if word in pdf_text_lower)
                            if matched_words_anywhere == len(company_words):
                                score = 200 + len(company_words) * 5
                                if score > best_score:
                                    best_match = row.to_dict()
                                    best_score = score
                else:
                    # Single significant word - must appear on first page
                    if company_words[0] in first_page_text:
                        score = 300
                        if score > best_score:
                            best_match = row.to_dict()
                            best_score = score
        
        if best_match:
            logger.info(f"Matched by content: {best_match['company_name']} (score: {best_score})")
            logger.debug(f"  Matching strategy: {'Full name on first page' if best_score >= 1000 else 'Consecutive pair' if best_score >= 500 else 'All words' if best_score >= 200 else 'Single word'}")
        else:
            logger.warning("No company match found by content")
        
        return best_match
    
    def extract_year_from_pdf(self, pdf_text: str, company_data: Dict) -> Optional[str]:
        """
        Extract financial year from PDF text by reading content.
        Prioritizes finding year on cover page (first 2000 chars).
        
        Args:
            pdf_text: Extracted text from PDF
            company_data: Matched company data
            
        Returns:
            Financial year string (e.g., "2023-24") or None
        """
        if not pdf_text or len(pdf_text.strip()) < 50:
            return None
        
        pdf_text_lower = pdf_text.lower()
        
        # Focus on first 2000 characters (cover page) for year - most reliable
        cover_page_text = pdf_text[:2000]
        cover_page_lower = cover_page_text.lower()
        
        # Try to find year patterns - prioritize cover page
        # Pattern 1: YYYY-YY (e.g., 2023-24, 2024-25) - most common format
        pattern1 = r'\b(20\d{2})-(\d{2})\b'
        
        # First check cover page
        matches1_cover = re.findall(pattern1, cover_page_text)
        if matches1_cover:
            for start, end in matches1_cover:
                try:
                    start_int = int(start)
                    end_int = int(end)
                    if end_int == (start_int % 100) + 1 or end_int == start_int % 100:
                        year_str = f"{start}-{end}"
                        # Validate against configured years
                        if year_str in BRSR_FINANCIAL_YEARS:
                            logger.info(f"Found year in PDF content (cover page): {year_str}")
                            return year_str
                except ValueError:
                    continue
        
        # Check full text if not found on cover page
        matches1_full = re.findall(pattern1, pdf_text)
        if matches1_full:
            for start, end in matches1_full:
                try:
                    start_int = int(start)
                    end_int = int(end)
                    if end_int == (start_int % 100) + 1 or end_int == start_int % 100:
                        year_str = f"{start}-{end}"
                        if year_str in BRSR_FINANCIAL_YEARS:
                            logger.info(f"Found year in PDF content (full text): {year_str}")
                            return year_str
                except ValueError:
                    continue
        
        # Pattern 2: FY YY or FY-YY (e.g., FY24, FY 24, FY-24)
        pattern2 = r'\bfy[\s-]?(\d{2})\b'
        matches2_cover = re.findall(pattern2, cover_page_lower)
        if matches2_cover:
            for year_short in matches2_cover:
                try:
                    year_int = int(year_short)
                    if year_int < 50:
                        full_year = 2000 + year_int
                    else:
                        full_year = 1900 + year_int
                    prev_year = full_year - 1
                    year_str = f"{prev_year}-{year_short}"
                    if year_str in BRSR_FINANCIAL_YEARS:
                        logger.info(f"Found year in PDF content (FY format, cover page): {year_str}")
                        return year_str
                except ValueError:
                    continue
        
        # Pattern 3: YYYY-YYYY (e.g., 2023-2024)
        pattern3 = r'\b(20\d{2})-(20\d{2})\b'
        matches3_cover = re.findall(pattern3, cover_page_text)
        if matches3_cover:
            for start, end in matches3_cover:
                try:
                    start_int = int(start)
                    end_int = int(end)
                    if end_int == start_int + 1:
                        year_short = str(end_int % 100).zfill(2)
                        year_str = f"{start_int}-{year_short}"
                        if year_str in BRSR_FINANCIAL_YEARS:
                            logger.info(f"Found year in PDF content (full format, cover page): {year_str}")
                            return year_str
                except ValueError:
                    continue
        
        # Pattern 4: Financial Year text followed by year (e.g., "Financial Year 2023-24")
        pattern4 = r'(?:financial\s+year|fy|year)[\s:-]+(?:20)?(\d{2})[\s-]+(?:to|-)?[\s-]*(?:20)?(\d{2})'
        matches4 = re.findall(pattern4, cover_page_lower, re.IGNORECASE)
        if matches4:
            for start, end in matches4:
                try:
                    start_int = int(start)
                    end_int = int(end)
                    # Handle 2-digit years
                    if start_int < 50:
                        start_full = 2000 + start_int
                    else:
                        start_full = 1900 + start_int
                    
                    if end_int == (start_int % 100) + 1 or end_int == start_int % 100:
                        year_str = f"{start_full}-{end:02d}"
                        if year_str in BRSR_FINANCIAL_YEARS:
                            logger.info(f"Found year in PDF content (Financial Year text): {year_str}")
                            return year_str
                except ValueError:
                    continue
        
        # Pattern 5: Just year (e.g., 2023) - check if it matches configured years
        pattern5 = r'\b(20\d{2})\b'
        matches5_cover = re.findall(pattern5, cover_page_text)
        # Get unique matches and check against configured years
        seen_years = set()
        for year_str in matches5_cover:
            if year_str in seen_years:
                continue
            seen_years.add(year_str)
            # Check if this year matches any of our configured years
            for config_year in BRSR_FINANCIAL_YEARS:
                if config_year.startswith(year_str):
                    logger.info(f"Found year in PDF content (single year): {config_year}")
                    return config_year
        
        logger.debug("Could not extract year from PDF content")
        return None
    
    def apply_naming_convention(
        self, 
        company_data: Dict, 
        year: str
    ) -> str:
        """
        Apply naming convention from Excel EXACTLY, only replacing year.
        Preserves exact format from CSV until year part, then replaces year with extracted year from PDF.
        
        Args:
            company_data: Company data dictionary
            year: Financial year string extracted from PDF
            
        Returns:
            Formatted filename (without extension)
        """
        naming_convention = company_data.get('naming_convention')
        
        if naming_convention:
            # Use EXACT naming convention from CSV - preserve format exactly
            filename = str(naming_convention).strip()
            
            logger.info(f"Original naming convention: {filename}")
            logger.info(f"Extracted year from PDF: {year}")
            
            # Replace {year} placeholder with actual year (case-insensitive)
            filename = re.sub(r'\{year\}', year, filename, flags=re.IGNORECASE)
            
            # Extract year components for replacement
            year_parts = year.split('-')
            year_start = year_parts[0] if len(year_parts) > 0 else year
            year_end_short = year_parts[1] if len(year_parts) > 1 else ""
            year_end_full = str(int(year_start) + 1) if year_start.isdigit() else ""
            
            # Replace any existing year pattern in the naming convention with extracted year
            # IMPORTANT: Check the ORIGINAL filename (before replacement) to determine which pattern to use
            # This prevents double replacement when one pattern creates text that matches another pattern
            
            original_filename = filename  # Save original to check patterns
            
            # Pattern 1: YYYY_YYYY format (e.g., 2023_2024) - Check ORIGINAL filename
            if year_end_full and re.search(r'(\d{4})_(\d{4})', original_filename):
                old_pattern = r'(\d{4})_(\d{4})'
                new_value = f"{year_start}_{year_end_full}"
                filename = re.sub(old_pattern, new_value, filename, count=1)
                logger.info(f"✓ Replaced YYYY_YYYY pattern: {new_value}")
            
            # Pattern 2: YYYY-YYYY format (e.g., 2023-2024) - Check ORIGINAL filename
            elif year_end_full and re.search(r'(\d{4})-(\d{4})', original_filename):
                old_pattern = r'(\d{4})-(\d{4})'
                new_value = f"{year_start}-{year_end_full}"
                filename = re.sub(old_pattern, new_value, filename, count=1)
                logger.info(f"✓ Replaced YYYY-YYYY pattern: {new_value}")
            
            # Pattern 3: YYYY_YY format (e.g., 2023_24) - Check ORIGINAL filename
            elif year_end_short and re.search(r'(\d{4})_(\d{2})(?!\d)', original_filename):
                old_pattern = r'(\d{4})_(\d{2})(?!\d)'
                new_value = year.replace('-', '_')
                filename = re.sub(old_pattern, new_value, filename, count=1)
                logger.info(f"✓ Replaced YYYY_YY pattern: {new_value}")
            
            # Pattern 4: YYYY-YY format (e.g., 2023-24) - Check ORIGINAL filename
            elif year_end_short and re.search(r'(\d{4})-(\d{2})(?!\d)', original_filename):
                old_pattern = r'(\d{4})-(\d{2})(?!\d)'
                new_value = year
                filename = re.sub(old_pattern, new_value, filename, count=1)
                logger.info(f"✓ Replaced YYYY-YY pattern: {new_value}")
            
            logger.info(f"Final filename after year replacement: {filename}")
            
            # Replace {symbol} placeholder if it exists (case-insensitive)
            symbol = str(company_data.get('symbol', '')).strip()
            if symbol:
                filename = re.sub(r'\{symbol\}', symbol, filename, flags=re.IGNORECASE)
            
            # Replace {serial} placeholder if it exists (case-insensitive)
            serial = str(company_data.get('serial_number', ''))
            if serial:
                filename = re.sub(r'\{serial(_number)?\}', serial, filename, flags=re.IGNORECASE)
            
            # ONLY remove strictly illegal filesystem characters (preserve everything else)
            illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in illegal_chars:
                filename = filename.replace(char, '_')  # Replace with underscore instead of removing
            
            filename = filename.strip()
            
            # Remove .pdf extension if present (we'll add it later)
            if filename.lower().endswith('.pdf'):
                filename = filename[:-4]
            
            return filename
        else:
            # Fallback: Use default naming
            serial = company_data.get('serial_number', '')
            symbol = company_data.get('symbol', '')
            company_name = company_data.get('company_name', 'Unknown')
            
            # Clean company name for filename
            company_name_clean = re.sub(r'[<>:"/\\|?*]', '_', company_name)
            company_name_clean = company_name_clean.replace(' ', '_')
            
            return f"{serial}_{company_name_clean}_{year}"
    
    def process_pdf(self, pdf_path: Path) -> Tuple[bool, str]:
        """
        Process a single PDF: identify company, extract year, rename and move.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        # Step A: PRIORITY 1 - Try symbol match from filename (FASTEST)
        company_data = self.match_company_by_symbol(pdf_path.name)
        matched_by_symbol_filename = company_data is not None
        
        if matched_by_symbol_filename:
            logger.info(f"✓ Matched by SYMBOL in filename: {company_data['symbol']}")
            # Extract text to verify company name appears in PDF
            pdf_text_for_verification = self.extract_text_from_pdf(pdf_path, max_pages=5)
            if not self.verify_company_match(pdf_text_for_verification, company_data):
                logger.warning(f"⚠ VERIFICATION FAILED: Symbol '{company_data['symbol']}' matched from filename, but company name '{company_data['company_name']}' not found in PDF")
                logger.warning(f"   This is likely a false match. Trying alternative matching...")
                company_data = None  # Reject this match
                matched_by_symbol_filename = False
        
        if not matched_by_symbol_filename:
            # Step B: PRIORITY 2 - Extract text and search for symbol in PDF content (HIGH PRIORITY)
            logger.info("No valid symbol match in filename, extracting text to search for symbol in PDF content...")
            pdf_text = self.extract_text_from_pdf(pdf_path, max_pages=10)
            
            if not pdf_text or len(pdf_text.strip()) < 50:
                logger.warning("Could not extract sufficient text from PDF (might be scanned)")
                logger.warning("Please ensure the PDF contains selectable text, or add company symbol to filename")
                return False, "Could not extract sufficient text from PDF (might be scanned - try adding company symbol to filename)"
            
            # PRIORITY: Search for symbol in PDF content first
            company_data = self.match_company_by_symbol_in_content(pdf_text)
            matched_by_symbol_content = company_data is not None
            
            if matched_by_symbol_content:
                logger.info(f"✓ Matched by SYMBOL in PDF content: {company_data['symbol']}")
                # Verify company name appears in PDF (already extracted text)
                if not self.verify_company_match(pdf_text, company_data):
                    logger.warning(f"⚠ VERIFICATION FAILED: Symbol '{company_data['symbol']}' found in PDF, but company name '{company_data['company_name']}' not found prominently")
                    logger.warning(f"   This is likely a false match (e.g., 'BSE' referring to exchange). Trying alternative matching...")
                    company_data = None  # Reject this match
                    matched_by_symbol_content = False
            
            if not matched_by_symbol_content:
                # Step C: FALLBACK - Try content matching (company name) ONLY if symbol match failed
                logger.info("No valid symbol match found, trying company name matching...")
                company_data = self.match_company_by_content(pdf_text)
                
                if company_data:
                    logger.info(f"✓ Matched by COMPANY NAME in content: {company_data['company_name']}")
                    # Verify (should already be validated by match_company_by_content, but double-check)
                    if not self.verify_company_match(pdf_text, company_data):
                        logger.warning(f"⚠ VERIFICATION FAILED: Company name match failed verification")
                        company_data = None
                else:
                    return False, "Could not identify company (no valid symbol match, no company name match)"
        
        if not company_data:
            return False, "Could not identify company (all matches failed verification)"
        
        logger.info(f"✓ Identified company: {company_data['company_name']} ({company_data['symbol']})")
        
        # Step 3: Extract year from PDF content (primary source) - MUST be accurate
        logger.info("Extracting year from PDF content (reading first 15 pages)...")
        # Reuse pdf_text if already extracted (for efficiency), otherwise extract now
        if 'pdf_text' not in locals() or len(pdf_text) < 100:
            pdf_text = self.extract_text_from_pdf(pdf_path, max_pages=15)  # Extract more pages for better year detection
        else:
            # If we already have text, try to get more pages for year extraction
            pdf_text_extended = self.extract_text_from_pdf(pdf_path, max_pages=15)
            if len(pdf_text_extended) > len(pdf_text):
                pdf_text = pdf_text_extended
        
        if not pdf_text or len(pdf_text.strip()) < 50:
            logger.warning(f"⚠ Could not extract sufficient text from PDF (got {len(pdf_text) if pdf_text else 0} chars)")
            logger.warning(f"   Trying to extract from fewer pages...")
            # Try with fewer pages
            pdf_text = self.extract_text_from_pdf(pdf_path, max_pages=3)
            if not pdf_text or len(pdf_text.strip()) < 50:
                pdf_text = ""  # Continue with empty text
        
        year = self.extract_year_from_pdf(pdf_text, company_data)
        
        # Log what was extracted for debugging
        if year:
            logger.info(f"✓ Year extracted from PDF content: {year}")
        else:
            logger.warning(f"⚠ No year found in PDF content for {pdf_path.name}")
            logger.warning(f"   PDF text length: {len(pdf_text) if pdf_text else 0} chars")
        
        # STRICT: Only trust year from PDF content, NOT from filename
        # If no year found in PDF content, ask user for input
        if not year:
            logger.warning(f"⚠ Could not extract year from PDF content: {pdf_path.name}")
            logger.warning(f"   Company: {company_data['company_name']}")
            logger.warning(f"   Available years: {', '.join(BRSR_FINANCIAL_YEARS)}")
            logger.warning(f"   Note: Year must be extracted from PDF content, not from filename")
            print(f"\n⚠ Could not extract year from PDF content: {pdf_path.name}")
            print(f"   Company: {company_data['company_name']}")
            print(f"   Available years: {', '.join(BRSR_FINANCIAL_YEARS)}")
            print(f"   Note: Year must be extracted from PDF content (not filename)")
            
            user_input = input("   Enter year (or press Enter to skip): ").strip()
            if user_input:
                year = user_input
                logger.info(f"Using user-provided year: {year}")
            else:
                logger.warning(f"Skipping file - year not found in PDF content and user declined to provide")
                return False, "Year not found in PDF content (and user declined to provide)"
        
        logger.info(f"✓ Extracted year: {year}")
        
        # Step 4: Apply naming convention and rename
        new_filename_base = self.apply_naming_convention(company_data, year)
        new_filename = f"{new_filename_base}.pdf"
        
        # Create output path
        output_path = self.output_folder / new_filename
        
        # If file already exists, DO NOT REPLACE IT - skip it
        if output_path.exists():
            logger.warning(f"⚠ SKIPPING: File already exists in final folder: {output_path.name}")
            logger.info(f"   Original file will remain in manual_downloads: {pdf_path.name}")
            logger.info(f"   Company: {company_data['company_name']}, Year: {year}")
            logger.info(f"   Skipping to preserve existing file (not replacing)")
            return False, f"File already exists in final folder: {output_path.name} (skipped to preserve existing file)"
        
        # Copy and rename file (all in same folder, no subfolders)
        try:
            shutil.copy2(pdf_path, output_path)
            logger.info(f"✓ Renamed and copied: {pdf_path.name} -> {output_path.name}")
            
            # Delete original file from manual_downloads after successful copy
            try:
                pdf_path.unlink()
                logger.info(f"✓ Deleted original file: {pdf_path.name}")
            except Exception as e:
                logger.warning(f"⚠ Could not delete original file {pdf_path.name}: {e}")
                # Don't fail the operation if deletion fails
            
            return True, output_path.name  # Return just the filename
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return False, f"Error copying file: {e}"
    
    def process_all(self) -> Tuple[Dict[str, int], pd.DataFrame]:
        """
        Process all PDFs in the input folder and create comprehensive tracking CSV.
        
        Returns:
            Tuple of (stats dictionary, comprehensive tracking DataFrame)
        """
        pdf_files = list(self.input_folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_folder}")
            # Still generate comprehensive CSV from existing files
            return self._generate_comprehensive_csv()
        
        logger.info(f"\nFound {len(pdf_files)} PDF file(s) to process")
        
        stats = {'processed': 0, 'successful': 0, 'failed': 0}
        
        # Track all processed files in a list (for backward compatibility)
        tracking_data = []
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            stats['processed'] += 1
            original_name = pdf_path.name
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file {idx}/{len(pdf_files)}: {original_name}")
            logger.info(f"{'='*60}")
            
            try:
                success, message = self.process_pdf(pdf_path)
                
                # Get the final filename if successful
                if success:
                    stats['successful'] += 1
                    # Message is the new filename
                    final_name = message if message.endswith('.pdf') else message
                    
                    logger.info(f"✓ SUCCESS: {original_name} -> {final_name}")
                    
                    tracking_data.append({
                        'Original Filename': original_name,
                        'Renamed Filename': final_name,
                        'Status': 'Success',
                        'Message': 'Renamed successfully'
                    })
                else:
                    stats['failed'] += 1
                    logger.error(f"✗ FAILED: {original_name} - {message}")
                    tracking_data.append({
                        'Original Filename': original_name,
                        'Renamed Filename': '',
                        'Status': 'Failed',
                        'Message': message
                    })
            except Exception as e:
                # Catch any unexpected errors to ensure all files are processed
                stats['failed'] += 1
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"✗ ERROR processing {original_name}: {error_msg}", exc_info=True)
                tracking_data.append({
                    'Original Filename': original_name,
                    'Renamed Filename': '',
                    'Status': 'Failed',
                    'Message': error_msg
                })
            
            logger.info(f"Completed file {idx}/{len(pdf_files)}")
        
        # Generate comprehensive CSV with all companies and all years
        comprehensive_df = self._generate_comprehensive_csv()
        
        return stats, comprehensive_df
    
    def _generate_comprehensive_csv(self) -> pd.DataFrame:
        """
        Generate comprehensive CSV showing all companies and all 3 years with status.
        
        Returns:
            DataFrame with columns: Company Name, Symbol, Serial Number, 
            Year 2022-23 Status, Year 2022-23 Filename, 
            Year 2023-24 Status, Year 2023-24 Filename,
            Year 2024-25 Status, Year 2024-25 Filename
        """
        from config.config import BRSR_FINANCIAL_YEARS
        
        # Load all companies
        companies_df = self.company_reader.get_companies_dataframe()
        
        # Get all files in final folder
        final_files = list(self.output_folder.glob("*.pdf"))
        final_filenames = {f.name for f in final_files}
        
        # Build comprehensive tracking data
        comprehensive_data = []
        
        for _, company_row in companies_df.iterrows():
            company_name = company_row['company_name']
            symbol = company_row['symbol']
            serial_number = company_row['serial_number']
            naming_convention = company_row.get('naming_convention', '')
            
            # Initialize row data
            row_data = {
                'Company Name': company_name,
                'Symbol': symbol,
                'Serial Number': serial_number,
            }
            
            # Check each year
            for year in BRSR_FINANCIAL_YEARS:
                # Generate expected filename for this company and year
                if naming_convention:
                    expected_filename = str(naming_convention).strip()
                    
                    # Replace year pattern
                    year_parts = year.split('-')
                    year_start = year_parts[0]
                    year_end_full = str(int(year_start) + 1) if year_start.isdigit() else ""
                    year_end_short = year_parts[1] if len(year_parts) > 1 else ""
                    
                    # Replace {year} placeholder
                    expected_filename = re.sub(r'\{year\}', year, expected_filename, flags=re.IGNORECASE)
                    
                    # Replace year patterns
                    if year_end_full and re.search(r'(\d{4})_(\d{4})', expected_filename):
                        expected_filename = re.sub(r'(\d{4})_(\d{4})', f"{year_start}_{year_end_full}", expected_filename, count=1)
                    elif year_end_full and re.search(r'(\d{4})-(\d{4})', expected_filename):
                        expected_filename = re.sub(r'(\d{4})-(\d{4})', f"{year_start}-{year_end_full}", expected_filename, count=1)
                    elif year_end_short and re.search(r'(\d{4})_(\d{2})(?!\d)', expected_filename):
                        expected_filename = re.sub(r'(\d{4})_(\d{2})(?!\d)', year.replace('-', '_'), expected_filename, count=1)
                    elif year_end_short and re.search(r'(\d{4})-(\d{2})(?!\d)', expected_filename):
                        expected_filename = re.sub(r'(\d{4})-(\d{2})(?!\d)', year, expected_filename, count=1)
                    
                    # Replace placeholders
                    if symbol:
                        expected_filename = re.sub(r'\{symbol\}', symbol, expected_filename, flags=re.IGNORECASE)
                    if serial_number:
                        expected_filename = re.sub(r'\{serial(_number)?\}', str(serial_number), expected_filename, flags=re.IGNORECASE)
                    
                    # Clean illegal characters
                    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
                    for char in illegal_chars:
                        expected_filename = expected_filename.replace(char, '_')
                    expected_filename = expected_filename.strip()
                    
                    if not expected_filename.lower().endswith('.pdf'):
                        expected_filename = f"{expected_filename}.pdf"
                else:
                    # Fallback naming
                    expected_filename = f"{serial_number}_{company_name.replace(' ', '_')}_{year}.pdf"
                
                # Check if file exists
                if expected_filename in final_filenames:
                    row_data[f'{year} Status'] = 'Renamed'
                    row_data[f'{year} Filename'] = expected_filename
                else:
                    row_data[f'{year} Status'] = 'Not Found'
                    row_data[f'{year} Filename'] = ''
            
            comprehensive_data.append(row_data)
        
        return pd.DataFrame(comprehensive_data)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rename manually downloaded BRSR PDFs according to Excel naming convention"
    )
    parser.add_argument(
        '--excel',
        type=str,
        default=None,
        help='Path to NIFTY 500 firms.xlsx (default: auto-detect)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='manual_downloads',
        help='Input folder with manually downloaded PDFs (default: manual_downloads/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='brsr_reports/final',
        help='Output folder for renamed PDFs (default: brsr_reports/final/)'
    )
    
    args = parser.parse_args()
    
    # Find Excel file
    if args.excel:
        excel_path = Path(args.excel)
    else:
        # Auto-detect: check parent directory
        base_dir = Path(__file__).parent.parent
        excel_path = base_dir / "NIFTY 500 firms.xlsx"
        
        if not excel_path.exists():
            # Try current directory
            excel_path = Path("NIFTY 500 firms.xlsx")
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.error("Please specify --excel path or place 'NIFTY 500 firms.xlsx' in parent directory")
        sys.exit(1)
    
    # Setup paths
    script_dir = Path(__file__).parent
    input_folder = script_dir / args.input
    output_folder = script_dir / args.output
    
    # Create renamer and process
    renamer = ManualPDFRenamer(excel_path, input_folder, output_folder)
    
    logger.info(f"\n{'='*60}")
    logger.info("MANUAL PDF RENAMER")
    logger.info(f"{'='*60}")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Excel file: {excel_path}")
    logger.info(f"{'='*60}\n")
    
    stats, tracking_df = renamer.process_all()
    
    # Save tracking CSV
    if not tracking_df.empty:
        csv_path = script_dir / "manual_rename_tracking.csv"
        tracking_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"\n✓ Tracking CSV saved to: {csv_path}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"{'='*60}\n")
    
    if not tracking_df.empty:
        print(f"\nComprehensive Tracking CSV Preview (first 10 companies):")
        print(tracking_df.head(10).to_string(index=False))
        print(f"\nFull CSV saved to: {csv_path}")
        print(f"   (Shows all companies with status for all 3 years: 2022-23, 2023-24, 2024-25)")
        print(f"   Status: 'Renamed' = file exists, 'Not Found' = file not found")


if __name__ == "__main__":
    main()
