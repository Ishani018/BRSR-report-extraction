"""
Section boundary detector - analyzes PDF layout to identify section boundaries.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pdfplumber

from section_metadata import (
    TextBlock, SectionBoundary, SectionType, 
    SECTION_KEYWORDS, SECTION_END_KEYWORDS
)

logger = logging.getLogger(__name__)


class SectionBoundaryDetector:
    """
    Detects section boundaries from PDF layout metadata.
    Uses font size, position, and keyword matching.
    """
    
    def __init__(self, pdf_path: Path):
        """
        Initialize detector with PDF path.
        
        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = pdf_path
        self.text_blocks: List[TextBlock] = []
        
    def extract_layout_metadata(self) -> List[TextBlock]:
        """
        Extract text blocks with layout metadata from PDF.
        
        Returns:
            List of TextBlock objects with font size and position info
        """
        logger.info(f"Extracting layout metadata from {self.pdf_path.name}")
        blocks = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text with layout information
                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False
                    )
                    
                    if not words:
                        continue
                    
                    # Calculate median font size for this page
                    font_sizes = [w.get('height', 10) for w in words]
                    page_median_font = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10
                    
                    # Group words into lines based on y-position
                    lines = self._group_words_into_lines(words)
                    
                    for line in lines:
                        text = ' '.join([w['text'] for w in line])
                        
                        if not text.strip():
                            continue
                        
                        # Use first word's metrics as line metrics
                        first_word = line[0]
                        font_size = first_word.get('height', page_median_font)
                        y_pos = first_word['top']
                        x_pos = first_word['x0']
                        
                        # Bounding box for entire line
                        x0 = min(w['x0'] for w in line)
                        y0 = min(w['top'] for w in line)
                        x1 = max(w['x1'] for w in line)
                        y1 = max(w['bottom'] for w in line)
                        
                        block = TextBlock(
                            text=text,
                            page_number=page_num,
                            font_size=font_size,
                            y_position=y_pos,
                            x_position=x_pos,
                            bbox=(x0, y0, x1, y1)
                        )
                        blocks.append(block)
                        
        except Exception as e:
            logger.error(f"Error extracting layout metadata: {e}", exc_info=True)
        
        logger.info(f"Extracted {len(blocks)} text blocks from {page_num} pages")
        self.text_blocks = blocks
        return blocks
    
    def _group_words_into_lines(self, words: List[dict]) -> List[List[dict]]:
        """
        Group words into lines based on y-position proximity.
        
        Args:
            words: List of word dictionaries from pdfplumber
            
        Returns:
            List of lines, where each line is a list of words
        """
        if not words:
            return []
        
        # Sort by y-position, then x-position
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        lines = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0]['top']
        
        for word in sorted_words[1:]:
            # If word is on same line (within 3 pixels), add to current line
            if abs(word['top'] - current_y) <= 3:
                current_line.append(word)
            else:
                # Start new line
                lines.append(current_line)
                current_line = [word]
                current_y = word['top']
        
        # Don't forget last line
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def detect_section_boundaries(self) -> Dict[str, Optional[SectionBoundary]]:
        """
        Detect MD&A and Letter to Stakeholders boundaries.
        
        Returns:
            Dictionary mapping section type to SectionBoundary (or None if not found)
        """
        logger.info("Detecting section boundaries...")
        
        if not self.text_blocks:
            self.extract_layout_metadata()
        
        boundaries = {
            "mdna": None,
            "letter_to_stakeholders": None
        }
        
        # Detect each section type
        for section_type in [SectionType.MDNA, SectionType.LETTER_TO_STAKEHOLDERS]:
            boundary = self._find_section_boundary(section_type)
            if boundary:
                boundaries[section_type.value] = boundary
                logger.info(
                    f"Found {section_type.value}: pages {boundary.start_page}-{boundary.end_page}, "
                    f"confidence={boundary.confidence:.2f}"
                )
        
        return boundaries
    
    def _find_section_boundary(self, section_type: SectionType) -> Optional[SectionBoundary]:
        """
        Find boundary for a specific section type.
        
        Args:
            section_type: Type of section to find
            
        Returns:
            SectionBoundary if found, None otherwise
        """
        keywords = SECTION_KEYWORDS.get(section_type, [])
        
        # Find candidate headings
        candidates = []
        
        for block in self.text_blocks:
            # Check if this looks like a heading
            if not self._is_potential_heading(block):
                continue
            
            # Check if text matches section keywords
            normalized = block.normalized_text
            
            for keyword in keywords:
                if keyword in normalized:
                    confidence = self._calculate_confidence(block, keyword, normalized)
                    candidates.append((block, confidence, keyword))
                    logger.debug(
                        f"Candidate heading on page {block.page_number}: "
                        f"'{block.text}' (confidence={confidence:.2f})"
                    )
                    break
        
        if not candidates:
            logger.info(f"No candidates found for {section_type.value}")
            return None
        
        # Select best candidate (highest confidence)
        best_block, best_confidence, matched_keyword = max(candidates, key=lambda x: x[1])
        
        # Find section end
        end_page = self._find_section_end(best_block.page_number, section_type)
        
        return SectionBoundary(
            section_type=section_type,
            start_page=best_block.page_number,
            end_page=end_page,
            confidence=best_confidence,
            start_heading=best_block.text,
            detection_method="layout_and_keywords"
        )
    
    def _is_potential_heading(self, block: TextBlock) -> bool:
        """
        Check if text block could be a section heading.
        
        Criteria:
        - Line length < 120 characters
        - In top 30% of page (approximate)
        - Font size prominence
        
        Args:
            block: TextBlock to evaluate
            
        Returns:
            True if block could be a heading
        """
        # Line length check
        if block.line_length > 120:
            return False
        
        # Y-position check (top 30% of typical page ~800pts = top 240pts)
        if block.y_position > 300:
            return False
        
        # Get page median font size
        page_blocks = [b for b in self.text_blocks if b.page_number == block.page_number]
        if page_blocks:
            page_fonts = [b.font_size for b in page_blocks]
            median_font = sorted(page_fonts)[len(page_fonts) // 2]
            
            # Font should be larger than median
            if block.font_size < median_font * 1.1:
                return False
        
        return True
    
    def _calculate_confidence(self, block: TextBlock, keyword: str, normalized_text: str) -> float:
        """
        Calculate confidence score for a heading match.
        
        Factors:
        - Keyword match quality (exact vs partial)
        - Font size prominence
        - Y-position (prefer higher on page)
        - Line length (prefer shorter)
        
        Args:
            block: Detected text block
            keyword: Matched keyword
            normalized_text: Normalized block text
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Keyword match quality
        if normalized_text.strip() == keyword:
            confidence += 0.3  # Exact match
        elif normalized_text.startswith(keyword) or normalized_text.endswith(keyword):
            confidence += 0.2  # Strong match
        else:
            confidence += 0.1  # Partial match
        
        # Font size prominence (get page median)
        page_blocks = [b for b in self.text_blocks if b.page_number == block.page_number]
        if page_blocks:
            page_fonts = [b.font_size for b in page_blocks]
            median_font = sorted(page_fonts)[len(page_fonts) // 2]
            font_ratio = block.font_size / median_font if median_font > 0 else 1
            
            if font_ratio > 1.5:
                confidence += 0.15
            elif font_ratio > 1.2:
                confidence += 0.1
        
        # Y-position bonus (higher = better)
        if block.y_position < 150:
            confidence += 0.1
        elif block.y_position < 250:
            confidence += 0.05
        
        # Line length penalty
        if block.line_length < 50:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _find_section_end(self, start_page: int, section_type: SectionType) -> Optional[int]:
        """
        Find where a section ends.
        
        End detection:
        - Another major heading appears
        - Financial statement keywords appear
        - End of document
        
        Args:
            start_page: Page where section starts
            section_type: Type of section
            
        Returns:
            End page number or None if continues to document end
        """
        # Look for end indicators after start page
        subsequent_blocks = [b for b in self.text_blocks if b.page_number > start_page]
        
        for block in subsequent_blocks:
            normalized = block.normalized_text
            
            # Check for financial statement keywords
            for end_keyword in SECTION_END_KEYWORDS:
                if end_keyword in normalized and self._is_potential_heading(block):
                    logger.debug(
                        f"Section end detected at page {block.page_number}: "
                        f"'{block.text}' (financial statement keyword)"
                    )
                    return block.page_number - 1
            
            # Check for other major section headings
            for other_type in [SectionType.MDNA, SectionType.LETTER_TO_STAKEHOLDERS]:
                if other_type == section_type:
                    continue
                
                other_keywords = SECTION_KEYWORDS.get(other_type, [])
                for keyword in other_keywords:
                    if keyword in normalized and self._is_potential_heading(block):
                        logger.debug(
                            f"Section end detected at page {block.page_number}: "
                            f"'{block.text}' (other section start)"
                        )
                        return block.page_number - 1
        
        # If no end found, section continues to end of document
        if subsequent_blocks:
            last_page = max(b.page_number for b in subsequent_blocks)
            return last_page
        
        return start_page  # Single page section
