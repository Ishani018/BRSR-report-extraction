"""
Module for segmenting PDF text into logical sections.
"""
import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from extract_text import PageText
from config import SECTION_KEYWORDS

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a document section with title and content."""
    title: str
    content: str
    start_page: int
    end_page: int
    level: int  # Heading level (1 = main section, 2 = subsection, etc.)


def is_likely_heading(line: str) -> Tuple[bool, int]:
    """
    Determine if a line is likely a section heading.
    
    Args:
        line: Text line to check
        
    Returns:
        Tuple of (is_heading, level) where level indicates heading hierarchy
    """
    line = line.strip()
    
    # Empty lines are not headings
    if not line or len(line) < 3:
        return False, 0
    
    # Skip lines that are too long (likely paragraphs)
    if len(line) > 150:
        return False, 0
    
    # Check for all caps (common in headings) - but not single words
    if line.isupper() and len(line) < 100 and len(line.split()) >= 2:
        return True, 1
    
    # Check for numbered sections (e.g., "1. Introduction", "1.1 Overview")
    if re.match(r'^\d+\.(\d+\.)*\s+[A-Z]', line):
        # Count dots to determine level
        level = line.split()[0].count('.') + 1
        return True, level
    
    # Check for Roman numerals (I., II., III., etc.)
    if re.match(r'^[IVX]+\.\s+[A-Z]', line):
        return True, 1
    
    # Check if line matches known section keywords
    line_lower = line.lower()
    for keyword in SECTION_KEYWORDS:
        if keyword in line_lower:
            return True, 1
    
    # Check for title case without ending punctuation
    words = line.split()
    if (len(words) <= 10 and 
        len(words) > 1 and
        line[0].isupper() and 
        line[-1] not in '.!?,' and
        sum(1 for w in words if w[0].isupper()) >= len(words) * 0.7):
        return True, 2
    
    return False, 0


def extract_toc(pages: List[PageText]) -> Optional[Dict[str, int]]:
    """
    Attempt to extract table of contents if present.
    
    Args:
        pages: List of PageText objects
        
    Returns:
        Dictionary mapping section titles to page numbers, or None
    """
    logger.info("Attempting to extract table of contents...")
    
    toc = {}
    in_toc = False
    toc_pattern = re.compile(r'(.+?)\s+\.{2,}\s*(\d+)')
    
    # Look for TOC in first 20 pages
    for page in pages[:20]:
        lines = page.text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Detect TOC start
            if 'table of contents' in line.lower() or 'contents' == line.lower():
                in_toc = True
                continue
            
            # If in TOC, try to extract entries
            if in_toc:
                match = toc_pattern.match(line)
                if match:
                    section_title = match.group(1).strip()
                    page_num = int(match.group(2))
                    toc[section_title] = page_num
                elif line and not line[0].isdigit() and len(line) > 30:
                    # Likely end of TOC
                    break
    
    if toc:
        logger.info(f"Extracted TOC with {len(toc)} entries")
        return toc
    else:
        logger.info("No TOC found")
        return None


def segment_by_headings(pages: List[PageText]) -> List[Section]:
    """
    Segment document into sections based on detected headings.
    
    Args:
        pages: List of PageText objects
        
    Returns:
        List of Section objects
    """
    logger.info("Segmenting document by headings...")
    
    sections = []
    current_section = None
    current_content = []
    
    for page in pages:
        lines = page.text.split('\n')
        
        for line in lines:
            is_heading, level = is_likely_heading(line)
            
            if is_heading:
                # Save previous section if exists
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    current_section.end_page = page.page_number
                    sections.append(current_section)
                
                # Start new section
                current_section = Section(
                    title=line.strip(),
                    content="",
                    start_page=page.page_number,
                    end_page=page.page_number,
                    level=level
                )
                current_content = []
            else:
                # Add to current section content
                if line.strip():
                    current_content.append(line)
    
    # Don't forget the last section
    if current_section:
        current_section.content = '\n'.join(current_content).strip()
        current_section.end_page = pages[-1].page_number if pages else 0
        sections.append(current_section)
    
    logger.info(f"Segmented document into {len(sections)} sections")
    return sections


def segment_by_keywords(pages: List[PageText]) -> List[Section]:
    """
    Segment document using predefined section keywords.
    
    Args:
        pages: List of PageText objects
        
    Returns:
        List of Section objects
    """
    logger.info("Segmenting document by keywords...")
    
    sections = []
    current_section = None
    current_content = []
    
    for page in pages:
        lines = page.text.split('\n')
        text_lower = page.text.lower()
        
        # Check if any keyword appears on this page
        found_keyword = None
        for keyword in SECTION_KEYWORDS:
            if keyword in text_lower:
                found_keyword = keyword
                break
        
        if found_keyword:
            # Find the line with the keyword
            for i, line in enumerate(lines):
                if found_keyword in line.lower():
                    # Save previous section
                    if current_section:
                        current_section.content = '\n'.join(current_content).strip()
                        current_section.end_page = page.page_number
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = Section(
                        title=line.strip(),
                        content="",
                        start_page=page.page_number,
                        end_page=page.page_number,
                        level=1
                    )
                    current_content = lines[i+1:]  # Content after keyword
                    break
        else:
            # Add entire page to current section
            if current_section:
                current_content.extend(lines)
    
    # Save last section
    if current_section:
        current_section.content = '\n'.join(current_content).strip()
        current_section.end_page = pages[-1].page_number if pages else 0
        sections.append(current_section)
    
    logger.info(f"Segmented document into {len(sections)} sections by keywords")
    return sections


def segment_document(pages: List[PageText]) -> List[Section]:
    """
    Main function to segment document into logical sections.
    Tries multiple strategies and uses the best result.
    
    Args:
        pages: List of PageText objects
        
    Returns:
        List of Section objects
    """
    if not pages:
        return []
    
    # Try to extract TOC first
    toc = extract_toc(pages)
    
    # Try heading-based segmentation
    sections_by_headings = segment_by_headings(pages)
    
    # Try keyword-based segmentation
    sections_by_keywords = segment_by_keywords(pages)
    
    # Choose the method that produced more sections (likely more accurate)
    if len(sections_by_headings) >= len(sections_by_keywords):
        logger.info(f"Using heading-based segmentation ({len(sections_by_headings)} sections)")
        return sections_by_headings
    else:
        logger.info(f"Using keyword-based segmentation ({len(sections_by_keywords)} sections)")
        return sections_by_keywords


def get_section_by_title(sections: List[Section], title: str) -> Optional[Section]:
    """
    Find a section by its title (case-insensitive partial match).
    
    Args:
        sections: List of Section objects
        title: Title to search for
        
    Returns:
        Matching Section or None
    """
    title_lower = title.lower()
    for section in sections:
        if title_lower in section.title.lower():
            return section
    return None


def get_sections_by_level(sections: List[Section], level: int) -> List[Section]:
    """
    Get all sections of a specific heading level.
    
    Args:
        sections: List of Section objects
        level: Heading level to filter by
        
    Returns:
        Filtered list of sections
    """
    return [s for s in sections if s.level == level]


def create_section_hierarchy(sections: List[Section]) -> Dict:
    """
    Create a hierarchical structure of sections based on levels.
    
    Args:
        sections: List of Section objects
        
    Returns:
        Nested dictionary representing section hierarchy
    """
    hierarchy = {}
    current_parent = None
    
    for section in sections:
        if section.level == 1:
            hierarchy[section.title] = {
                'section': section,
                'subsections': []
            }
            current_parent = section.title
        elif current_parent and section.level > 1:
            hierarchy[current_parent]['subsections'].append(section)
    
    return hierarchy
