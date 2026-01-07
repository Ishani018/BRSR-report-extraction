"""
Utility functions for the PDF processing pipeline.
"""
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Remove or replace characters that are invalid in filenames.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    return filename


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def estimate_processing_time(file_size_mb: float, pdf_type: str) -> str:
    """
    Estimate processing time based on file size and type.
    
    Args:
        file_size_mb: File size in megabytes
        pdf_type: Type of PDF ('text' or 'scanned')
        
    Returns:
        Estimated time as string
    """
    # Rough estimates (vary based on hardware)
    if pdf_type == "scanned":
        minutes = file_size_mb * 0.5  # OCR is slower
    else:
        minutes = file_size_mb * 0.1  # Text extraction is fast
    
    if minutes < 1:
        return "< 1 minute"
    elif minutes < 60:
        return f"~{int(minutes)} minutes"
    else:
        hours = minutes / 60
        return f"~{hours:.1f} hours"


def validate_pdf(pdf_path: Path) -> bool:
    """
    Validate that a file is a readable PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if valid PDF, False otherwise
    """
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return False
    
    if not pdf_path.is_file():
        logger.error(f"Not a file: {pdf_path}")
        return False
    
    if pdf_path.suffix.lower() != '.pdf':
        logger.error(f"Not a PDF file: {pdf_path}")
        return False
    
    if pdf_path.stat().st_size == 0:
        logger.error(f"Empty file: {pdf_path}")
        return False
    
    return True


def create_progress_callback(total_items: int, description: str = "Processing"):
    """
    Create a progress callback for tracking processing.
    
    Args:
        total_items: Total number of items to process
        description: Description for the progress bar
        
    Returns:
        Callback function
    """
    from tqdm import tqdm
    
    pbar = tqdm(total=total_items, desc=description)
    
    def callback():
        pbar.update(1)
    
    return callback, pbar


def get_year_from_text(text: str) -> Optional[str]:
    """
    Extract year from text (useful for parsing filenames/metadata).
    
    Args:
        text: Text to search for year
        
    Returns:
        Year as string or None
    """
    # Look for 4-digit year between 1900-2099
    matches = re.findall(r'\b(19|20)\d{2}\b', text)
    if matches:
        return matches[0]
    return None


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
