"""
File Naming Utilities - Standardized naming conventions for BRSR files.
"""
import re
from pathlib import Path
from typing import Optional

from config.config import (
    BRSR_STANDALONE_PREFIX,
    BRSR_EMBEDDED_PREFIX,
    BRSR_FROM_ANNUAL_SUFFIX
)


def clean_company_name(name: str) -> str:
    """
    Clean company name for use in filenames.
    
    Args:
        name: Company name
        
    Returns:
        Cleaned company name safe for filenames
    """
    if not name:
        return "Unknown"
    
    # Remove special characters not allowed in filenames
    cleaned = name.replace('/', '_').replace('\\', '_').replace(':', '_')
    cleaned = cleaned.replace('*', '').replace('?', '').replace('"', '')
    cleaned = cleaned.replace('<', '').replace('>', '').replace('|', '')
    
    # Remove leading/trailing spaces and dots
    cleaned = cleaned.strip().strip('.')
    
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Replace spaces with underscores for filename
    cleaned = cleaned.replace(' ', '_')
    
    return cleaned


def format_brsr_filename(
    company_name: str,
    year: str,
    is_standalone: bool = True,
    file_extension: str = "pdf"
) -> str:
    """
    Format filename for BRSR report.
    
    Args:
        company_name: Company name
        year: Financial year (e.g., '2022-23')
        is_standalone: True if standalone BRSR, False if annual report
        file_extension: File extension (without dot)
        
    Returns:
        Formatted filename
    """
    cleaned_name = clean_company_name(company_name)
    
    if is_standalone:
        prefix = BRSR_STANDALONE_PREFIX
        filename = f"{cleaned_name}_{prefix}_{year}.{file_extension}"
    else:
        prefix = BRSR_EMBEDDED_PREFIX
        filename = f"{cleaned_name}_{prefix}_{year}.{file_extension}"
    
    return filename


def format_brsr_output_filename(
    company_name: str,
    year: str,
    is_standalone: bool = True,
    is_from_annual: bool = False,
    file_type: str = "docx",
    naming_convention: Optional[str] = None,
    symbol: Optional[str] = None,
    serial_number: Optional[int] = None
) -> str:
    """
    Format filename for processed BRSR output.
    Uses naming convention from Excel if provided, otherwise uses default format.
    
    Args:
        company_name: Company name
        year: Financial year (e.g., '2022-23')
        is_standalone: True if standalone BRSR, False if embedded
        is_from_annual: True if extracted from annual report
        file_type: Output file type ('docx', 'json', 'metadata')
        naming_convention: Optional naming convention from Excel (preserves exact format)
        symbol: Optional company symbol for placeholder replacement
        serial_number: Optional serial number for placeholder replacement
        
    Returns:
        Formatted filename
    """
    # If naming convention is provided, use it (like downloads do)
    if naming_convention and naming_convention.strip():
        # Use exact naming convention from Excel/CSV - preserve as-is
        filename = naming_convention.strip()
        
        # Replace common placeholders if they exist (case-insensitive)
        # Replace {year}, {YEAR}, {Year} with actual year
        filename = re.sub(r'\{year\}', year, filename, flags=re.IGNORECASE)
        # Replace {symbol}, {SYMBOL}, {Symbol} with actual symbol
        if symbol:
            filename = re.sub(r'\{symbol\}', symbol.upper(), filename, flags=re.IGNORECASE)
        # Replace {serial}, {SERIAL}, {serial_number}, {SERIAL_NUMBER} with serial number if available
        if serial_number is not None:
            filename = re.sub(r'\{serial(_number)?\}', str(serial_number), filename, flags=re.IGNORECASE)
        
        # ONLY remove strictly illegal filesystem characters to prevent crashes
        illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in illegal_chars:
            filename = filename.replace(char, '')
        
        # Strip ONLY leading/trailing whitespace (preserve internal spaces)
        filename = filename.strip()
        
        # Remove .pdf extension if present (we'll add the correct extension)
        if filename.lower().endswith('.pdf'):
            filename = filename[:-4]
        
        # Add appropriate extension based on file_type
        if file_type == "metadata":
            filename = f"{filename}_metadata.json"
        else:
            filename = f"{filename}.{file_type}"
        
        return filename
    
    # Fall back to default format if no naming convention
    cleaned_name = clean_company_name(company_name)
    
    if is_from_annual:
        # Format: CompanyName_BRSR_Year_from_AnnualReport.{ext}
        if file_type == "metadata":
            filename = f"{cleaned_name}_BRSR_{year}_{BRSR_FROM_ANNUAL_SUFFIX}_metadata.json"
        else:
            filename = f"{cleaned_name}_BRSR_{year}_{BRSR_FROM_ANNUAL_SUFFIX}.{file_type}"
    else:
        # Format: CompanyName_BRSR_Year.{ext}
        if file_type == "metadata":
            filename = f"{cleaned_name}_BRSR_{year}_metadata.json"
        else:
            filename = f"{cleaned_name}_BRSR_{year}.{file_type}"
    
    return filename


def parse_filename(filename: str) -> Optional[dict]:
    """
    Parse BRSR filename to extract information.
    
    Args:
        filename: BRSR filename
        
    Returns:
        Dictionary with parsed information or None if not a BRSR filename
    """
    # Remove extension
    name = Path(filename).stem
    
    # Pattern 1: CompanyName_BRSR_Year or CompanyName_BRSR_Year_from_AnnualReport
    pattern1 = r'^(.+?)_BRSR_(\d{4}-\d{2})(?:_from_AnnualReport)?(?:_metadata)?$'
    match1 = re.match(pattern1, name)
    
    if match1:
        company_name = match1.group(1).replace('_', ' ')
        year = match1.group(2)
        is_from_annual = '_from_AnnualReport' in name
        is_metadata = '_metadata' in name
        
        return {
            'company_name': company_name,
            'year': year,
            'is_standalone': not is_from_annual,
            'is_from_annual': is_from_annual,
            'is_metadata': is_metadata,
            'file_type': 'metadata' if is_metadata else 'unknown'
        }
    
    # Pattern 2: CompanyName_AnnualReport_Year
    pattern2 = r'^(.+?)_AnnualReport_(\d{4}-\d{2})$'
    match2 = re.match(pattern2, name)
    
    if match2:
        company_name = match2.group(1).replace('_', ' ')
        year = match2.group(2)
        
        return {
            'company_name': company_name,
            'year': year,
            'is_standalone': False,
            'is_from_annual': False,
            'is_metadata': False,
            'file_type': 'annual_report'
        }
    
    return None


def create_output_path(
    base_dir: Path,
    year: str,
    company_name: str,
    filename: str
) -> Path:
    """
    Create full output path for BRSR file.
    
    Args:
        base_dir: Base directory for outputs
        year: Financial year
        company_name: Company name
        filename: Filename
        
    Returns:
        Full path to output file
    """
    cleaned_name = clean_company_name(company_name)
    output_dir = base_dir / year / cleaned_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / filename


if __name__ == "__main__":
    # Test file naming functions
    print("Testing file naming utilities...")
    
    # Test clean_company_name
    test_names = [
        "Reliance Industries Ltd.",
        "Tata Steel/TCS",
        "Company   Name",
        "Test <Company> Name?"
    ]
    
    print("\n1. Testing clean_company_name:")
    for name in test_names:
        cleaned = clean_company_name(name)
        print(f"  '{name}' -> '{cleaned}'")
    
    # Test format_brsr_filename
    print("\n2. Testing format_brsr_filename:")
    print(f"  Standalone: {format_brsr_filename('Reliance Industries', '2022-23', True, 'pdf')}")
    print(f"  Annual Report: {format_brsr_filename('Tata Steel', '2023-24', False, 'pdf')}")
    
    # Test format_brsr_output_filename
    print("\n3. Testing format_brsr_output_filename:")
    print(f"  Standalone DOCX: {format_brsr_output_filename('Reliance Industries', '2022-23', True, False, 'docx')}")
    print(f"  Embedded DOCX: {format_brsr_output_filename('Tata Steel', '2023-24', False, True, 'docx')}")
    print(f"  Metadata: {format_brsr_output_filename('Reliance Industries', '2022-23', True, False, 'metadata')}")
    
    # Test parse_filename
    print("\n4. Testing parse_filename:")
    test_filenames = [
        "Reliance_Industries_BRSR_2022-23.pdf",
        "Tata_Steel_BRSR_2023-24_from_AnnualReport.docx",
        "TCS_AnnualReport_2024-25.pdf",
        "Reliance_Industries_BRSR_2022-23_metadata.json"
    ]
    
    for filename in test_filenames:
        parsed = parse_filename(filename)
        if parsed:
            print(f"  '{filename}' -> {parsed}")
        else:
            print(f"  '{filename}' -> Could not parse")
    
    print("\n✓ All tests completed")

