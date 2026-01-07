"""
Module for extracting tables from PDFs.
"""
import logging
import re
from pathlib import Path
from typing import List, Optional
import warnings

import pandas as pd
import camelot
import tabula

from config import MAX_TABLE_EXPORT_SIZE

logger = logging.getLogger(__name__)

# Suppress camelot warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ExtractedTable:
    """Container for extracted table data with metadata."""
    
    def __init__(self, page_number: int, table_index: int, dataframe: pd.DataFrame, method: str):
        self.page_number = page_number
        self.table_index = table_index
        self.dataframe = dataframe
        self.method = method  # 'camelot' or 'tabula'
        self.rows = len(dataframe)
        self.cols = len(dataframe.columns)


def extract_tables_with_camelot(pdf_path: Path, pages: str = "all") -> List[ExtractedTable]:
    """
    Extract tables using Camelot (vector-based extraction).
    
    Args:
        pdf_path: Path to the PDF file
        pages: Pages to extract tables from (default: "all")
        
    Returns:
        List of ExtractedTable objects
    """
    logger.info(f"Extracting tables with Camelot from: {pdf_path.name}")
    tables = []
    
    try:
        # Try lattice method first (for tables with lines)
        table_list = camelot.read_pdf(
            str(pdf_path),
            pages=pages,
            flavor='lattice',
            suppress_stdout=True
        )
        
        logger.info(f"Camelot (lattice) found {len(table_list)} tables")
        
        for idx, table in enumerate(table_list):
            df = table.df
            
            # Skip empty or very small tables
            if df.shape[0] > 1 and df.shape[1] > 1:
                tables.append(ExtractedTable(
                    page_number=table.page,
                    table_index=idx,
                    dataframe=df,
                    method='camelot-lattice'
                ))
        
        # If no tables found, try stream method
        if len(tables) == 0:
            logger.info("Trying Camelot stream method...")
            table_list = camelot.read_pdf(
                str(pdf_path),
                pages=pages,
                flavor='stream',
                suppress_stdout=True
            )
            
            logger.info(f"Camelot (stream) found {len(table_list)} tables")
            
            for idx, table in enumerate(table_list):
                df = table.df
                
                if df.shape[0] > 1 and df.shape[1] > 1:
                    tables.append(ExtractedTable(
                        page_number=table.page,
                        table_index=idx,
                        dataframe=df,
                        method='camelot-stream'
                    ))
                    
    except Exception as e:
        logger.error(f"Error extracting tables with Camelot: {e}")
    
    return tables


def extract_tables_with_tabula(pdf_path: Path, pages: str = "all") -> List[ExtractedTable]:
    """
    Extract tables using Tabula (fallback method).
    
    Args:
        pdf_path: Path to the PDF file
        pages: Pages to extract tables from (default: "all")
        
    Returns:
        List of ExtractedTable objects
    """
    logger.info(f"Extracting tables with Tabula from: {pdf_path.name}")
    tables = []
    
    try:
        # Extract all tables
        dfs = tabula.read_pdf(
            str(pdf_path),
            pages=pages,
            multiple_tables=True,
            silent=True
        )
        
        logger.info(f"Tabula found {len(dfs)} tables")
        
        for idx, df in enumerate(dfs):
            if isinstance(df, pd.DataFrame) and df.shape[0] > 1 and df.shape[1] > 1:
                tables.append(ExtractedTable(
                    page_number=0,  # Tabula doesn't provide page numbers easily
                    table_index=idx,
                    dataframe=df,
                    method='tabula'
                ))
                
    except Exception as e:
        logger.error(f"Error extracting tables with Tabula: {e}")
    
    return tables


def extract_tables(pdf_path: Path, pages: str = "all") -> List[ExtractedTable]:
    """
    Main function to extract tables from a PDF.
    Tries both Camelot methods and combines results.
    
    Args:
        pdf_path: Path to the PDF file
        pages: Pages to extract tables from (default: "all")
        
    Returns:
        List of ExtractedTable objects
    """
    logger.info(f"Starting table extraction from: {pdf_path.name}")
    
    all_tables = []
    
    # Try Camelot lattice first (for tables with lines)
    tables_lattice = extract_tables_with_camelot(pdf_path, pages)
    all_tables.extend(tables_lattice)
    
    # Also try stream mode (catches tables without lines)
    logger.info("Trying Camelot stream method for additional tables...")
    try:
        table_list = camelot.read_pdf(
            str(pdf_path),
            pages=pages,
            flavor='stream',
            suppress_stdout=True
        )
        
        logger.info(f"Camelot (stream) found {len(table_list)} additional tables")
        
        for idx, table in enumerate(table_list):
            df = table.df
            if df.shape[0] > 1 and df.shape[1] > 1:
                all_tables.append(ExtractedTable(
                    page_number=table.page,
                    table_index=idx + len(tables_lattice),
                    dataframe=df,
                    method='camelot-stream'
                ))
    except Exception as e:
        logger.error(f"Error with Camelot stream: {e}")
    
    # Remove duplicates (tables found by both methods)
    all_tables = _deduplicate_tables(all_tables)
    
    if len(all_tables) > 0:
        logger.info(f"Successfully extracted {len(all_tables)} total tables")
    else:
        logger.warning(f"No tables extracted from {pdf_path.name}")
    
    return all_tables


def _deduplicate_tables(tables: List[ExtractedTable]) -> List[ExtractedTable]:
    """Remove duplicate tables based on page and similarity."""
    if len(tables) <= 1:
        return tables
    
    unique_tables = []
    seen = set()
    
    for table in tables:
        # Create signature based on page and shape
        sig = (table.page_number, table.dataframe.shape)
        if sig not in seen:
            unique_tables.append(table)
            seen.add(sig)
    
    return unique_tables


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize an extracted table.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)
    
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def identify_financial_table(df: pd.DataFrame) -> Optional[str]:
    """
    Attempt to identify the type of financial table.
    
    Args:
        df: DataFrame containing the table
        
    Returns:
        Table type identifier or None
    """
    # Convert to string and make lowercase for matching
    text = " ".join([str(val).lower() for val in df.values.flatten()])
    
    if "balance sheet" in text or "assets" in text and "liabilities" in text:
        return "balance_sheet"
    elif "income statement" in text or "profit" in text and "loss" in text:
        return "income_statement"
    elif "cash flow" in text:
        return "cash_flow"
    elif "equity" in text:
        return "statement_of_equity"
    elif "notes" in text:
        return "notes"
    else:
        return None


def score_table_quality(df: pd.DataFrame) -> float:
    """
    Score the quality of an extracted table (0-1).
    Higher scores indicate tables more likely to contain useful financial data.
    
    Args:
        df: DataFrame to score
        
    Returns:
        Quality score between 0 and 1
    """
    score = 0.0
    
    # Check minimum size
    if df.shape[0] < 3 or df.shape[1] < 2:
        return 0.0
    
    # Count numeric cells
    numeric_count = 0
    empty_count = 0
    total_cells = df.shape[0] * df.shape[1]
    
    for col in df.columns:
        for val in df[col]:
            val_str = str(val).strip()
            if val_str in ['', 'nan', 'None']:
                empty_count += 1
            elif re.match(r'^[-+]?\(?[\d,]+\.?\d*\)?$', val_str):
                numeric_count += 1
    
    numeric_ratio = numeric_count / total_cells if total_cells > 0 else 0
    empty_ratio = empty_count / total_cells if total_cells > 0 else 0
    
    # Penalize tables with too many empty cells
    if empty_ratio > 0.5:
        return 0.0
    
    # Good tables have 15-70% numeric content
    if 0.15 <= numeric_ratio <= 0.7:
        score += 0.4
    elif numeric_ratio > 0:
        score += 0.2
    
    # Check for financial keywords
    text = ' '.join([str(val).lower() for val in df.values.flatten()])
    financial_keywords = [
        'revenue', 'profit', 'loss', 'assets', 'liabilities', 'equity',
        'cash', 'income', 'expenses', 'ebitda', 'total', 'current',
        'balance', 'statement', 'financial', 'year', 'amount'
    ]
    
    keyword_count = sum(1 for kw in financial_keywords if kw in text)
    if keyword_count >= 3:
        score += 0.4
    elif keyword_count >= 1:
        score += 0.2
    
    # Check structure - first column should have labels, others have values
    if df.shape[1] >= 2:
        first_col_text = sum(1 for val in df[df.columns[0]] if len(str(val).strip()) > 3)
        if first_col_text >= df.shape[0] * 0.5:
            score += 0.2
    
    return min(score, 1.0)


def filter_quality_tables(tables: List[ExtractedTable], min_quality: float = 0.3) -> List[ExtractedTable]:
    """
    Filter tables based on quality score.
    
    Args:
        tables: List of ExtractedTable objects
        min_quality: Minimum quality score (0-1)
        
    Returns:
        Filtered list of high-quality tables
    """
    filtered = []
    
    logger.info(f"Filtering {len(tables)} tables by quality (min_quality={min_quality})")
    
    for table in tables:
        quality = score_table_quality(table.dataframe)
        
        if quality >= min_quality:
            filtered.append(table)
            logger.debug(f"Table on page {table.page_number}: quality={quality:.2f} - KEPT")
        else:
            logger.debug(f"Table on page {table.page_number}: quality={quality:.2f} - FILTERED OUT")
    
    logger.info(f"Kept {len(filtered)} high-quality tables (filtered out {len(tables) - len(filtered)})")
    return filtered


def filter_large_tables(tables: List[ExtractedTable], max_size: int = MAX_TABLE_EXPORT_SIZE) -> List[ExtractedTable]:
    """
    Filter out tables that are too large.
    
    Args:
        tables: List of ExtractedTable objects
        max_size: Maximum number of rows per table
        
    Returns:
        Filtered list of tables
    """
    filtered = []
    
    for table in tables:
        if table.rows <= max_size:
            filtered.append(table)
        else:
            logger.warning(f"Table on page {table.page_number} is too large ({table.rows} rows), skipping")
    
    return filtered
