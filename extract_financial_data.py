"""
Module for extracting financial data using pattern matching and NLP.
Works across different company formats without deterministic rules.
"""
import logging
import re
from typing import Dict, List, Optional, Any
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)


class FinancialDataExtractor:
    """Extract financial metrics from text and tables using pattern matching."""
    
    # Common financial terms and their variations
    FINANCIAL_PATTERNS = {
        'revenue': [
            r'total\s+revenue', r'revenue\s+from\s+operations?', r'sales\s+revenue',
            r'gross\s+revenue', r'operating\s+revenue', r'total\s+income'
        ],
        'profit': [
            r'net\s+profit', r'profit\s+after\s+tax', r'profit\s+for\s+the\s+year',
            r'net\s+income', r'profit\s+attributable'
        ],
        'ebitda': [
            r'ebitda', r'earnings\s+before\s+interest'
        ],
        'assets': [
            r'total\s+assets', r'non-current\s+assets', r'current\s+assets'
        ],
        'equity': [
            r'total\s+equity', r'shareholders?\s+equity', r'net\s+worth',
            r'equity\s+share\s+capital'
        ],
        'liabilities': [
            r'total\s+liabilities', r'current\s+liabilities', r'non-current\s+liabilities'
        ],
        'cash_flow': [
            r'cash\s+flow\s+from\s+operating', r'net\s+cash\s+flow',
            r'operating\s+cash\s+flow'
        ],
        'earnings_per_share': [
            r'earnings?\s+per\s+share', r'eps', r'basic\s+eps', r'diluted\s+eps'
        ]
    }
    
    # Number patterns
    NUMBER_PATTERN = r'[-+]?\s*\(?[\d,]+\.?\d*\)?'
    
    def __init__(self):
        self.extracted_data = defaultdict(dict)
    
    def extract_from_text(self, text: str, year: str) -> Dict[str, Any]:
        """
        Extract financial metrics from text using pattern matching.
        
        Args:
            text: Text to extract from
            year: Year identifier
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        text_lower = text.lower()
        
        for metric_name, patterns in self.FINANCIAL_PATTERNS.items():
            for pattern in patterns:
                # Look for pattern followed by numbers
                regex = pattern + r'\s*[:\-]?\s*(' + self.NUMBER_PATTERN + r')'
                matches = re.finditer(regex, text_lower, re.IGNORECASE)
                
                for match in matches:
                    try:
                        value_str = match.group(1)
                        value = self._parse_number(value_str)
                        if value:
                            if metric_name not in metrics:
                                metrics[metric_name] = []
                            metrics[metric_name].append({
                                'value': value,
                                'raw': value_str,
                                'context': text[max(0, match.start()-50):match.end()+50]
                            })
                    except Exception as e:
                        logger.debug(f"Error parsing {metric_name}: {e}")
        
        return metrics
    
    def extract_from_tables(self, tables: List) -> Dict[str, Any]:
        """
        Extract financial data from tables using intelligent pattern matching.
        
        Args:
            tables: List of ExtractedTable objects
            
        Returns:
            Dictionary of extracted metrics
        """
        financial_data = {
            'balance_sheet': [],
            'income_statement': [],
            'cash_flow': [],
            'key_metrics': {}
        }
        
        for table in tables:
            df = table.dataframe
            table_type = self._identify_financial_statement(df)
            
            if table_type:
                parsed_data = self._parse_financial_table(df, table_type)
                if parsed_data:
                    financial_data[table_type].append({
                        'page': table.page_number,
                        'data': parsed_data
                    })
        
        return financial_data
    
    def _identify_financial_statement(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify the type of financial statement from table content.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Statement type or None
        """
        # Convert all text to lowercase for matching
        text = ' '.join([str(val).lower() for val in df.values.flatten()])
        
        # Score different statement types
        scores = {
            'balance_sheet': 0,
            'income_statement': 0,
            'cash_flow': 0
        }
        
        # Balance Sheet indicators
        bs_keywords = ['assets', 'liabilities', 'equity', 'current assets', 
                       'non-current', 'shareholders equity', 'reserves']
        scores['balance_sheet'] = sum(1 for kw in bs_keywords if kw in text)
        
        # Income Statement indicators
        is_keywords = ['revenue', 'expenses', 'profit', 'loss', 'income', 
                       'ebitda', 'operating', 'net profit']
        scores['income_statement'] = sum(1 for kw in is_keywords if kw in text)
        
        # Cash Flow indicators
        cf_keywords = ['cash flow', 'operating activities', 'investing activities',
                       'financing activities', 'cash and cash equivalents']
        scores['cash_flow'] = sum(1 for kw in cf_keywords if kw in text)
        
        # Return type with highest score if above threshold
        max_score = max(scores.values())
        if max_score >= 3:
            return max(scores, key=scores.get)
        
        return None
    
    def _parse_financial_table(self, df: pd.DataFrame, table_type: str) -> Dict:
        """
        Parse financial data from a table.
        
        Args:
            df: DataFrame to parse
            table_type: Type of financial statement
            
        Returns:
            Parsed financial data
        """
        parsed = {
            'type': table_type,
            'rows': []
        }
        
        # Try to identify year columns
        year_columns = self._identify_year_columns(df)
        
        for idx, row in df.iterrows():
            row_data = self._parse_table_row(row, year_columns)
            if row_data:
                parsed['rows'].append(row_data)
        
        return parsed
    
    def _identify_year_columns(self, df: pd.DataFrame) -> Dict:
        """
        Identify which columns contain year data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column indices to years
        """
        year_cols = {}
        
        # Check first few rows for year patterns
        for idx in range(min(3, len(df))):
            row = df.iloc[idx]
            for col_idx, val in enumerate(row):
                val_str = str(val)
                # Look for 4-digit years or year patterns
                year_match = re.search(r'\b(20\d{2}|19\d{2})\b', val_str)
                if year_match:
                    year_cols[col_idx] = year_match.group(1)
        
        return year_cols
    
    def _parse_table_row(self, row: pd.Series, year_columns: Dict) -> Optional[Dict]:
        """
        Parse a single table row.
        
        Args:
            row: Row to parse
            year_columns: Mapping of column indices to years
            
        Returns:
            Parsed row data
        """
        # First column usually contains the label
        label = str(row.iloc[0]).strip()
        
        # Skip empty or formatting rows
        if not label or len(label) < 3 or label in ['', 'nan', 'None']:
            return None
        
        # Extract numeric values
        values = {}
        for col_idx in range(1, len(row)):
            val = row.iloc[col_idx]
            number = self._parse_number(str(val))
            if number is not None:
                year_key = year_columns.get(col_idx, f'col_{col_idx}')
                values[year_key] = number
        
        if values:
            return {
                'label': label,
                'values': values
            }
        
        return None
    
    def _parse_number(self, text: str) -> Optional[float]:
        """
        Parse a number from text, handling various formats.
        
        Args:
            text: Text containing number
            
        Returns:
            Parsed number or None
        """
        if not text or text.lower() in ['nan', 'none', '', '-']:
            return None
        
        try:
            # Remove common formatting
            text = str(text).strip()
            
            # Check for parentheses (negative)
            is_negative = text.startswith('(') and text.endswith(')')
            
            # Remove non-numeric characters except . and -
            text = re.sub(r'[^\d.\-]', '', text)
            
            if not text or text == '-':
                return None
            
            number = float(text)
            return -number if is_negative else number
            
        except (ValueError, AttributeError):
            return None
    
    def extract_key_highlights(self, text: str) -> Dict[str, str]:
        """
        Extract key highlights or summary sections.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary of highlights
        """
        highlights = {}
        
        # Look for highlights section
        sections = [
            (r'financial\s+highlights?', 'financial_highlights'),
            (r'key\s+highlights?', 'key_highlights'),
            (r'performance\s+highlights?', 'performance_highlights'),
            (r'operational\s+highlights?', 'operational_highlights')
        ]
        
        text_lower = text.lower()
        
        for pattern, key in sections:
            match = re.search(pattern + r'(.*?)(?=\n\n[A-Z]|\Z)', text_lower, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if len(content) > 50:
                    highlights[key] = content[:1000]  # Limit length
        
        return highlights
    
    def score_table_quality(self, df: pd.DataFrame) -> float:
        """
        Score the quality of an extracted table (0-1).
        
        Args:
            df: DataFrame to score
            
        Returns:
            Quality score
        """
        score = 0.0
        
        # Check size
        if df.shape[0] < 3 or df.shape[1] < 2:
            return 0.0
        
        # Count numeric cells
        numeric_count = 0
        total_cells = df.shape[0] * df.shape[1]
        
        for col in df.columns:
            for val in df[col]:
                if self._parse_number(str(val)) is not None:
                    numeric_count += 1
        
        numeric_ratio = numeric_count / total_cells if total_cells > 0 else 0
        
        # Good tables have 20-80% numeric content
        if 0.2 <= numeric_ratio <= 0.8:
            score += 0.5
        
        # Check for financial keywords
        text = ' '.join([str(val).lower() for val in df.values.flatten()])
        financial_keywords = ['revenue', 'profit', 'assets', 'liabilities', 
                             'equity', 'cash', 'income', 'expenses']
        
        keyword_count = sum(1 for kw in financial_keywords if kw in text)
        if keyword_count >= 2:
            score += 0.3
        
        # Check structure (has labels and values)
        if df.shape[1] >= 2 and numeric_ratio > 0.1:
            score += 0.2
        
        return min(score, 1.0)
