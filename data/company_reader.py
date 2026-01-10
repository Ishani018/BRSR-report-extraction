"""
Company Data Reader - Parse NIFTY 500 Excel file and extract company information.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)


class CompanyReader:
    """Reads and processes company data from NIFTY 500 Excel file."""
    
    def __init__(self, excel_path: Path):
        """
        Initialize CompanyReader with Excel file path.
        
        Args:
            excel_path: Path to NIFTY 500 firms.xlsx file
        """
        self.excel_path = Path(excel_path)
        self.companies = []
        self.symbol_lookup = {}
        
    def read_excel(self) -> pd.DataFrame:
        """
        Read the Excel file and return DataFrame.
        
        Returns:
            DataFrame with company information
        """
        logger.info(f"Reading Excel file: {self.excel_path}")
        
        try:
            df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded {len(df)} rows from Excel file")
            return df
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise
    
    def clean_company_name(self, name: str) -> str:
        """
        Clean and normalize company name for consistent naming.
        
        Args:
            name: Raw company name from Excel
            
        Returns:
            Cleaned company name
        """
        if pd.isna(name) or not name:
            return ""
        
        # Remove extra whitespace
        name = str(name).strip()
        
        # Remove common suffixes for consistency
        suffixes = [" Ltd.", " Limited", " Pvt Ltd", " Private Limited"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        return name
    
    def extract_companies(self, df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Extract company information from DataFrame.
        
        Args:
            df: Optional DataFrame (if None, reads from Excel)
            
        Returns:
            List of company dictionaries with name, symbol, industry, etc.
        """
        if df is None:
            df = self.read_excel()
        
        logger.info("Extracting company information...")
        
        companies = []
        
        # Map Excel columns to our structure
        for idx, row in df.iterrows():
            company_name = self.clean_company_name(row.get('Company Name', ''))
            symbol = str(row.get('Symbol', '')).strip() if pd.notna(row.get('Symbol')) else ''
            industry = str(row.get('Industry', '')).strip() if pd.notna(row.get('Industry')) else ''
            isin = str(row.get('ISIN Code', '')).strip() if pd.notna(row.get('ISIN Code')) else ''
            series = str(row.get('Series', '')).strip() if pd.notna(row.get('Series')) else ''
            
            # Try to get serial number from various possible column names
            serial_number = None
            for col_name in ['Serial Number', 'Serial No', 'S.No', 'S No', 'Sr No', 'Sr. No', 'Serial', 'SN', 'S.N']:
                if col_name in row and pd.notna(row.get(col_name)):
                    try:
                        serial_number = int(row.get(col_name))
                        break
                    except (ValueError, TypeError):
                        pass
            
            # If no serial number column found, use row index (1-based)
            if serial_number is None:
                serial_number = idx + 1  # 1-based index
            
            # Skip rows without company name or symbol
            if not company_name or not symbol:
                continue
            
            company = {
                'company_name': company_name,
                'symbol': symbol,
                'industry': industry,
                'isin': isin,
                'series': series,
                'row_index': idx + 1,  # 1-based index (Excel row)
                'serial_number': serial_number  # Serial number from Excel or row_index
            }
            
            companies.append(company)
            
            # Build symbol lookup (company name -> symbol)
            self.symbol_lookup[company_name.lower()] = symbol
            
        self.companies = companies
        logger.info(f"Extracted {len(companies)} companies")
        
        return companies
    
    def get_symbol(self, company_name: str) -> Optional[str]:
        """
        Get NSE symbol for a company name.
        
        Args:
            company_name: Company name (case-insensitive)
            
        Returns:
            NSE symbol or None if not found
        """
        return self.symbol_lookup.get(company_name.lower())
    
    def save_to_json(self, output_path: Path) -> Path:
        """
        Save company data to JSON file.
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'total_companies': len(self.companies),
            'companies': self.companies,
            'symbol_lookup': self.symbol_lookup
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved company data to {output_path}")
        return output_path
    
    def save_to_csv(self, output_path: Path) -> Path:
        """
        Save company data to CSV file.
        
        Args:
            output_path: Path to save CSV file
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.companies)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved company data to {output_path}")
        return output_path
    
    def get_companies_dataframe(self) -> pd.DataFrame:
        """
        Get companies as DataFrame.
        
        Returns:
            DataFrame with company information
        """
        return pd.DataFrame(self.companies)


def read_companies(excel_path: Path) -> CompanyReader:
    """
    Convenience function to read companies from Excel file.
    
    Args:
        excel_path: Path to NIFTY 500 firms.xlsx file
        
    Returns:
        CompanyReader instance with loaded data
    """
    reader = CompanyReader(excel_path)
    reader.extract_companies()
    return reader


if __name__ == "__main__":
    # Test the reader
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Find Excel file (assume it's in parent directory)
    base_dir = Path(__file__).parent.parent.parent
    excel_path = base_dir / "NIFTY 500 firms.xlsx"
    
    if not excel_path.exists():
        print(f"Excel file not found at: {excel_path}")
        sys.exit(1)
    
    reader = read_companies(excel_path)
    
    # Print summary
    print(f"\nLoaded {len(reader.companies)} companies")
    print(f"\nFirst 5 companies:")
    for i, company in enumerate(reader.companies[:5], 1):
        print(f"{i}. {company['company_name']} ({company['symbol']})")
    
    # Test symbol lookup
    if reader.companies:
        test_name = reader.companies[0]['company_name']
        symbol = reader.get_symbol(test_name)
        print(f"\nSymbol lookup test: '{test_name}' -> {symbol}")
    
    # Save to JSON and CSV (optional)
    output_dir = base_dir / "pdf-to-structured-reports" / "data"
    reader.save_to_json(output_dir / "companies.json")
    reader.save_to_csv(output_dir / "companies.csv")
    
    print(f"\nCompany data saved to: {output_dir}")

