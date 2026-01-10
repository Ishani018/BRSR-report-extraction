"""
Missing Reports Logger - Tier 3: Log failures to missing_reports.json for manual review.
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from config.config import MISSING_REPORTS_FILE

logger = logging.getLogger(__name__)


class MissingReportsLogger:
    """
    Logger for tracking missing reports when all download tiers fail.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize Missing Reports Logger.
        
        Args:
            log_file: Path to missing_reports.json file (defaults to config value)
        """
        self.log_file = Path(log_file) if log_file else MISSING_REPORTS_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []
        
        # Load existing entries if file exists
        self.load()
    
    def load(self) -> None:
        """Load existing missing reports from JSON file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both old format (list) and new format (dict with entries)
                    if isinstance(data, list):
                        self.entries = data
                    elif isinstance(data, dict) and 'entries' in data:
                        self.entries = data['entries']
                    else:
                        self.entries = []
                
                logger.info(f"Loaded {len(self.entries)} existing missing report entries")
            except Exception as e:
                logger.error(f"Error loading missing reports file: {e}")
                self.entries = []
        else:
            logger.debug("Missing reports file does not exist, starting fresh")
            self.entries = []
    
    def save(self) -> None:
        """Save missing reports to JSON file."""
        try:
            data = {
                'total_entries': len(self.entries),
                'last_updated': datetime.now().isoformat(),
                'entries': self.entries
            }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.entries)} missing report entries to {self.log_file}")
        except Exception as e:
            logger.error(f"Error saving missing reports file: {e}")
            raise
    
    def add_entry(
        self,
        company_name: str,
        symbol: str,
        website: str,
        year: str,
        tier_1_error: Optional[str] = None,
        tier_2_error: Optional[str] = None
    ) -> Dict:
        """
        Add a missing report entry.
        
        Args:
            company_name: Company name
            symbol: NSE symbol
            website: Company website
            year: Financial year (e.g., '2022-23')
            tier_1_error: Error message from Tier 1 (NSE API)
            tier_2_error: Error message from Tier 2 (Google Search)
            
        Returns:
            Created entry dictionary
        """
        entry = {
            'company_name': company_name,
            'symbol': symbol,
            'website': website or '',
            'year': year,
            'tier_1_error': tier_1_error or 'No results from NSE API',
            'tier_2_error': tier_2_error or 'No results from Google Search',
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if entry already exists (same company, symbol, year)
        existing = self.find_entry(company_name, symbol, year)
        if existing:
            logger.debug(f"Entry already exists for {company_name} ({year}), updating...")
            # Update existing entry
            existing.update(entry)
        else:
            # Add new entry
            self.entries.append(entry)
            logger.info(f"Added missing report entry: {company_name} ({symbol}) - {year}")
        
        # Save immediately
        self.save()
        
        return entry
    
    def find_entry(self, company_name: str, symbol: str, year: str) -> Optional[Dict]:
        """
        Find existing entry for company, symbol, and year.
        
        Args:
            company_name: Company name
            symbol: NSE symbol
            year: Financial year
            
        Returns:
            Entry dictionary if found, None otherwise
        """
        for entry in self.entries:
            if (entry.get('company_name') == company_name and
                entry.get('symbol') == symbol and
                entry.get('year') == year):
                return entry
        return None
    
    def get_summary(self) -> Dict:
        """
        Get summary of missing reports.
        
        Returns:
            Dictionary with summary statistics
        """
        total = len(self.entries)
        
        # Count by company
        companies = {}
        for entry in self.entries:
            company = entry.get('company_name', 'Unknown')
            companies[company] = companies.get(company, 0) + 1
        
        # Count by year
        years = {}
        for entry in self.entries:
            year = entry.get('year', 'Unknown')
            years[year] = years.get(year, 0) + 1
        
        # Count by tier failure
        tier_1_failures = sum(1 for e in self.entries if e.get('tier_1_error'))
        tier_2_failures = sum(1 for e in self.entries if e.get('tier_2_error'))
        
        return {
            'total_missing_reports': total,
            'unique_companies': len(companies),
            'companies': companies,
            'years': years,
            'tier_1_failures': tier_1_failures,
            'tier_2_failures': tier_2_failures,
            'last_updated': datetime.now().isoformat()
        }
    
    def export_summary(self, output_path: Optional[Path] = None) -> Path:
        """
        Export summary to JSON file.
        
        Args:
            output_path: Path to save summary (defaults to log_file with '_summary' suffix)
            
        Returns:
            Path to saved summary file
        """
        if output_path is None:
            output_path = self.log_file.parent / f"{self.log_file.stem}_summary.json"
        
        output_path = Path(output_path)
        summary = self.get_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported summary to: {output_path}")
        return output_path
    
    def get_entries_by_company(self, company_name: str) -> List[Dict]:
        """
        Get all entries for a specific company.
        
        Args:
            company_name: Company name
            
        Returns:
            List of entry dictionaries
        """
        return [e for e in self.entries if e.get('company_name') == company_name]
    
    def get_entries_by_year(self, year: str) -> List[Dict]:
        """
        Get all entries for a specific year.
        
        Args:
            year: Financial year
            
        Returns:
            List of entry dictionaries
        """
        return [e for e in self.entries if e.get('year') == year]
    
    def clear(self) -> None:
        """Clear all entries."""
        self.entries = []
        self.save()
        logger.info("Cleared all missing report entries")


def log_missing_report(
    company_name: str,
    symbol: str,
    website: str,
    year: str,
    tier_1_error: Optional[str] = None,
    tier_2_error: Optional[str] = None,
    logger_instance: Optional[MissingReportsLogger] = None
) -> Dict:
    """
    Convenience function to log a missing report.
    
    Args:
        company_name: Company name
        symbol: NSE symbol
        website: Company website
        year: Financial year
        tier_1_error: Error message from Tier 1
        tier_2_error: Error message from Tier 2
        logger_instance: Optional MissingReportsLogger instance (creates new if None)
        
    Returns:
        Created entry dictionary
    """
    if logger_instance is None:
        logger_instance = MissingReportsLogger()
    
    return logger_instance.add_entry(company_name, symbol, website, year, tier_1_error, tier_2_error)


if __name__ == "__main__":
    # Test the logger
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with temporary file
    test_file = Path(__file__).parent.parent / "status" / "test_missing_reports.json"
    logger = MissingReportsLogger(test_file)
    
    # Add test entries
    logger.add_entry(
        company_name="Test Company 1",
        symbol="TEST1",
        website="https://test1.com",
        year="2022-23",
        tier_1_error="No results from NSE API",
        tier_2_error="No results from Google Search"
    )
    
    logger.add_entry(
        company_name="Test Company 2",
        symbol="TEST2",
        website="https://test2.com",
        year="2023-24",
        tier_1_error="403 Forbidden",
        tier_2_error="No PDF links found"
    )
    
    # Get summary
    summary = logger.get_summary()
    print(f"\nSummary:")
    print(json.dumps(summary, indent=2))
    
    # Export summary
    summary_path = logger.export_summary()
    print(f"\nSummary exported to: {summary_path}")
    
    print(f"\n✓ Missing Reports Logger test completed")
    sys.exit(0)

