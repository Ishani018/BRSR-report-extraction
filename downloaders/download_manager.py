"""
Download Manager - Orchestrates tiered download strategy for BRSR reports.
"""
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

from config.config import (
    BRSR_FINANCIAL_YEARS,
    DOWNLOAD_BASE_DIR,
    STATUS_DIR,
    DOWNLOAD_CHECKPOINT_FILE,
    DOWNLOAD_STATUS_FILE,
    NSE_RATE_LIMIT_DELAY,
    NSE_MAX_CONCURRENT
)

from .nse_downloader import NSEDownloader, get_nse_report
from .google_search_downloader import GoogleSearchDownloader, get_google_search_report
from .missing_reports_logger import MissingReportsLogger

logger = logging.getLogger(__name__)


def format_filename(company_name: str, year: str, is_standalone: bool = True) -> str:
    """
    Format filename for downloaded BRSR report.
    
    Args:
        company_name: Company name
        year: Financial year (e.g., '2022-23')
        is_standalone: True if standalone BRSR, False if annual report
        
    Returns:
        Formatted filename
    """
    # Clean company name for filename (remove special characters)
    clean_name = company_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    clean_name = clean_name.replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '')
    clean_name = clean_name.strip()
    
    if is_standalone:
        prefix = "BRSR"
    else:
        prefix = "AnnualReport"
    
    filename = f"{clean_name}_{prefix}_{year}.pdf"
    return filename


def download_brsr_report(
    company_name: str,
    symbol: str,
    website: str,
    year: str,
    output_dir: Path,
    nse_downloader: Optional[NSEDownloader] = None,
    google_downloader: Optional[GoogleSearchDownloader] = None,
    missing_logger: Optional[MissingReportsLogger] = None
) -> Dict:
    """
    Download BRSR report using tiered strategy:
    - Tier 1: NSE API
    - Tier 2: Google Search
    - Tier 3: Log to missing_reports.json
    
    Args:
        company_name: Company name
        symbol: NSE symbol
        website: Company website
        year: Financial year (e.g., '2022-23')
        output_dir: Directory to save downloaded PDF
        nse_downloader: Optional NSE downloader instance (creates new if None)
        google_downloader: Optional Google downloader instance (creates new if None)
        missing_logger: Optional missing reports logger (creates new if None)
        
    Returns:
        Dictionary with download status:
        {
            'success': bool,
            'tier': 'tier1' | 'tier2' | None,
            'file_path': str or None,
            'error': str or None,
            'is_standalone': bool  # True if standalone BRSR, False if annual report
        }
    """
    if nse_downloader is None:
        nse_downloader = NSEDownloader(rate_limit_delay=NSE_RATE_LIMIT_DELAY)
    
    if google_downloader is None:
        google_downloader = GoogleSearchDownloader()
    
    if missing_logger is None:
        missing_logger = MissingReportsLogger()
    
    # Create output directory for this year
    year_dir = Path(output_dir) / year
    year_dir.mkdir(parents=True, exist_ok=True)
    
    tier_1_error = None
    tier_2_error = None
    
    # Tier 1: Try NSE API (if symbol is available)
    if symbol:
        logger.debug(f"Tier 1: Attempting NSE API download for {company_name} ({symbol}) - {year}")
        output_path = year_dir / format_filename(company_name, year, is_standalone=True)
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"✓ Already downloaded: {output_path.name}")
            return {
                'success': True,
                'tier': 'tier1',
                'file_path': str(output_path),
                'error': None,
                'is_standalone': True
            }
        
        success, error = nse_downloader.download(symbol, output_path, year)
        
        if success:
            logger.info(f"✓ Tier 1 success: {company_name} ({year})")
            return {
                'success': True,
                'tier': 'tier1',
                'file_path': str(output_path),
                'error': None,
                'is_standalone': True
            }
        else:
            tier_1_error = error or "No results from NSE API"
            logger.info(f"✗ Tier 1 failed: {company_name} ({year}) - {tier_1_error}")
    else:
        tier_1_error = "No NSE symbol available"
        logger.debug(f"Skipping Tier 1 for {company_name}: {tier_1_error}")
    
    # Tier 2: Try Google Search (if website is available)
    if website:
        logger.debug(f"Tier 2: Attempting Google Search for {company_name} - {year}")
        output_path = year_dir / format_filename(company_name, year, is_standalone=True)
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"✓ Already downloaded: {output_path.name}")
            return {
                'success': True,
                'tier': 'tier2',
                'file_path': str(output_path),
                'error': None,
                'is_standalone': True
            }
        
        success, error = google_downloader.download(company_name, website, year, output_path)
        
        if success:
            logger.info(f"✓ Tier 2 success: {company_name} ({year})")
            return {
                'success': True,
                'tier': 'tier2',
                'file_path': str(output_path),
                'error': None,
                'is_standalone': True
            }
        else:
            tier_2_error = error or "No results from Google Search"
            logger.info(f"✗ Tier 2 failed: {company_name} ({year}) - {tier_2_error}")
    else:
        tier_2_error = "No website available"
        logger.debug(f"Skipping Tier 2 for {company_name}: {tier_2_error}")
    
    # Tier 3: Log to missing_reports.json
    logger.info(f"✗ All tiers failed: {company_name} ({year}) - Logging to missing_reports.json")
    missing_logger.add_entry(
        company_name=company_name,
        symbol=symbol or '',
        website=website or '',
        year=year,
        tier_1_error=tier_1_error,
        tier_2_error=tier_2_error
    )
    
    return {
        'success': False,
        'tier': None,
        'file_path': None,
        'error': f"Tier 1: {tier_1_error}, Tier 2: {tier_2_error}",
        'is_standalone': None
    }


class DownloadManager:
    """
    Download Manager for batch processing BRSR reports.
    """
    
    def __init__(
        self,
        output_base_dir: Optional[Path] = None,
        checkpoint_file: Optional[Path] = None,
        status_file: Optional[Path] = None,
        max_workers: int = NSE_MAX_CONCURRENT
    ):
        """
        Initialize Download Manager.
        
        Args:
            output_base_dir: Base directory for downloads (defaults to config)
            checkpoint_file: Path to checkpoint file (defaults to config)
            status_file: Path to status file (defaults to config)
            max_workers: Maximum concurrent downloads (defaults to config)
        """
        self.output_base_dir = Path(output_base_dir) if output_base_dir else DOWNLOAD_BASE_DIR
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else DOWNLOAD_CHECKPOINT_FILE
        self.status_file = Path(status_file) if status_file else DOWNLOAD_STATUS_FILE
        self.max_workers = max_workers
        
        # Ensure directories exist
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize downloaders
        self.nse_downloader = NSEDownloader(rate_limit_delay=NSE_RATE_LIMIT_DELAY)
        self.google_downloader = GoogleSearchDownloader()
        self.missing_logger = MissingReportsLogger()
        
        # Status tracking
        self.status = self.load_status()
        self.checkpoint = self.load_checkpoint()
        
    def load_checkpoint(self) -> Dict:
        """Load checkpoint from JSON file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return {}
        return {}
    
    def save_checkpoint(self, checkpoint: Dict) -> None:
        """Save checkpoint to JSON file."""
        try:
            checkpoint['last_updated'] = datetime.now().isoformat()
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def load_status(self) -> Dict:
        """Load status from JSON file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading status: {e}")
                return {'downloads': []}
        return {'downloads': []}
    
    def save_status(self) -> None:
        """Save status to JSON file."""
        try:
            self.status['last_updated'] = datetime.now().isoformat()
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving status: {e}")
    
    def is_downloaded(self, company_name: str, year: str) -> bool:
        """
        Check if report is already downloaded.
        
        Args:
            company_name: Company name
            year: Financial year
            
        Returns:
            True if already downloaded
        """
        # Check status file
        for download in self.status.get('downloads', []):
            if (download.get('company_name') == company_name and
                download.get('year') == year and
                download.get('success') == True):
                return True
        
        # Check if file exists
        year_dir = self.output_base_dir / year
        if year_dir.exists():
            # Check for both standalone and annual report formats
            standalone_file = year_dir / format_filename(company_name, year, is_standalone=True)
            annual_file = year_dir / format_filename(company_name, year, is_standalone=False)
            
            if standalone_file.exists() or annual_file.exists():
                return True
        
        return False
    
    def download_single(
        self,
        company_name: str,
        symbol: str,
        website: str,
        year: str
    ) -> Dict:
        """
        Download BRSR report for single company/year.
        
        Args:
            company_name: Company name
            symbol: NSE symbol
            website: Company website
            year: Financial year
            
        Returns:
            Download status dictionary
        """
        result = download_brsr_report(
            company_name=company_name,
            symbol=symbol,
            website=website,
            year=year,
            output_dir=self.output_base_dir,
            nse_downloader=self.nse_downloader,
            google_downloader=self.google_downloader,
            missing_logger=self.missing_logger
        )
        
        # Add metadata
        result['company_name'] = company_name
        result['symbol'] = symbol
        result['website'] = website
        result['year'] = year
        result['timestamp'] = datetime.now().isoformat()
        
        # Update status
        self.status.setdefault('downloads', []).append(result)
        self.save_status()
        
        return result
    
    def batch_download(
        self,
        companies_df: pd.DataFrame,
        years: Optional[List[str]] = None,
        resume: bool = True
    ) -> Dict:
        """
        Batch download BRSR reports for multiple companies and years.
        
        Args:
            companies_df: DataFrame with company information (columns: company_name, symbol, website)
            years: List of financial years (defaults to config years)
            resume: If True, skip already downloaded files
            
        Returns:
            Summary dictionary with statistics
        """
        if years is None:
            years = BRSR_FINANCIAL_YEARS
        
        logger.info(f"Starting batch download for {len(companies_df)} companies × {len(years)} years = {len(companies_df) * len(years)} downloads")
        
        # Prepare download tasks
        tasks = []
        for _, row in companies_df.iterrows():
            company_name = row.get('company_name', '')
            symbol = str(row.get('symbol', '')).strip() if pd.notna(row.get('symbol')) else ''
            website = str(row.get('website', '')).strip() if pd.notna(row.get('website')) else ''
            
            if not company_name:
                continue
            
            for year in years:
                # Check if already downloaded
                if resume and self.is_downloaded(company_name, year):
                    logger.debug(f"Skipping {company_name} ({year}) - already downloaded")
                    continue
                
                tasks.append({
                    'company_name': company_name,
                    'symbol': symbol,
                    'website': website,
                    'year': year
                })
        
        logger.info(f"Prepared {len(tasks)} download tasks")
        
        # Process downloads with progress bar
        results = []
        successful = 0
        failed = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self.download_single,
                    task['company_name'],
                    task['symbol'],
                    task['website'],
                    task['year']
                ): task
                for task in tasks
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(tasks), desc="Downloading BRSR reports", unit="file") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            successful += 1
                            pbar.set_postfix({'success': successful, 'failed': failed, 'tier': result.get('tier', 'N/A')})
                        else:
                            failed += 1
                            pbar.set_postfix({'success': successful, 'failed': failed})
                    except Exception as e:
                        logger.error(f"Error processing {task['company_name']} ({task['year']}): {e}", exc_info=True)
                        failed += 1
                        pbar.set_postfix({'success': successful, 'failed': failed})
                    finally:
                        pbar.update(1)
        
        # Generate summary
        summary = {
            'total_tasks': len(tasks),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(tasks) * 100) if tasks else 0,
            'tier_breakdown': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate tier breakdown
        for result in results:
            tier = result.get('tier', 'none')
            summary['tier_breakdown'][tier] = summary['tier_breakdown'].get(tier, 0) + 1
        
        logger.info(f"\n{'='*80}")
        logger.info("BATCH DOWNLOAD SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total tasks: {summary['total_tasks']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Tier breakdown: {summary['tier_breakdown']}")
        
        # Save summary to status  
        self.status['last_summary'] = summary
        self.status['downloads'].extend(results)
        self.save_status()
        
        return summary


def batch_download(
    companies_df: pd.DataFrame,
    years: Optional[List[str]] = None,
    output_base_dir: Optional[Path] = None,
    resume: bool = True,
    max_workers: int = NSE_MAX_CONCURRENT
) -> Dict:
    """
    Convenience function for batch downloading BRSR reports.
    
    Args:
        companies_df: DataFrame with company information
        years: List of financial years
        output_base_dir: Base directory for downloads
        resume: If True, skip already downloaded files
        max_workers: Maximum concurrent downloads
        
    Returns:
        Summary dictionary
    """
    manager = DownloadManager(
        output_base_dir=output_base_dir,
        max_workers=max_workers
    )
    
    return manager.batch_download(companies_df, years, resume)


if __name__ == "__main__":
    # Test the download manager
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with sample data
    test_companies = pd.DataFrame([
        {'company_name': 'TCS', 'symbol': 'TCS', 'website': 'https://www.tcs.com'},
        {'company_name': 'Reliance Industries', 'symbol': 'RELIANCE', 'website': 'https://www.ril.com'},
    ])
    
    test_years = ['2023-24']
    
    logger.info("Testing Download Manager with sample companies...")
    manager = DownloadManager(max_workers=2)
    summary = manager.batch_download(test_companies, test_years, resume=False)
    
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    sys.exit(0)

