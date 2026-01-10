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

logger = logging.getLogger(__name__)


def format_filename(company_name: str, year: str, is_standalone: bool = True, symbol: Optional[str] = None, serial_number: Optional[int] = None) -> str:
    """
    Format filename for downloaded BRSR report.
    Format: {SerialNumber}_{SYMBOL}_BRSR_{Year}.pdf
    Uses company SYMBOL from Excel/CSV if available, otherwise uses cleaned company name.
    Matches the expected naming convention from the Excel/CSV file.
    
    Args:
        company_name: Company name (used as fallback if symbol not available)
        year: Financial year (e.g., '2022-23')
        is_standalone: True if standalone BRSR, False if annual report
        symbol: Optional company symbol from Excel/CSV (e.g., 'RELIANCE', 'TCS')
        serial_number: Optional serial number from Excel/CSV (placed at the front)
        
    Returns:
        Formatted filename: {SerialNumber}_{SYMBOL}_BRSR_{Year}.pdf or {SerialNumber}_{SYMBOL}_AnnualReport_{Year}.pdf
        If symbol not available: {SerialNumber}_{CompanyName}_BRSR_{Year}.pdf
        If serial_number not available: {SYMBOL}_BRSR_{Year}.pdf
    """
    # Format serial number with leading zeros (3 digits for up to 999, adjust as needed)
    serial_prefix = ""
    if serial_number is not None:
        # Format with leading zeros (e.g., 001, 002, 010, 100)
        serial_prefix = f"{serial_number:03d}_"  # 3-digit format with leading zeros
    
    # Use symbol if available (preferred - matches CSV/Excel format)
    if symbol and symbol.strip():
        symbol = symbol.strip().upper()  # Normalize symbol to uppercase
        if is_standalone:
            filename = f"{serial_prefix}{symbol}_BRSR_{year}.pdf"
        else:
            filename = f"{serial_prefix}{symbol}_AnnualReport_{year}.pdf"
        return filename
    
    # Fallback to company name if symbol not available
    try:
        from pipeline.file_naming import clean_company_name
        
        # Use the standardized naming convention
        cleaned_name = clean_company_name(company_name)
        
        if is_standalone:
            filename = f"{serial_prefix}{cleaned_name}_BRSR_{year}.pdf"
        else:
            filename = f"{serial_prefix}{cleaned_name}_AnnualReport_{year}.pdf"
        
        return filename
    except ImportError:
        # Fallback if module not available
        import re
        # Clean company name for filename (remove special characters)
        clean_name = company_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        clean_name = clean_name.replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '')
        clean_name = clean_name.strip()
        clean_name = re.sub(r'\s+', '_', clean_name)  # Replace spaces with underscores
        
        if is_standalone:
            prefix = "BRSR"
        else:
            prefix = "AnnualReport"
        
        filename = f"{serial_prefix}{clean_name}_{prefix}_{year}.pdf"
        return filename


def download_brsr_report(
    company_name: str,
    symbol: str,
    year: str,
    output_base_dir: Path,
    serial_number: Optional[int] = None,
    nse_downloader: Optional[NSEDownloader] = None
) -> Dict:
    """
    Download BRSR report from NSE API only (no fallbacks).
    Structure: {output_base_dir}/{company_folder}/{year}/{filename}.pdf
    Skips download if NSE API fails (no fallbacks).
    
    Args:
        company_name: Company name (used for folder name)
        symbol: NSE symbol (required for download)
        year: Financial year (e.g., '2022-23')
        output_base_dir: Base directory for downloads
        serial_number: Optional serial number for filename
        nse_downloader: Optional NSE downloader instance (creates new if None)
        
    Returns:
        Dictionary with download status:
        {
            'success': bool,
            'file_path': str or None,
            'error': str or None
        }
    """
    if nse_downloader is None:
        nse_downloader = NSEDownloader(rate_limit_delay=NSE_RATE_LIMIT_DELAY)
    
    # Create company folder structure: {company}/{year}/
    # Use serial number + symbol for company folder name if serial_number available
    from pipeline.file_naming import clean_company_name
    if serial_number is not None:
        company_folder = f"{serial_number:03d}_{symbol.upper()}"
    else:
        company_folder = symbol.upper() if symbol else clean_company_name(company_name)
    
    company_dir = Path(output_base_dir) / company_folder
    year_dir = company_dir / year
    year_dir.mkdir(parents=True, exist_ok=True)
    
    # Only try NSE API - skip if symbol not available or fails
    if not symbol or not symbol.strip():
        error_msg = "No NSE symbol available"
        logger.info(f"✗ Skipping {company_name} ({year}): {error_msg}")
        return {
            'success': False,
            'file_path': None,
            'error': error_msg
        }
    
    # Format filename exactly as specified in CSV
    filename = format_filename(company_name, year, is_standalone=True, symbol=symbol, serial_number=serial_number)
    output_path = year_dir / filename
    
    # Check if already downloaded
    if output_path.exists():
        logger.info(f"✓ Already downloaded: {company_folder}/{year}/{filename}")
        return {
            'success': True,
            'file_path': str(output_path),
            'error': None
        }
    
    # Attempt NSE API download
    logger.debug(f"Attempting NSE API download for {company_name} ({symbol}) - {year}")
    success, error = nse_downloader.download(symbol, output_path, year)
    
    if success:
        logger.info(f"✓ Successfully downloaded: {company_folder}/{year}/{filename}")
        return {
            'success': True,
            'file_path': str(output_path),
            'error': None
        }
    else:
        error_msg = error or "No results from NSE API"
        logger.info(f"✗ Failed to download {company_name} ({year}): {error_msg}")
        # Skip if fails - no fallbacks
        return {
            'success': False,
            'file_path': None,
            'error': error_msg
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
        
        # Initialize NSE downloader only (no fallbacks)
        self.nse_downloader = NSEDownloader(rate_limit_delay=NSE_RATE_LIMIT_DELAY)
        
        # Status CSV tracking (replaces JSON status)
        self.status_csv_path = STATUS_DIR / "download_status.csv"
        self.status_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_df = self.load_status_csv()
        
    def load_status_csv(self) -> pd.DataFrame:
        """Load status from CSV file or create new DataFrame."""
        if self.status_csv_path.exists():
            try:
                df = pd.read_csv(self.status_csv_path)
                logger.info(f"Loaded status from {self.status_csv_path} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.error(f"Error loading status CSV: {e}")
                return pd.DataFrame()
        else:
            logger.info("Creating new status CSV")
            return pd.DataFrame()
    
    def save_status_csv(self) -> None:
        """Save status to CSV file."""
        try:
            self.status_df.to_csv(self.status_csv_path, index=False)
            logger.debug(f"Saved status to {self.status_csv_path} ({len(self.status_df)} rows)")
        except Exception as e:
            logger.error(f"Error saving status CSV: {e}")
    
    def is_downloaded(self, symbol: str, year: str, serial_number: Optional[int] = None) -> bool:
        """
        Check if report is already downloaded.
        Uses company-based folder structure: {company}/{year}/
        
        Args:
            symbol: Company symbol
            year: Financial year
            serial_number: Optional serial number
            
        Returns:
            True if already downloaded
        """
        # Check status CSV
        if not self.status_df.empty:
            mask = (self.status_df['symbol'] == symbol) & (self.status_df['year'] == year)
            if serial_number is not None:
                mask = mask & (self.status_df['serial_number'] == serial_number)
            downloaded = self.status_df[mask]
            if not downloaded.empty and downloaded.iloc[0]['status'] == 'Downloaded':
                return True
        
        # Check if file exists in company/year folder structure
        if serial_number is not None:
            company_folder = f"{serial_number:03d}_{symbol.upper()}"
        else:
            company_folder = symbol.upper()
        
        company_dir = self.output_base_dir / company_folder / year
        if company_dir.exists():
            # Check if any PDF exists in the year folder
            pdf_files = list(company_dir.glob("*.pdf"))
            if pdf_files:
                return True
        
        return False
    
    def download_single(
        self,
        company_name: str,
        symbol: str,
        year: str,
        serial_number: Optional[int] = None
    ) -> Dict:
        """
        Download BRSR report for single company/year.
        Only uses NSE API - no fallbacks.
        
        Args:
            company_name: Company name
            symbol: NSE symbol
            year: Financial year
            serial_number: Optional serial number from Excel/CSV
            
        Returns:
            Download status dictionary
        """
        result = download_brsr_report(
            company_name=company_name,
            symbol=symbol,
            year=year,
            output_base_dir=self.output_base_dir,
            serial_number=serial_number,
            nse_downloader=self.nse_downloader
        )
        
        # Add metadata
        result['company_name'] = company_name
        result['symbol'] = symbol
        result['year'] = year
        result['serial_number'] = serial_number
        result['timestamp'] = datetime.now().isoformat()
        result['status'] = 'Downloaded' if result['success'] else 'Failed'
        
        # Update status CSV DataFrame
        new_row = {
            'serial_number': serial_number or '',
            'company_name': company_name,
            'symbol': symbol,
            'year': year,
            'status': result['status'],
            'error': result.get('error', ''),
            'file_path': result.get('file_path', ''),
            'timestamp': result['timestamp']
        }
        
        # Add or update row in DataFrame
        if self.status_df.empty:
            self.status_df = pd.DataFrame([new_row])
        else:
            # Check if row exists
            mask = (self.status_df['symbol'] == symbol) & (self.status_df['year'] == year)
            if serial_number is not None:
                mask = mask & (self.status_df['serial_number'] == serial_number)
            
            existing = self.status_df[mask]
            if not existing.empty:
                # Update existing row
                idx = existing.index[0]
                for key, value in new_row.items():
                    self.status_df.at[idx, key] = value
            else:
                # Add new row
                self.status_df = pd.concat([self.status_df, pd.DataFrame([new_row])], ignore_index=True)
        
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
            companies_df: DataFrame with company information (must have: company_name, symbol, serial_number)
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
            company_name = str(row.get('company_name', '')).strip()
            symbol = str(row.get('symbol', '')).strip() if pd.notna(row.get('symbol')) else ''
            
            # Get serial number (prefer from Excel, fallback to row_index)
            serial_number = None
            if pd.notna(row.get('serial_number')):
                try:
                    serial_number = int(row.get('serial_number'))
                except (ValueError, TypeError):
                    pass
            if serial_number is None and pd.notna(row.get('row_index')):
                try:
                    serial_number = int(row.get('row_index'))
                except (ValueError, TypeError):
                    pass
            
            if not company_name or not symbol:
                logger.warning(f"Skipping row with missing company_name or symbol: {row}")
                continue
            
            for year in years:
                # Check if already downloaded (uses company-based folder structure)
                if resume and self.is_downloaded(symbol, year, serial_number=serial_number):
                    logger.debug(f"Skipping {company_name} ({year}) - already downloaded")
                    continue
                
                tasks.append({
                    'company_name': company_name,
                    'symbol': symbol,
                    'year': year,
                    'serial_number': serial_number
                })
        
        logger.info(f"Prepared {len(tasks)} download tasks")
        
        # Process downloads with progress bar
        results = []
        successful = 0
        failed = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks (no website parameter needed - only NSE API)
            future_to_task = {
                executor.submit(
                    self.download_single,
                    task['company_name'],
                    task['symbol'],
                    task['year'],
                    task.get('serial_number')  # Pass serial_number if available
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
                        else:
                            failed += 1
                            # Add failed result to status CSV
                            new_row = {
                                'serial_number': task.get('serial_number') or '',
                                'company_name': task['company_name'],
                                'symbol': task['symbol'],
                                'year': task['year'],
                                'status': 'Failed',
                                'error': result.get('error', 'Unknown error'),
                                'file_path': '',
                                'timestamp': datetime.now().isoformat()
                            }
                            if self.status_df.empty:
                                self.status_df = pd.DataFrame([new_row])
                            else:
                                self.status_df = pd.concat([self.status_df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        pbar.set_postfix({'success': successful, 'failed': failed})
                    except Exception as e:
                        logger.error(f"Error processing {task['company_name']} ({task['year']}): {e}", exc_info=True)
                        failed += 1
                        # Add failed result to status CSV
                        new_row = {
                            'serial_number': task.get('serial_number') or '',
                            'company_name': task['company_name'],
                            'symbol': task['symbol'],
                            'year': task['year'],
                            'status': 'Failed',
                            'error': str(e),
                            'file_path': '',
                            'timestamp': datetime.now().isoformat()
                        }
                        if self.status_df.empty:
                            self.status_df = pd.DataFrame([new_row])
                        else:
                            self.status_df = pd.concat([self.status_df, pd.DataFrame([new_row])], ignore_index=True)
                        pbar.set_postfix({'success': successful, 'failed': failed})
                    finally:
                        pbar.update(1)
        
        # Save final status CSV
        self.save_status_csv()
        
        # Generate summary
        summary = {
            'total_tasks': len(tasks),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(tasks) * 100) if tasks else 0,
            'timestamp': datetime.now().isoformat(),
            'status_csv_path': str(self.status_csv_path)
        }
        
        logger.info(f"\n{'='*80}")
        logger.info("BATCH DOWNLOAD SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total tasks: {summary['total_tasks']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Status CSV saved to: {summary['status_csv_path']}")
        
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

