import sys
import logging
from pathlib import Path
import pandas as pd

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import BRSR_FINANCIAL_YEARS
from data.company_reader import read_companies
from downloaders.download_manager import DownloadManager
from brsr_main import process_brsr_pdf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VERIFY")

def verify():
    logger.info("Starting verification...")
    
    # 1. Load companies
    base_dir = Path(__file__).parent.parent
    excel_path = base_dir.parent / "NIFTY 500 firms.xlsx"
    if not excel_path.exists():
        logger.error(f"Excel not found: {excel_path}")
        sys.exit(1)
        
    reader = read_companies(excel_path)
    df = reader.get_companies_dataframe()
    if df.empty:
        logger.error("No companies loaded")
        sys.exit(1)
        
    # 2. Pick a known major company (Infosys) for reliability
    # Try to find a company with a symbol
    target_symbol = "INFY" 
    df = df[df['symbol'] == target_symbol]
    
    if df.empty:
         logger.warning(f"Infosys ({target_symbol}) not found, falling back to first company")
         df = reader.get_companies_dataframe()
         df = df[df['symbol'].notna()]
    
    if df.empty:
         logger.error("No companies with symbols found")
         sys.exit(1)
         
    company = df.iloc[0]
    name = company['company_name']
    symbol = company['symbol']
    # Handle serial number safely (convert numpy type to python int if needed)
    serial = company.get('serial_number')
    if pd.notna(serial):
        try:
            serial = int(serial)
        except:
            pass
    else:
        serial = None
            
    naming = company.get('naming_convention')
    if pd.isna(naming): 
        naming = None
        
    logger.info(f"Testing with: {name} ({symbol})")
    
    # 3. Download ONE report
    dm = DownloadManager()
    # Try the most recent year first, or iterate
    year_to_try = BRSR_FINANCIAL_YEARS[1] if len(BRSR_FINANCIAL_YEARS) > 1 else BRSR_FINANCIAL_YEARS[0]
    logger.info(f"Attempting download for year {year_to_try}...")
    
    result = dm.download_single(
        company_name=name,
        symbol=symbol,
        year=year_to_try,
        serial_number=serial,
        naming_convention=naming,
        force_reload=False
    )
    
    if not result['success']:
        logger.error(f"Download failed: {result.get('error')}")
        logger.info("Trying another year...")
        for y in BRSR_FINANCIAL_YEARS:
            if y == year_to_try: continue
            logger.info(f"Trying {y}...")
            result = dm.download_single(
                company_name=name,
                symbol=symbol,
                year=y,
                serial_number=serial,
                naming_convention=naming,
                force_reload=False
            )
            if result['success']:
                year_to_try = y
                break
        
        if not result['success']:
             logger.error("All download attempts failed.")
             sys.exit(1)
        
    pdf_path = Path(result['file_path'])
    logger.info(f"Downloaded: {pdf_path}")
    
    # 4. Process
    logger.info("Processing PDF...")
    proc_result = process_brsr_pdf(
        pdf_path, 
        name, 
        year_to_try, 
        symbol=symbol, 
        serial_number=serial,
        naming_convention=naming
    )
    
    if proc_result['status'] == 'success':
        logger.info("Processing SUCCESS")
        logger.info(f"Output files: {proc_result['output_files']}")
        print("VERIFICATION_SUCCESS")
        sys.exit(0)
    else:
        logger.error(f"Processing FAILED: {proc_result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
