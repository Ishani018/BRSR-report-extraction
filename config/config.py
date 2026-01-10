"""
Configuration settings for the PDF processing pipeline.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# PDF Processing Settings
MIN_TEXT_LENGTH = 100  # Minimum text length to consider PDF as text-based
OCR_DPI = 300  # DPI for OCR processing
MAX_PAGES_PER_BATCH = 50  # Process PDFs in batches to manage memory

# Text Cleaning Settings
MIN_LINE_LENGTH = 3  # Minimum characters in a line to keep
HEADER_FOOTER_THRESHOLD = 0.7  # Similarity threshold for detecting headers/footers

# Section Keywords (for segmentation)
SECTION_KEYWORDS = [
    "table of contents",
    "executive summary",
    "financial statements",
    "balance sheet",
    "income statement",
    "cash flow",
    "notes to accounts",
    "auditor's report",
    "director's report",
    "management discussion",
    "corporate governance",
    "risk management",
]

# Export Settings
EXPORT_FORMATS = ["docx", "csv", "json"]
MAX_TABLE_EXPORT_SIZE = 1000  # Maximum rows per table CSV

# Logging Settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# BRSR-Specific Settings
BRSR_FINANCIAL_YEARS = ['2022-23', '2023-24', '2024-25']

# NSE API Settings
NSE_BASE_URL = "https://www.nseindia.com"
# Using Annual Reports API (stable and official source for BRSR reports)
NSE_API_ENDPOINT = "https://www.nseindia.com/api/annual-reports"
NSE_ARCHIVES_BASE_URL = "https://nsearchives.nseindia.com/annual_reports"
NSE_API_TIMEOUT = 15  # seconds for API call
NSE_DOWNLOAD_TIMEOUT = 30  # seconds for PDF download
NSE_RATE_LIMIT_DELAY = 2.0  # seconds between requests to avoid 403 Forbidden (increased for stability)
NSE_MAX_CONCURRENT = 8  # max concurrent downloads for NSE API

# NSE API Headers (mimic browser to avoid 403 Forbidden)
# Critical: Referer must match the Annual Reports page to avoid 403 errors
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.nseindia.com/companies-listing/corporate-filings-annual-reports',
    'Accept': 'application/json'
}

# Google Search Settings (optional - for Tier 2 fallback)
GOOGLE_SEARCH_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY', '')  # Optional: Google Custom Search API key
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')  # Optional: Custom Search Engine ID
GOOGLE_SEARCH_TIMEOUT = 10  # seconds
GOOGLE_SEARCH_MAX_RESULTS = 10  # max results to check

# Download Settings
DOWNLOAD_BASE_DIR = BASE_DIR.parent / "brsr_reports" / "downloads"
OUTPUT_BASE_DIR = BASE_DIR.parent / "brsr_reports" / "outputs"
STATUS_DIR = BASE_DIR.parent / "brsr_reports" / "status"
MISSING_REPORTS_FILE = STATUS_DIR / "missing_reports.json"
DOWNLOAD_CHECKPOINT_FILE = STATUS_DIR / "download_checkpoint.json"
DOWNLOAD_STATUS_FILE = STATUS_DIR / "download_status.json"

# Create BRSR directories if they don't exist
DOWNLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)

# BRSR Detection Settings
BRSR_STANDALONE_MAX_PAGES = 50  # Documents <= 50 pages are likely standalone BRSR
BRSR_EMBEDDED_MIN_PAGES = 100  # Documents >= 100 pages likely have embedded BRSR
BRSR_CONTENT_THRESHOLD = 0.3  # Minimum fraction of BRSR-related content to classify as BRSR-focused

# BRSR Keywords for detection
BRSR_KEYWORDS = [
    "Business Responsibility and Sustainability Report",
    "BRSR",
    "Business Responsibility Report",
    "BRR",
    "Business Responsibility",
    "Sustainability Report",
    "ESG Report",
    "Corporate Social Responsibility",
    "CSR Report",
    "Sustainability"
]

# File Naming Conventions
BRSR_STANDALONE_PREFIX = "BRSR"
BRSR_EMBEDDED_PREFIX = "AnnualReport"
BRSR_FROM_ANNUAL_SUFFIX = "from_AnnualReport"