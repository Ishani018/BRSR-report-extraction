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
