# PDF to Machine-Readable Format Pipeline

Automated Python pipeline for converting large annual report PDFs into machine-readable formats for NLP and data analysis.

## Features

- **PDF Type Detection**: Automatically detects text-based vs scanned PDFs
- **Text Extraction**: Page-wise text extraction with OCR support
- **Table Extraction**: Extracts financial tables using multiple methods
- **Text Cleaning**: Removes headers, footers, and formatting noise
- **Section Segmentation**: Intelligently segments content into logical sections
- **Multiple Export Formats**: DOCX, CSV, and JSON outputs
- **Fault Tolerant**: Continues processing even if individual PDFs fail
- **Batch Processing**: Efficiently processes hundreds of PDFs

## Architecture

The pipeline follows a modular design with separated concerns:

```
├── main.py                 # Main orchestrator
├── config.py               # Configuration settings
├── detect_pdf_type.py      # PDF type detection
├── extract_text.py         # Text extraction (direct + OCR)
├── extract_tables.py       # Table extraction
├── clean_text.py           # Text cleaning and normalization
├── segment_sections.py     # Section segmentation
├── export_outputs.py       # Export to various formats
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Requirements

- Python 3.8+
- Tesseract OCR (for scanned PDFs)

### Installing Tesseract

**Windows:**

```powershell
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Or install via chocolatey:
choco install tesseract
```

**Linux:**

```bash
sudo apt-get install tesseract-ocr
```

**macOS:**

```bash
brew install tesseract
```

## Installation

1. Clone or download this project

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Verify Tesseract installation:

```powershell
tesseract --version
```

## Usage

### Directory Structure

Place your PDF files in the `data` directory with the following structure:

```
data/
├── CompanyName1/
│   ├── 2020/
│   │   └── report.pdf
│   ├── 2021/
│   │   └── report.pdf
│   └── 2022/
│       └── report.pdf
└── CompanyName2/
    └── report_2023.pdf
```

The pipeline will automatically detect company names and years from the directory structure or filenames.

### Running the Pipeline

```powershell
python main.py
```

The pipeline will:

1. Scan the `data` directory for all PDFs
2. Process each PDF through all stages
3. Export results to the `outputs` directory
4. Generate a processing summary

### Output Structure

```
outputs/
├── CompanyName/
│   └── 2022/
│       ├── report.docx          # Full text with sections
│       ├── tables/
│       │   ├── table_1_page_5.csv
│       │   └── table_2_page_12.csv
│       └── metadata.json        # Processing metadata
└── processing_summary.json      # Overall summary
```

## Configuration

Edit `config.py` to customize:

- **Processing Settings**: OCR DPI, batch sizes, text thresholds
- **Cleaning Parameters**: Header/footer detection, minimum line lengths
- **Section Keywords**: Keywords for section detection
- **Export Settings**: Output formats, table size limits

## Modules

### detect_pdf_type.py

- Analyzes PDFs to determine if they contain extractable text
- Returns metadata about page count and text density

### extract_text.py

- Extracts text from text-based PDFs using pdfplumber/PyMuPDF
- Performs OCR on scanned PDFs using Tesseract
- Preserves page numbers as metadata

### extract_tables.py

- Attempts vector-based extraction using Camelot
- Falls back to Tabula if needed
- Identifies financial table types (balance sheet, income statement, etc.)

### clean_text.py

- Removes repeated headers and footers
- Fixes broken lines from PDF extraction
- Normalizes whitespace and Unicode characters
- Removes formatting artifacts

### segment_sections.py

- Detects section headings using multiple strategies
- Attempts to extract table of contents
- Creates hierarchical section structure
- Uses keyword matching for common report sections

### export_outputs.py

- Exports to DOCX with preserved heading structure
- Saves tables as individual CSV files
- Generates JSON metadata with statistics

## Error Handling

The pipeline is designed to be fault-tolerant:

- Individual PDF failures don't stop the entire batch
- Comprehensive logging tracks all errors
- Processing continues with next file if one fails
- Final summary reports all successes and failures

## Logging

Logs are saved to the `logs` directory with timestamps. Each run generates:

- Console output with progress bars
- Detailed log file with debug information
- Processing summary in JSON format

## Performance Tips

1. **Batch Size**: Adjust `MAX_PAGES_PER_BATCH` in config.py for memory management
2. **OCR DPI**: Lower DPI (150-200) for faster processing, higher (300+) for better accuracy
3. **Parallel Processing**: For large datasets, consider running multiple instances on different subdirectories

## Troubleshooting

**Issue**: "Tesseract not found"

- **Solution**: Install Tesseract and add to system PATH

**Issue**: "Memory error during OCR"

- **Solution**: Reduce `OCR_DPI` or `MAX_PAGES_PER_BATCH` in config.py

**Issue**: "No tables extracted"

- **Solution**: Tables in scanned PDFs or image-based tables may not extract well. Try improving PDF quality.

**Issue**: "Encoding errors in output"

- **Solution**: The pipeline uses UTF-8. Ensure your text editor supports UTF-8.

## Dependencies

Core libraries:

- `pdfplumber` - Text-based PDF extraction
- `PyMuPDF` - Alternative PDF processing
- `pytesseract` - OCR wrapper for Tesseract
- `camelot-py` - Table extraction (lattice/stream)
- `tabula-py` - Alternative table extraction
- `python-docx` - DOCX generation
- `pandas` - Table manipulation
- `tqdm` - Progress bars

## License

This project is provided as-is for educational and research purposes.

## Contributing

To extend this pipeline:

1. Follow the modular architecture
2. Add type hints and docstrings
3. Include comprehensive logging
4. Handle errors gracefully
5. Write clean, readable code

## Future Enhancements

Potential improvements:

- Multi-processing for parallel PDF processing
- Advanced section detection using ML models
- Better table structure recognition
- Support for multi-column layouts
- Database storage instead of files
- Web interface for monitoring
