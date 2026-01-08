# PDF to DOCX Converter

Automated Python pipeline to convert annual report PDFs into clean, readable DOCX files with proper layout preservation.

## Features

- **Automatic PDF Type Detection**: Distinguishes between text-based and scanned PDFs
- **OCR Support**: Automatically applies OCR to scanned PDFs using Tesseract
- **Smart Column Detection**: Dynamically handles any number of pages per PDF page (1, 2, 3+), each with their own column structure
- **Layout Preservation**: Maintains proper reading order (left-to-right, top-to-bottom)
- **Page-by-Page Organization**: Structured DOCX output with clear page separators
- **Batch Processing**: Process multiple companies and years efficiently

## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR (for scanned PDFs)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/srinix18/pdf-to-structured-reports.git
cd pdf-to-structured-reports
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Update `TESSERACT_CMD` path in `config.py` if needed

## Usage

### Single PDF Processing

```bash
python test_single.py
```

### Batch Processing

```bash
python main.py
```

Place PDFs in the `data/` folder with structure:
```
data/
  Company Name/
    company_name_year.pdf
```

## Output

Generated files in `outputs/` folder:
```
outputs/
  Company Name/
    year/
      report.docx       # Clean, formatted document
      metadata.json     # Processing statistics
```

## How It Works

1. **PDF Detection**: Identifies if PDF contains text or requires OCR
2. **Text Extraction**: Extracts text with smart column detection
   - Detects multiple pages per PDF page (common in annual reports)
   - Identifies column structure within each page
   - Maintains proper reading order
3. **Text Cleaning**: Conservative cleaning that preserves layout
4. **DOCX Export**: Generates structured document with page-by-page organization

## Configuration

Edit `config.py` to customize:
- OCR resolution (DPI)
- Input/output directories
- Tesseract path
- Minimum text length thresholds

## Architecture

- `main.py` - Main orchestrator for batch processing
- `test_single.py` - Test single PDF processing
- `detect_pdf_type.py` - PDF type detection (text vs scanned)
- `extract_text.py` - Text extraction with column detection
- `clean_text.py` - Text cleaning while preserving layout
- `export_outputs.py` - DOCX file generation
- `config.py` - Configuration settings
- `utils.py` - Helper functions

## Requirements

- pdfplumber - PDF text extraction
- PyMuPDF (fitz) - Fallback PDF processing
- pytesseract - OCR interface
- Pillow - Image processing for OCR
- python-docx - DOCX file generation
- pandas - Data handling

## License

MIT
