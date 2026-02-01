# BRSR Report Extraction and Analysis Pipeline

Automated Python pipeline for extracting, processing, and analyzing Business Responsibility and Sustainability Reporting (BRSR) data from Indian companies. Includes comprehensive topic modeling capabilities for ESG theme discovery.

## Overview

This project provides end-to-end tools for:
- Downloading BRSR reports from the BSE India website
- Extracting structured data from PDF reports
- Performing advanced topic modeling on sustainability disclosures
- Generating interactive visualizations and dashboards
- Exporting results in multiple formats (JSON, CSV, DOCX, HTML)

## Key Features

### PDF Processing
- Automatic PDF type detection (text-based vs scanned)
- OCR support for scanned PDFs using Tesseract
- Smart column detection with layout preservation
- Section extraction (MD&A, Letter to Stakeholders)
- Hierarchical JSON output with nested structure
- Page-by-page organization in DOCX format

### Topic Modeling
- **Multiple Algorithms**: BERTopic, Top2Vec, LDA support
- **Advanced Embeddings**: Sentence-BERT models (all-mpnet-base-v2, all-MiniLM-L6-v2)
- **Enhanced BERTopic Pipeline**:
  - Multiple representation models (MMR, KeyBERT, c-TF-IDF)
  - Outlier reduction for better topic coverage
  - Hierarchical topic reduction
  - Coherence scoring (c_v metric)
  - ESG acronym preservation
- **Optimized Parameters**: Fine-tuned UMAP and HDBSCAN configurations
- **Interactive Visualizations**: pyLDAvis, bar charts, distance maps, topic explorer

### Data Analysis
- 473 BRSR reports processed
- 49,178+ structured paragraphs extracted
- 122 topics discovered in optimized model
- Comprehensive ESG theme coverage (Environmental, Social, Governance)

## Installation

### Prerequisites
- Python 3.12+
- Tesseract OCR (for scanned PDFs)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Ishani018/BRSR-report-extraction.git
cd BRSR-report-extraction
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`
   - Update `TESSERACT_CMD` path in `config.py` if needed

5. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### BRSR Report Download and Processing

#### Download Reports from BSE
```bash
python download_brsr_reports.py
```

Place PDFs in the `brsr_reports/downloads/` folder. The script will:
- Download reports from BSE India
- Extract structured content to JSON
- Track processing status

#### Single PDF Processing
```bash
python test_single.py
```

#### Batch Processing
```bash
python main.py
```

### Topic Modeling

#### Basic BERTopic Analysis
```bash
python esg_bertopic_pipeline.py
```

#### Optimized BERTopic (Recommended)
```bash
python esg_bertopic_optimized.py
```

Features:
- Fine-tuned UMAP parameters (15 components, cosine metric)
- Optimized HDBSCAN (min_cluster_size=50, cluster_selection_epsilon=0.1)
- Readable topic names generated automatically
- Multiple visualizations (bar chart, distance map, category breakdown)

#### Enhanced BERTopic (Advanced)
```bash
# Basic usage with default settings
python esg_bertopic_enhanced.py

# With outlier reduction
python esg_bertopic_enhanced.py --reduce_outliers

# Custom settings
python esg_bertopic_enhanced.py --embedding_model all-MiniLM-L6-v2 --nr_topics 80
```

Advanced features:
- High-quality embeddings (all-mpnet-base-v2, 768 dimensions)
- Multiple representation models for diverse keywords
- Hierarchical topic reduction to target number
- Coherence scoring for quality assessment
- ESG acronym preservation in preprocessing

See [ENHANCED_BERTOPIC_GUIDE.md](ENHANCED_BERTOPIC_GUIDE.md) for detailed documentation.

#### Other Topic Models
```bash
# Top2Vec
python esg_top2vec_pipeline.py

# LDA (Legacy)
python esg_lda_pipeline.py
```

#### Model Comparison
```bash
python compare_topic_models.py
```

### Visualizations

#### Unified Dashboard
Open `bertopic_dashboard.html` in a web browser for:
- All visualizations in one interface
- Interactive topic exploration
- Exploratory themes table (pyLDAvis)
- Topic relationship maps
- ESG category breakdowns

#### Generate Custom Visualizations
```bash
# Readable topic names
python generate_readable_topic_names.py

# Detailed topic viz
python create_detailed_topic_viz.py

# Custom labels
python update_visualization_labels.py
```

## Project Structure

```
project/
├── pipeline/                      # Core processing modules
│   ├── detect_pdf_type.py
│   ├── extract_text.py
│   ├── clean_text.py
│   ├── export_outputs.py
│   ├── section_*.py              # Section extraction modules
│   └── utils.py
├── config/                       # Configuration
│   └── config.py
├── tests/                        # Testing scripts
│   ├── test_single.py
│   ├── test_section_extraction.py
│   └── verify_download_and_process.py
├── brsr_reports/                 # BRSR data
│   ├── downloads/                # Raw PDFs
│   ├── outputs/                  # Extracted JSON
│   └── status/                   # Processing status
├── esg_topics_output/           # Original BERTopic results
├── esg_topics_optimized/        # Optimized BERTopic results
├── esg_topics_enhanced/         # Enhanced BERTopic results
├── esg_bertopic_pipeline.py     # Basic BERTopic
├── esg_bertopic_optimized.py    # Optimized BERTopic
├── esg_bertopic_enhanced.py     # Enhanced BERTopic
├── bertopic_dashboard.html      # Unified visualization dashboard
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Output Files

### PDF Processing Outputs
```
outputs/
  Company Name/
    year/
      report.docx                        # Full report with formatting
      report.json                        # Hierarchical JSON structure
      metadata.json                      # Processing statistics
      sections/
        mdna.docx                        # MD&A section
        mdna.json                        # MD&A hierarchical JSON
        letter_to_stakeholders.docx      # Letter section
        letter_to_stakeholders.json      # Letter hierarchical JSON
        sections_metadata.json           # Section boundaries
```

### Topic Modeling Outputs

#### Optimized Results (esg_topics_optimized/)
- `topic_info.csv` - Detailed topic statistics (122 topics)
- `esg_topic_keywords.csv` - Topic names with keywords
- `document_topics.csv` - Document-to-topic mappings
- `topic_barchart.html` - Interactive bar chart
- `intertopic_distance_map.html` - Topic similarity visualization

#### Enhanced Results (esg_topics_enhanced/)
- All optimized outputs plus:
- `model_metrics.csv` - Coherence scores
- `topic_hierarchy.html` - Hierarchical topic tree
- Enhanced keyword representations (MMR, KeyBERT, c-TF-IDF)

#### Original Results (esg_topics_output/)
- `topics_with_readable_names.csv` - Human-readable topic names
- `all_topics_detailed.html` - Comprehensive topic browser
- `esg_category_breakdown.html` - ESG categorization
- `topic_explorer.html` - Interactive topic explorer

## JSON Structure Example

Hierarchical JSON output with detected headings:

```json
{
  "company": "Company Name",
  "year": "2023",
  "section": "Annual Report",
  "start_page": 1,
  "end_page": 142,
  "structure": [
    {
      "heading": "Environmental Disclosures",
      "level": 2,
      "content": ["paragraph 1...", "paragraph 2..."],
      "subsections": [
        {
          "heading": "Climate Change Mitigation",
          "level": 3,
          "content": ["..."]
        }
      ]
    }
  ],
  "metadata": {
    "total_headings": 241,
    "character_count": 859845,
    "paragraph_count": 532
  }
}
```

## Topic Modeling Results

### Statistics
- **Reports Analyzed**: 473 BRSR reports
- **Documents Generated**: 49,178 paragraphs
- **Topics Discovered**: 122 (optimized model)
- **Outlier Rate**: 15.5%

### Top Topics by Document Count
1. Financial Assets & Liability (3,730 docs)
2. Risk & Opportunity Management (2,825 docs)
3. Waste Management (2,343 docs)
4. Health & Safety (1,840 docs)
5. Human Rights & Grievances (1,378 docs)
6. Employee Turnover (1,113 docs)

### ESG Coverage
- Environmental topics: 35 topics
- Social topics: 48 topics
- Governance topics: 39 topics

## Configuration

Edit `config.py` to customize:
- OCR resolution (DPI)
- Input/output directories
- Tesseract path
- Minimum text length thresholds
- Topic modeling parameters

## Requirements

### Core Dependencies
- pdfplumber - PDF text extraction
- PyMuPDF (fitz) - Fallback PDF processing
- pytesseract - OCR interface
- Pillow - Image processing
- python-docx - DOCX generation
- pandas - Data handling

### Topic Modeling
- bertopic[visualization] - BERTopic with visualizations
- sentence-transformers - Advanced embeddings
- top2vec - Alternative topic modeling
- gensim - LDA and coherence metrics
- pyLDAvis - Interactive visualizations
- spacy - NLP preprocessing

See `requirements.txt` for complete list.

## Advanced Features

### Heading Detection
Deterministic heuristics for hierarchy building:
- Uppercase headings (Level 1)
- Title case headings (Level 2)
- Keyword matching (ESG topics)
- Structural cues (short lines, colons, numeric prefixes)

### Topic Model Optimization
- UMAP dimensionality reduction (15 components)
- HDBSCAN clustering (min_cluster_size=50)
- Custom vectorizer with ESG-specific stop words
- MMR diversity in keyword selection
- KeyBERT context-aware representations

### Quality Metrics
- Coherence scoring (c_v metric)
- Outlier analysis
- Topic distribution statistics
- Representative document samples

## Troubleshooting

### Common Issues

**Tesseract not found:**
- Verify Tesseract installation
- Update `TESSERACT_CMD` in `config.py`

**Memory errors during topic modeling:**
- Use faster embedding model: `--embedding_model all-MiniLM-L6-v2`
- Reduce dataset size or process in batches

**Visualization not loading:**
- Ensure all HTML files are in correct directories
- Open `bertopic_dashboard.html` directly in browser
- Check browser console for errors

## Performance Notes

### Processing Times
- PDF extraction: ~2-5 seconds per report
- Basic BERTopic: ~5-10 minutes for 49K documents
- Enhanced BERTopic: ~8-12 hours (high-quality embeddings)

### Hardware Recommendations
- Minimum: 8GB RAM, 4 CPU cores
- Recommended: 16GB RAM, 8+ CPU cores
- GPU: Optional, speeds up embedding generation

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this project in your research, please cite:

```bibtex
@software{brsr_extraction_2024,
  title={BRSR Report Extraction and Analysis Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/Ishani018/BRSR-report-extraction}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- BERTopic library by Maarten Grootendorst
- Sentence-Transformers by UKPLab
- BSE India for BRSR data availability
