# How to Test Topic Modeling on One JSON File

## Step 1: Install Required Packages

First, install the topic modeling packages:

```powershell
cd pdf-to-structured-reports
pip install fastopic bertopic[visualization] gensim pyLDAvis
```

Or install all requirements:

```powershell
pip install -r requirements.txt
```

## Step 2: Pick a JSON File

You have many JSON files in `brsr_reports/outputs/`. Pick any one to test. For example:

- `brsr_reports/outputs/102_Castrol India Ltd._2023_2024.json`
- `brsr_reports/outputs/113_Cipla Ltd._2023_2024.json`
- `brsr_reports/outputs/203_Hero MotoCorp Ltd._2023_2024.json`

Or any other JSON file from the outputs folder.

## Step 3: Run Each Script

Navigate to the `pdf-to-structured-reports` directory and run each script:

### Test FASTopic:
```powershell
python topic_modeling_fastopic.py "brsr_reports/outputs/102_Castrol India Ltd._2023_2024.json" --num_topics 5
```

### Test BERTopic:
```powershell
python topic_modeling_bertopic.py "brsr_reports/outputs/102_Castrol India Ltd._2023_2024.json" --num_topics 5
```

### Test LDA:
```powershell
python topic_modeling_lda.py "brsr_reports/outputs/102_Castrol India Ltd._2023_2024.json" --num_topics 5
```

## Step 4: Check Results

After running each script, check the results in:

```
brsr_reports/topic_modeling_results/
  ├── fastopic/
  │   └── 102_Castrol India Ltd._2023_2024/
  │       ├── section_topics.csv       # Which sections belong to which topics
  │       ├── topic_keywords.csv       # Top keywords for each topic
  │       ├── visualization.html       # Interactive visualization
  │       └── summary.txt              # Model summary
  │
  ├── bertopic/
  │   └── 102_Castrol India Ltd._2023_2024/
  │       ├── section_topics.csv
  │       ├── topic_keywords.csv
  │       ├── visualization.html       # Combined visualization page
  │       ├── visualization_intertopic.html  # Intertopic distance map
  │       ├── visualization_barchart.html    # Topic bar chart
  │       └── summary.txt
  │
  └── lda/
      └── 102_Castrol India Ltd._2023_2024/
          ├── section_topics.csv
          ├── topic_keywords.csv
          ├── visualization.html       # pyLDAvis interactive visualization
          └── summary.txt
```

## Step 5: Compare Results

1. **Open the visualization HTML files** in your browser to see topic visualizations
2. **Check the CSV files** to see which sections belong to which topics
3. **Compare topic keywords** across the three methods
4. **Read the summary.txt** files to see model statistics

## Command-Line Options

All scripts support these options:

- `--num_topics`: Number of topics to extract (default: 5)
- `--strategy`: How to split documents - `sections`, `paragraphs`, or `both` (default: sections)
- `--min_length`: Minimum document length in characters (default: 100)
- `--output_dir`: Custom output directory (optional)

### Example with custom options:

```powershell
python topic_modeling_bertopic.py "brsr_reports/outputs/102_Castrol India Ltd._2023_2024.json" --num_topics 10 --strategy paragraphs --min_length 200
```

## Quick Test Example

Here's a quick test you can run right now (using PowerShell):

```powershell
# Navigate to project directory
cd "pdf-to-structured-reports"

# Test FASTopic (replace with your actual JSON file path)
python topic_modeling_fastopic.py "brsr_reports/outputs/102_Castrol India Ltd._2023_2024.json" --num_topics 5
```

If the script runs successfully, you should see:
- Progress messages in the terminal
- A message saying "Topic modeling complete! Results saved to: ..."
- Results folder created with CSV and HTML files

## Troubleshooting

### If you get "module not found" errors:
```powershell
pip install fastopic bertopic[visualization] gensim pyLDAvis
```

### If the JSON file is not found:
Make sure you're using the correct path. Use quotes if the path has spaces:
```powershell
python topic_modeling_fastopic.py "brsr_reports/outputs/102_Castrol India Ltd._2023_2024.json"
```

### If you get "not enough documents" error:
Try a different JSON file or reduce `--num_topics`:
```powershell
python topic_modeling_fastopic.py "your_file.json" --num_topics 3
```
