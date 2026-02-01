# Enhanced BERTopic Pipeline - Usage Guide

## Overview

The enhanced pipeline (`esg_bertopic_enhanced.py`) includes advanced features for better topic quality:

✅ **Advanced Embeddings**: Uses `all-mpnet-base-v2` (better quality than default)  
✅ **Multiple Representations**: MMR + KeyBERT for diverse, meaningful keywords  
✅ **Outlier Reduction**: Assigns outliers to nearest topics  
✅ **Hierarchical Reduction**: Smart topic merging based on similarity  
✅ **Quality Metrics**: Coherence scoring to measure topic quality  
✅ **ESG Acronym Preservation**: Keeps important acronyms (ESG, GHG, CO2, etc.)  

## Installation

Install additional dependencies:

```bash
pip install sentence-transformers gensim
```

Or update all dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage

### Run with Default Settings (Recommended)

```bash
python esg_bertopic_enhanced.py
```

This will:
- Use `all-mpnet-base-v2` embeddings
- Target 80 topics
- Use min_cluster_size of 50
- Save results to `esg_topics_enhanced/`

### Advanced Options

```bash
# Use faster embedding model
python esg_bertopic_enhanced.py --embedding_model all-MiniLM-L6-v2

# Reduce outliers aggressively
python esg_bertopic_enhanced.py --reduce_outliers

# Control number of topics
python esg_bertopic_enhanced.py --target_topics 60

# Adjust cluster size (higher = fewer, denser topics)
python esg_bertopic_enhanced.py --min_cluster_size 70

# Combine multiple options
python esg_bertopic_enhanced.py --reduce_outliers --target_topics 50 --min_cluster_size 60
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | `brsr_reports/outputs` | Directory with JSON reports |
| `--output_dir` | `esg_topics_enhanced` | Output directory |
| `--limit` | `0` | Limit files (0 = all) |
| `--min_chunk_len` | `150` | Minimum chunk length |
| `--min_cluster_size` | `50` | HDBSCAN min cluster size |
| `--target_topics` | `80` | Target number of topics |
| `--embedding_model` | `all-mpnet-base-v2` | Sentence transformer model |
| `--reduce_outliers` | `False` | Enable outlier reduction |

## Output Files

The enhanced pipeline creates:

### Core Results
- **`topic_info.csv`** - Topic statistics and info
- **`esg_topic_keywords_enhanced.csv`** - Keywords from 3 methods (Default, MMR, KeyBERT)
- **`document_topics.csv`** - Document-to-topic mapping
- **`model_metrics.csv`** - Quality metrics including coherence score
- **`bertopic_model/`** - Saved model for later use

### Visualizations
- **`topic_barchart.html`** - Topic distribution
- **`intertopic_distance_map.html`** - Topic relationships
- **`topic_hierarchy.html`** - Hierarchical topic structure (NEW!)

## Understanding the Enhancements

### 1. Better Embeddings

**Old:** Default BERT embeddings  
**New:** `all-mpnet-base-v2` - state-of-the-art sentence embeddings

**Impact:** Better semantic understanding, more meaningful topics

### 2. Multiple Representation Models

**MMR (Maximal Marginal Relevance):**
- Ensures keyword diversity
- Reduces redundant keywords
- Parameters: `diversity=0.3`

**KeyBERT:**
- Better keyword extraction
- More representative topic words

**Result:** Three keyword sets per topic for comparison

### 3. Outlier Reduction

**Strategy:** `distributions`  
**Threshold:** `0.1`

**Effect:** Assigns outlier documents to nearest topics based on probability distribution

### 4. Hierarchical Topic Reduction

Instead of forcing topics during training, the pipeline:
1. Discovers topics naturally
2. Reduces hierarchically based on similarity
3. Preserves topic quality better than forced reduction

### 5. Coherence Scoring

**Metric:** Gensim's c_v coherence  
**Scale:** 0-1 (higher is better)  
**Typical range:** 0.3-0.6 for good models

**Use:** Compare different runs to see which parameters work best

## Comparison with Previous Versions

| Feature | Original | Optimized | Enhanced |
|---------|----------|-----------|----------|
| Embeddings | Default | Default | all-mpnet-base-v2 |
| Representation | c-TF-IDF | c-TF-IDF | MMR + KeyBERT + c-TF-IDF |
| Outlier Handling | None | None | Optional reduction |
| Topic Reduction | Forced | Forced (120) | Hierarchical (80) |
| Quality Metrics | None | None | Coherence scoring |
| Hierarchy Viz | No | No | Yes |
| ESG Acronyms | Removed | Partially kept | Preserved |

## Tips for Best Results

### 1. Finding Optimal Cluster Size

Start with default (50), then experiment:
- **Too few topics?** Decrease `min_cluster_size` to 30-40
- **Too many topics?** Increase `min_cluster_size` to 60-80
- **Too many outliers?** Enable `--reduce_outliers`

### 2. Choosing Target Topics

- **Broad overview:** 40-60 topics
- **Balanced:** 80-100 topics (default)
- **Detailed analysis:** 120-150 topics

### 3. Embedding Model Selection

```bash
# Best quality (slower, ~420MB)
--embedding_model all-mpnet-base-v2

# Balanced (faster, ~80MB)
--embedding_model all-MiniLM-L6-v2

# Fastest (very fast, ~90MB)
--embedding_model paraphrase-MiniLM-L3-v2
```

### 4. Iterating on Results

1. Run with defaults first
2. Check coherence score in `model_metrics.csv`
3. Adjust parameters based on:
   - Number of outliers
   - Topic quality (manual inspection)
   - Coherence score
4. Re-run and compare metrics

## Example Workflow

```bash
# Step 1: Initial run with defaults
python esg_bertopic_enhanced.py

# Step 2: Check results
# Open: esg_topics_enhanced/model_metrics.csv
# Review: Coherence score, outlier rate

# Step 3: Fine-tune based on results
# If too many outliers:
python esg_bertopic_enhanced.py --reduce_outliers --min_cluster_size 40

# If topics are too granular:
python esg_bertopic_enhanced.py --target_topics 60 --min_cluster_size 60

# If need faster processing:
python esg_bertopic_enhanced.py --embedding_model all-MiniLM-L6-v2
```

## Troubleshooting

### "Out of Memory" Error
- Use lighter embedding model: `--embedding_model all-MiniLM-L6-v2`
- Limit files during testing: `--limit 50`

### Too Many Outliers
- Enable outlier reduction: `--reduce_outliers`
- Decrease cluster size: `--min_cluster_size 30`

### Low Coherence Score (< 0.3)
- Increase target topics: `--target_topics 100`
- Try different embedding model
- Check preprocessing (might be too aggressive)

### Topics Too Similar
- Increase `min_cluster_size`
- Decrease `target_topics`

## Next Steps

After running the enhanced pipeline:

1. **Compare Results**: Use your dashboard to compare with previous runs
2. **Analyze Keywords**: Check `esg_topic_keywords_enhanced.csv` - compare MMR vs KeyBERT
3. **Review Hierarchy**: Open `topic_hierarchy.html` to see topic relationships
4. **Iterate**: Adjust parameters based on coherence score and manual inspection

## Questions?

Common questions:

**Q: Should I always use the enhanced version?**  
A: Yes, if you have time. It takes longer but produces better quality topics.

**Q: What's a good coherence score?**  
A: 0.4+ is good, 0.5+ is excellent for this domain.

**Q: Can I use my own embedding model?**  
A: Yes! Any sentence-transformers model from HuggingFace works.

**Q: How much slower is it?**  
A: ~30-50% slower due to better embedding model, but worth it for quality.
