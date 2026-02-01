"""
ESG Topic Modeling Pipeline using Top2Vec
------------------------------------------
Alternative to BERTopic using Top2Vec for comparison.
Top2Vec creates a joint document-topic-word embedding that may better handle
diverse topics and reduce outliers.
"""

import sys
import json
import re
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Ensure we can import from existing utils
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

print("Importing utils...", flush=True)
try:
    from topic_modeling_utils import split_json_into_documents
except ImportError:
    pass

# Imports for Modeling
print("Importing ML libraries (this may take a while)...", flush=True)
try:
    import spacy
    from top2vec import Top2Vec
except ImportError as e:
    print(f"CRITICAL: Missing dependencies: {e}")
    print("Please install top2vec: pip install top2vec")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration ---

EXTRA_STOPWORDS = {
    'ltd', 'limited', 'private', 'plc', 'corp', 'corporation', 'inc', 'incorporated',
    'subsidiary', 'subsidiaries', 'group', 'holding', 'holdings', 'company', 'companies',
    'report', 'annual', 'financial', 'statement', 'fiscal', 'year', 'ended', 'march',
    'date', 'page', 'table', 'figure', 'section', 'annexure', 'appendix',
    'board', 'directors', 'director', 'committee', 'chairman', 'managing', 'executive',
    'shareholder', 'shareholders', 'meeting', 'equity', 'shares',
    'mr', 'mrs', 'ms', 'dr', 'shri', 'smt'
}

# --- Preprocessing ---

def strict_esg_preprocess(text: str, nlp_model) -> str:
    """
    Perform strict preprocessing:
    1. Fix text encoding issues (reversed text)
    2. Remove 4-digit years
    3. Remove entities: ORG, PERSON, GPE
    4. Lemmatize and clean
    """
    if not text:
        return ""

    # 0. Fix Text Encoding Issues
    reversed_patterns = ['htiw', 'ruo', 'srekrow', 'seeyolpme', 'eht']
    if any(pattern in text.lower() for pattern in reversed_patterns):
        reversed_count = sum(1 for p in reversed_patterns if p in text.lower())
        normal_count = sum(1 for p in ['with', 'our', 'workers', 'employees', 'the'] if p in text.lower())
        if reversed_count > normal_count:
            text = text[::-1]
    
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 1. Regex Cleaning
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)
    text = re.sub(r'\bFY\d{2,4}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Spacy Processing
    doc = nlp_model(text)
    
    kept_tokens = []
    BANNED_ENTS = {'ORG', 'PERSON', 'GPE'}
    
    for token in doc:
        if token.ent_type_ in BANNED_ENTS:
            continue
        if token.is_stop:
            continue
        if token.pos_ not in ['NOUN', 'ADJ', 'VERB']:
            continue
        if len(token.text) < 3:
            continue
            
        lemma = token.lemma_.lower()
        if lemma in EXTRA_STOPWORDS or token.text.lower() in EXTRA_STOPWORDS:
            continue
        
        kept_tokens.append(lemma)
        
    return " ".join(kept_tokens)

# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="Run ESG Top2Vec Pipeline")
    parser.add_argument('--input_dir', type=str, default='brsr_reports/outputs', help='Directory containing JSON reports')
    parser.add_argument('--output_dir', type=str, default='esg_topics_top2vec_optimized', help='Directory for results')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of files to process (for testing)')
    parser.add_argument('--min_chunk_len', type=int, default=150, help='Minimum char length for a text chunk')
    parser.add_argument('--min_count', type=int, default=100, help='Minimum document count for a topic (optimized for ~120 topics)')
    parser.add_argument('--speed', type=str, default='deep-learn', choices=['fast-learn', 'learn', 'deep-learn'], 
                        help='Speed mode: fast-learn, learn, or deep-learn (more accurate)')
    
    args = parser.parse_args()
    
    input_path = BASE_DIR / args.input_dir
    output_path = BASE_DIR / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Spacy
    logger.info("Loading Spacy model 'en_core_web_sm'...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error("Please download spacy model: python -m spacy download en_core_web_sm")
        sys.exit(1)
        
    # 2. Gather Files
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        sys.exit(1)
        
    json_files = list(input_path.glob("*.json"))
    if args.limit > 0:
        json_files = json_files[:args.limit]
        
    logger.info(f"Found {len(json_files)} JSON files to process.")
    
    # 3. Data Loading & Splitting
    all_documents = []
    doc_metadata = []
    
    for i, jf in enumerate(json_files):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = split_json_into_documents(data, strategy='paragraphs', min_length=args.min_chunk_len)
            
            processed_count = 0
            for chunk in chunks:
                raw_text = chunk['text']
                processed_text = strict_esg_preprocess(raw_text, nlp)
                
                if len(processed_text.split()) >= 5:
                    all_documents.append(processed_text)
                    doc_metadata.append({
                        'source_file': jf.name,
                        'raw_preview': raw_text[:100] + "...",
                        'section_path': chunk.get('section_path', ''),
                        'processed_text': processed_text
                    })
                    processed_count += 1
            
            if (i+1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(json_files)} files...")
                
        except Exception as e:
            logger.warning(f"Failed to process {jf.name}: {e}")
            
    logger.info(f"Total valid documents (chunks) after preprocessing: {len(all_documents)}")
    
    if len(all_documents) < 10:
        logger.error("Not enough documents to model. Check your data or filters.")
        sys.exit(1)

    # 4. Initialize and Train Top2Vec
    logger.info(f"Initializing Top2Vec (speed={args.speed}, min_count={args.min_count})...")
    logger.info("This may take 20-30 minutes depending on corpus size...")
    
    try:
        model = Top2Vec(
            documents=all_documents,
            speed=args.speed,
            workers=4,
            embedding_model='doc2vec',  # Use doc2vec instead (no TensorFlow needed)
            min_count=args.min_count,
            verbose=True
        )
        logger.info("Top2Vec training complete!")
    except Exception as e:
        logger.error(f"Top2Vec training failed: {e}")
        sys.exit(1)
    
    # 5. Export Results
    logger.info("Exporting results...")
    
    # Get topic information
    topic_sizes, topic_nums = model.get_topic_sizes()
    num_topics = len(topic_sizes)
    
    logger.info(f"Discovered {num_topics} topics")
    
    # A. Topic Keywords CSV
    keywords_data = []
    for topic_num in topic_nums:
        topic_words, word_scores, topic_nums_out = model.get_topics(topic_num)
        
        # Skip topics with no keywords
        if len(topic_words) == 0:
            continue
            
        # Create topic name safely
        name_parts = [str(topic_num)]
        for i in range(min(3, len(topic_words))):
            name_parts.append(topic_words[i])
        topic_name = "_".join(name_parts)
        
        keywords_data.append({
            'Topic': int(topic_num),
            'Count': int(topic_sizes[topic_nums.tolist().index(topic_num)]),
            'Top_Keywords': ", ".join(topic_words[:10]),
            'Name': topic_name
        })
    
    keywords_df = pd.DataFrame(keywords_data)
    keywords_df = keywords_df.sort_values('Count', ascending=False)
    keywords_df.to_csv(output_path / "topic_keywords.csv", index=False)
    
    # B. Document-Topic Mapping
    doc_topics, doc_dists, doc_nums = model.get_documents_topics(list(range(len(all_documents))))
    
    doc_df = pd.DataFrame(doc_metadata)
    doc_df['Topic'] = doc_topics
    doc_df['Topic_Distance'] = doc_dists
    
    # Map topic names
    topic_names_map = {row['Topic']: row['Name'] for _, row in keywords_df.iterrows()}
    doc_df['Topic_Name'] = doc_df['Topic'].map(topic_names_map)
    
    doc_df.to_csv(output_path / "document_topics.csv", index=False)
    
    # C. Save Model
    logger.info("Saving Top2Vec model...")
    model.save(str(output_path / "top2vec_model"))
    
    # D. Additional Stats
    outlier_count = 0  # Top2Vec doesn't have outliers in the same way
    stats = {
        'total_documents': len(all_documents),
        'num_topics': num_topics,
        'outlier_percentage': 0.0,  # Top2Vec assigns all docs
        'avg_topic_size': np.mean(topic_sizes),
        'median_topic_size': np.median(topic_sizes),
        'largest_topic_size': np.max(topic_sizes),
        'smallest_topic_size': np.min(topic_sizes)
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_path / "model_stats.csv", index=False)
    
    logger.info("="*80)
    logger.info("Top2Vec Results Summary:")
    logger.info(f"  Total Documents: {stats['total_documents']}")
    logger.info(f"  Topics Discovered: {stats['num_topics']}")
    logger.info(f"  Avg Topic Size: {stats['avg_topic_size']:.1f}")
    logger.info(f"  Largest Topic: {stats['largest_topic_size']}")
    logger.info("="*80)
    
    logger.info(f"Done! Check output directory: {output_path}")

if __name__ == "__main__":
    main()
