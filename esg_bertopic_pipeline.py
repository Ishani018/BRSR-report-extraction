"""
ESG Topic Modeling Pipeline using BERTopic
-----------------------------------------
This script looks for JSON files in a specified directory, extracts text,
performs strict entity removal (ORG, PERSON, GPE) and year removal,
and then runs BERTopic to find ESG themes.
"""

import sys
import json
import re
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Set
import pandas as pd
import numpy as np

# Ensure we can import from existing utils
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Try existing imports or define fallbacks
print("Importing utils...", flush=True)
try:
    from topic_modeling_utils import split_json_into_documents, load_spacy_model, EXCLUDE_WORDS, nlp as global_nlp, get_custom_stopwords
except ImportError:
    # Fallback if utils not available/compatible
    # We will redefine necessary parts if imports fail, but assuming they work for now
    pass

# Imports for Modeling
print("Importing ML libraries (this may take a while)...", flush=True)
try:
    import spacy
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from hdbscan import HDBSCAN
    from umap import UMAP
except ImportError as e:
    print(f"CRITICAL: Missing dependencies: {e}")
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

# Additional exclusions for boilerplates not covered in utils
EXTRA_STOPWORDS = {
    'ltd', 'limited', 'private', 'plc', 'corp', 'corporation', 'inc', 'incorporated',
    'subsidiary', 'subsidiaries', 'group', 'holding', 'holdings', 'company', 'companies',
    'report', 'annual', 'financial', 'statement', 'fiscal', 'year', 'ended', 'march',
    'date', 'page', 'table', 'figure', 'section', 'annexure', 'appendix',
    'board', 'directors', 'director', 'committee', 'chairman', 'managing', 'executive',
    'shareholder', 'shareholders', 'meeting', 'equity', 'shares',
    'mr', 'mrs', 'ms', 'dr', 'shri', 'smt'
}

# Seed words (ESG concepts)
SEED_TOPICS = [
    ['emission', 'carbon', 'ghg', 'footprint', 'climate'],
    ['waste', 'effluent', 'recycling', 'plastic', 'hazardous'],
    ['water', 'consumption', 'conservation', 'groundwater', 'rainwater'],
    ['energy', 'renewable', 'solar', 'wind', 'electricity'],
    ['social', 'community', 'csr', 'rural', 'development'],
    ['safety', 'health', 'incident', 'injury', 'training'],
    ['governance', 'ethics', 'compliance', 'policy', 'whistleblower'],
    ['diversity', 'inclusion', 'gender', 'women', 'equality']
]
SEED_FLAT = [word for topic in SEED_TOPICS for word in topic]

# --- Preprocessing ---

def strict_esg_preprocess(text: str, nlp_model) -> str:
    """
    Perform strict preprocessing:
    1. Remove 4-digit years.
    2. Remove entities: ORG, PERSON, GPE.
    3. Lemmatize and clean.
    """
    if not text:
        return ""

    # 1. Regex Cleaning
    # Remove dates/years (e.g., 2023, 2024, FY23, FY2023)
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)  # 1900-2099
    text = re.sub(r'\bFY\d{2,4}\b', '', text, flags=re.IGNORECASE)
    
    # Remove emails/URLs
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Spacy Processing
    doc = nlp_model(text)
    
    kept_tokens = []
    
    # Entities to remove (exact types requested)
    # Using 'ent_type_' check. Note: ent_type_ is empty string if no entity.
    # We iterate tokens. If token is part of an entity, we check the label.
    
    # Optimization: iterate and check if token is inside a banned entity
    # OR just use token.ent_type_
    
    BANNED_ENTS = {'ORG', 'PERSON', 'GPE'}
    
    for token in doc:
        # Entity Filter
        if token.ent_type_ in BANNED_ENTS:
            continue
            
        # Stopword Filter (standard + spacy default)
        if token.is_stop:
            continue
            
        # POS Filter (Keep content words)
        # Keeping NOUN, ADJ, VERB (actions!), PROPN (careful here)
        # We wanted "topics about actions", so VERB is important.
        # But we also need nouns (waste, emission).
        # Since we strictly removed ORG/PERSON/GPE, remaining PROPNs might be products or others,
        # but often cleaner to just stick to NOUN, ADJ, VERB.
        if token.pos_ not in ['NOUN', 'ADJ', 'VERB']:
            continue
            
        # Length check
        if len(token.text) < 3:
            continue
            
        # Check against custom extra stopwords
        lemma = token.lemma_.lower()
        if lemma in EXTRA_STOPWORDS or token.text.lower() in EXTRA_STOPWORDS:
            continue
        
        # Add lemma
        kept_tokens.append(lemma)
        
    return " ".join(kept_tokens)

# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="Run ESG BERTopic Pipeline")
    parser.add_argument('--input_dir', type=str, default='brsr_reports/outputs', help='Directory containing JSON reports')
    parser.add_argument('--output_dir', type=str, default='esg_topics_output', help='Directory for results')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of files to process (for testing)')
    parser.add_argument('--min_chunk_len', type=int, default=150, help='Minimum char length for a text chunk')
    
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
    all_documents = [] # List of strings (processed text)
    doc_metadata = [] # List of dicts (source file, original text length)
    
    for i, jf in enumerate(json_files):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Use utils function if available, else implement simple fallback
            # We want smaller chunks for better topic resolution
            # Strategy: 'paragraphs' usually works best for fine-grained topics
            chunks = split_json_into_documents(data, strategy='paragraphs', min_length=args.min_chunk_len)
            
            # Preprocess chunks
            processed_count = 0
            for chunk in chunks:
                raw_text = chunk['text']
                processed_text = strict_esg_preprocess(raw_text, nlp)
                
                # Filter out empty or too short results
                if len(processed_text.split()) >= 5: # At least 5 words remaining
                    all_documents.append(processed_text)
                    doc_metadata.append({
                        'source_file': jf.name,
                        'raw_preview': raw_text[:100] + "...",
                        'section_path': chunk.get('section_path', ''),
                        'processed_text': processed_text # Store for inspection if needed
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

    # 4. Configure Vectorizer (The Filter)
    # max_df=0.6, min_df=0.02, ngram_range=(1, 2)
    vectorizer_model = CountVectorizer(
        min_df=0.02,
        max_df=0.6,
        ngram_range=(1, 2),
        stop_words='english' # We already removed most, but a second pass doesn't hurt
    )
    
    # 5. Initialize BERTopic
    # Optimized for speed and quality on this specific task
    
    # UMAP: Random state for reproducibility
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    logger.info("Initializing BERTopic...")
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        nr_topics="auto", # Auto-reduce
        seed_topic_list=SEED_TOPICS,
        verbose=True,
        calculate_probabilities=False # Save memory
    )
    
    # 6. Fit Model
    logger.info("Fitting model...")
    topics, probs = topic_model.fit_transform(all_documents)
    
    # 7. Export Results
    logger.info("Exporting results...")
    
    # A. Topic Info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_path / "topic_info.csv", index=False)
    
    # B. Top Keywords per Topic (CSV)
    # Custom format: TopicID, Name, Top 10 Words
    keywords_data = []
    for topic_id in sorted(list(set(topics))):
        if topic_id == -1: continue # Outliers
        
        words_scores = topic_model.get_topic(topic_id)
        if not words_scores: continue
        
        top_words = [w for w, s in words_scores[:10]]
        keywords_data.append({
            'Topic': topic_id,
            'Count': topic_info[topic_info['Topic'] == topic_id]['Count'].values[0] if not topic_info.empty else 0,
            'Name': topic_info[topic_info['Topic'] == topic_id]['Name'].values[0] if not topic_info.empty else f"Topic {topic_id}",
            'Top_Keywords': ", ".join(top_words)
        })
        
    pd.DataFrame(keywords_data).to_csv(output_path / "esg_topic_keywords.csv", index=False)
    
    # C. Document Mapping (Assign topics back to metadata)
    doc_df = pd.DataFrame(doc_metadata)
    doc_df['Topic'] = topics
    # Add mapped topic name
    topic_names_map = {row['Topic']: row['Name'] for row in keywords_data}
    doc_df['Topic_Name'] = doc_df['Topic'].map(topic_names_map).fillna("Outlier")
    
    doc_df.to_csv(output_path / "document_topics.csv", index=False)
    
    # D. Visualizations
    try:
        vis_topics = topic_model.visualize_topics()
        vis_topics.write_html(str(output_path / "intertopic_distance_map.html"))
        
        vis_bar = topic_model.visualize_barchart(top_n_topics=20)
        vis_bar.write_html(str(output_path / "topic_barchart.html"))
        logger.info("Visualizations saved.")
    except Exception as e:
        logger.warning(f"Could not generate visualisations: {e}")

    logger.info("Done! Check output directory.")

if __name__ == "__main__":
    main()
