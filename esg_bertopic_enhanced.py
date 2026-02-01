"""
ESG Topic Modeling Pipeline using BERTopic - ENHANCED VERSION
-------------------------------------------------------------
This enhanced version includes:
- Advanced sentence embeddings (all-mpnet-base-v2)
- Multiple representation models (MMR + KeyBERT)
- Outlier reduction strategies
- Topic coherence scoring
- Hierarchical topic reduction
- ESG-specific acronym preservation
"""

import sys
import json
import re
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import pandas as pd
import numpy as np

# Ensure we can import from existing utils
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Try existing imports
print("Importing utils...", flush=True)
try:
    from topic_modeling_utils import split_json_into_documents, load_spacy_model, EXCLUDE_WORDS, nlp as global_nlp, get_custom_stopwords
except ImportError:
    pass

# Imports for Modeling
print("Importing ML libraries (this may take a while)...", flush=True)
try:
    import spacy
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from hdbscan import HDBSCAN
    from umap import UMAP
except ImportError as e:
    print(f"CRITICAL: Missing dependencies: {e}")
    print("Install with: pip install bertopic sentence-transformers")
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

# Additional exclusions for boilerplates
EXTRA_STOPWORDS = {
    'ltd', 'limited', 'private', 'plc', 'corp', 'corporation', 'inc', 'incorporated',
    'subsidiary', 'subsidiaries', 'group', 'holding', 'holdings', 'company', 'companies',
    'report', 'annual', 'financial', 'statement', 'fiscal', 'year', 'ended', 'march',
    'date', 'page', 'table', 'figure', 'section', 'annexure', 'appendix',
    'board', 'directors', 'director', 'committee', 'chairman', 'managing', 'executive',
    'shareholder', 'shareholders', 'meeting', 'equity', 'shares',
    'mr', 'mrs', 'ms', 'dr', 'shri', 'smt'
}

# ESG-specific acronyms to preserve
ESG_ACRONYMS = ['ESG', 'GHG', 'CO2', 'CSR', 'SDG', 'UN', 'NGO', 'BRSR', 'TCFD', 'CDP']

# --- Enhanced Preprocessing ---

def enhanced_esg_preprocess(text: str, nlp_model) -> str:
    """
    Enhanced preprocessing with:
    1. ESG acronym preservation
    2. Fix text encoding issues (reversed text)
    3. Remove 4-digit years
    4. Remove entities: ORG, PERSON, GPE
    5. Lemmatize and clean
    """
    if not text:
        return ""

    # 0. Preserve ESG acronyms with placeholders
    acronym_map = {}
    for i, acr in enumerate(ESG_ACRONYMS):
        placeholder = f"__ACRONYM{i}__"
        acronym_map[placeholder] = acr.lower()
        text = re.sub(rf'\b{acr}\b', placeholder, text, flags=re.IGNORECASE)

    # 1. Fix Text Encoding Issues
    reversed_patterns = ['htiw', 'ruo', 'srekrow', 'seeyolpme', 'eht']
    if any(pattern in text.lower() for pattern in reversed_patterns):
        reversed_count = sum(1 for p in reversed_patterns if p in text.lower())
        normal_count = sum(1 for p in ['with', 'our', 'workers', 'employees', 'the'] if p in text.lower())
        if reversed_count > normal_count:
            text = text[::-1]
    
    # Remove problematic non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 2. Regex Cleaning
    # Remove dates/years
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)
    text = re.sub(r'\bFY\d{2,4}\b', '', text, flags=re.IGNORECASE)
    
    # Handle numbers with units (preserve meaningful ones)
    text = re.sub(r'\d+\s*(tonnes?|kg|percent|%|MW|GW|kWh|litres?)', r'\1', text, flags=re.IGNORECASE)
    
    # Remove emails/URLs
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Spacy Processing
    doc = nlp_model(text)
    
    kept_tokens = []
    BANNED_ENTS = {'ORG', 'PERSON', 'GPE'}
    
    for token in doc:
        # Check if it's an acronym placeholder
        if token.text in acronym_map:
            kept_tokens.append(acronym_map[token.text])
            continue
            
        # Entity Filter
        if token.ent_type_ in BANNED_ENTS:
            continue
            
        # Stopword Filter
        if token.is_stop:
            continue
            
        # POS Filter (Keep content words)
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

# --- Coherence Calculation ---

def calculate_topic_coherence(topic_model, documents: List[str], topics: List[int]) -> float:
    """
    Calculate topic coherence using c_v metric.
    Higher is better (0-1 scale).
    """
    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.corpora import Dictionary
        
        # Prepare data
        texts = [doc.split() for doc in documents]
        dictionary = Dictionary(texts)
        
        # Get topics as word lists
        topics_words = []
        unique_topics = sorted(set(topics))
        for topic_id in unique_topics:
            if topic_id != -1:
                words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
                if words:
                    topics_words.append(words)
        
        if not topics_words:
            return 0.0
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        return coherence_score
    except Exception as e:
        logger.warning(f"Could not calculate coherence: {e}")
        return 0.0

# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced ESG BERTopic Pipeline")
    parser.add_argument('--input_dir', type=str, default='brsr_reports/outputs', help='Directory containing JSON reports')
    parser.add_argument('--output_dir', type=str, default='esg_topics_enhanced', help='Directory for results')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of files to process (for testing)')
    parser.add_argument('--min_chunk_len', type=int, default=150, help='Minimum char length for a text chunk')
    parser.add_argument('--min_cluster_size', type=int, default=50, help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--target_topics', type=int, default=80, help='Target number of topics after reduction')
    parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2', 
                        help='Sentence transformer model (all-mpnet-base-v2, all-MiniLM-L6-v2)')
    parser.add_argument('--reduce_outliers', action='store_true', help='Apply outlier reduction')
    
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
    
    # 3. Data Loading & Preprocessing
    logger.info("Loading and preprocessing documents...")
    all_documents = []
    doc_metadata = []
    
    for i, jf in enumerate(json_files):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = split_json_into_documents(data, strategy='paragraphs', min_length=args.min_chunk_len)
            
            for chunk in chunks:
                raw_text = chunk['text']
                processed_text = enhanced_esg_preprocess(raw_text, nlp)
                
                if len(processed_text.split()) >= 5:
                    all_documents.append(processed_text)
                    doc_metadata.append({
                        'source_file': jf.name,
                        'raw_preview': raw_text[:100] + "...",
                        'section_path': chunk.get('section_path', ''),
                        'processed_text': processed_text
                    })
            
            if (i+1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(json_files)} files...")
                
        except Exception as e:
            logger.warning(f"Failed to process {jf.name}: {e}")
            
    logger.info(f"Total valid documents after preprocessing: {len(all_documents)}")
    
    if len(all_documents) < 10:
        logger.error("Not enough documents to model. Check your data or filters.")
        sys.exit(1)

    # 4. Load Advanced Embedding Model
    logger.info(f"Loading sentence transformer model: {args.embedding_model}...")
    try:
        embedding_model = SentenceTransformer(args.embedding_model)
        logger.info(f"✓ Loaded {args.embedding_model}")
    except Exception as e:
        logger.warning(f"Could not load {args.embedding_model}, using default: {e}")
        embedding_model = "all-MiniLM-L6-v2"
    
    # 5. Configure Vectorizer
    vectorizer_model = CountVectorizer(
        min_df=0.03,
        max_df=0.5,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # 6. Configure UMAP
    umap_model = UMAP(
        n_neighbors=20,
        n_components=10,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # 7. Configure HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=15,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # 8. Configure Representation Models
    logger.info("Setting up representation models (MMR + KeyBERT)...")
    
    # MMR for keyword diversity
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    
    # KeyBERT for better keyword extraction
    keybert_model = KeyBERTInspired()
    
    # Combine both
    representation_model = {
        "MMR": mmr_model,
        "KeyBERT": keybert_model,
    }
    
    # 9. Initialize Enhanced BERTopic
    logger.info(f"Initializing Enhanced BERTopic...")
    logger.info(f"  - Embedding: {args.embedding_model}")
    logger.info(f"  - Min cluster size: {args.min_cluster_size}")
    logger.info(f"  - Target topics: {args.target_topics}")
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        nr_topics=None,  # Let it discover naturally, reduce later
        verbose=True,
        calculate_probabilities=False
    )
    
    # 10. Fit Model
    logger.info("Fitting model (this may take several minutes)...")
    topics, probs = topic_model.fit_transform(all_documents)
    
    initial_topics = len(set(topics)) - (1 if -1 in topics else 0)
    initial_outliers = sum(1 for t in topics if t == -1)
    logger.info(f"Initial results: {initial_topics} topics, {initial_outliers} outliers ({initial_outliers/len(topics)*100:.1f}%)")
    
    # 11. Reduce Outliers (if enabled)
    if args.reduce_outliers:
        logger.info("Reducing outliers...")
        try:
            new_topics = topic_model.reduce_outliers(
                all_documents,
                topics,
                strategy="distributions",
                threshold=0.1
            )
            topic_model.update_topics(all_documents, topics=new_topics)
            topics = new_topics
            
            reduced_outliers = sum(1 for t in topics if t == -1)
            logger.info(f"After outlier reduction: {reduced_outliers} outliers ({reduced_outliers/len(topics)*100:.1f}%)")
        except Exception as e:
            logger.warning(f"Could not reduce outliers: {e}")
    
    # 12. Hierarchical Topic Reduction
    logger.info(f"Reducing to {args.target_topics} topics hierarchically...")
    try:
        topic_model.reduce_topics(all_documents, nr_topics=args.target_topics)
        topics = topic_model.topics_
        final_topics = len(set(topics)) - (1 if -1 in topics else 0)
        logger.info(f"Final topics: {final_topics}")
    except Exception as e:
        logger.warning(f"Could not reduce topics: {e}")
    
    # 13. Calculate Coherence Score
    logger.info("Calculating topic coherence...")
    coherence_score = calculate_topic_coherence(topic_model, all_documents, topics)
    logger.info(f"Topic Coherence Score: {coherence_score:.4f} (0-1 scale, higher is better)")
    
    # 14. Export Results
    logger.info("Exporting results...")
    
    # A. Topic Info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_path / "topic_info.csv", index=False)
    
    # B. Enhanced Keywords (with both MMR and KeyBERT)
    keywords_data = []
    for topic_id in sorted(list(set(topics))):
        if topic_id == -1:
            continue
        
        # Get default c-TF-IDF representation
        default_words = topic_model.get_topic(topic_id)
        
        # Get representation-specific keywords
        try:
            mmr_words = topic_model.get_topics(full=True)["MMR"].get(topic_id, [])
            keybert_words = topic_model.get_topics(full=True)["KeyBERT"].get(topic_id, [])
        except:
            mmr_words = []
            keybert_words = []
        
        if default_words:
            keywords_data.append({
                'Topic': topic_id,
                'Count': topic_info[topic_info['Topic'] == topic_id]['Count'].values[0] if not topic_info.empty else 0,
                'Name': topic_info[topic_info['Topic'] == topic_id]['Name'].values[0] if not topic_info.empty else f"Topic {topic_id}",
                'Default_Keywords': ", ".join([w for w, s in default_words[:10]]),
                'MMR_Keywords': ", ".join([w for w, s in (mmr_words[:10] if mmr_words else default_words[:10])]),
                'KeyBERT_Keywords': ", ".join([w for w, s in (keybert_words[:10] if keybert_words else default_words[:10])])
            })
    
    pd.DataFrame(keywords_data).to_csv(output_path / "esg_topic_keywords_enhanced.csv", index=False)
    
    # C. Document Mapping
    doc_df = pd.DataFrame(doc_metadata)
    doc_df['Topic'] = topics
    topic_names_map = {row['Topic']: row['Name'] for row in keywords_data}
    doc_df['Topic_Name'] = doc_df['Topic'].map(topic_names_map).fillna("Outlier")
    doc_df.to_csv(output_path / "document_topics.csv", index=False)
    
    # D. Save Model
    logger.info("Saving BERTopic model...")
    topic_model.save(str(output_path / "bertopic_model"))
    
    # E. Save Metrics
    metrics = {
        'embedding_model': args.embedding_model,
        'initial_topics': initial_topics,
        'final_topics': len(set(topics)) - (1 if -1 in topics else 0),
        'initial_outliers': initial_outliers,
        'final_outliers': sum(1 for t in topics if t == -1),
        'coherence_score': coherence_score,
        'total_documents': len(all_documents)
    }
    pd.DataFrame([metrics]).to_csv(output_path / "model_metrics.csv", index=False)
    logger.info(f"Model metrics saved: {metrics}")
    
    # F. Generate Readable Topic Names
    logger.info("Generating readable topic names...")
    try:
        readable_names = {}
        for topic_id in topic_model.get_topics().keys():
            if topic_id == -1:
                readable_names[topic_id] = "Outliers - Mixed Topics"
            else:
                # Use MMR keywords if available, otherwise default
                try:
                    topic_words = topic_model.get_topics(full=True)["MMR"].get(topic_id, [])
                    if not topic_words:
                        topic_words = topic_model.get_topic(topic_id)
                except:
                    topic_words = topic_model.get_topic(topic_id)
                
                if topic_words:
                    keywords = [word for word, _ in topic_words[:3]]
                    readable_names[topic_id] = " & ".join([kw.replace("_", " ").title() for kw in keywords])
        
        topic_model.set_topic_labels(readable_names)
        logger.info(f"Updated labels for {len(readable_names)} topics")
    except Exception as e:
        logger.warning(f"Could not update topic labels: {e}")
    
    # G. Visualizations
    logger.info("Generating visualizations...")
    try:
        vis_topics = topic_model.visualize_topics(custom_labels=True)
        vis_topics.write_html(str(output_path / "intertopic_distance_map.html"))
        
        vis_bar = topic_model.visualize_barchart(top_n_topics=30, custom_labels=True)
        vis_bar.write_html(str(output_path / "topic_barchart.html"))
        
        # Hierarchy visualization
        vis_hierarchy = topic_model.visualize_hierarchy(custom_labels=True)
        vis_hierarchy.write_html(str(output_path / "topic_hierarchy.html"))
        
        logger.info("✓ Visualizations saved")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")

    logger.info("=" * 60)
    logger.info("ENHANCED BERTOPIC PIPELINE COMPLETE!")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Final Topics: {len(set(topics)) - (1 if -1 in topics else 0)}")
    logger.info(f"Coherence Score: {coherence_score:.4f}")
    logger.info(f"Outlier Rate: {sum(1 for t in topics if t == -1)/len(topics)*100:.1f}%")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
