import json
import re
import sys
from pathlib import Path
from typing import List
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_dir: Path) -> List[str]:
    """
    Load text from all JSON files in the directory.
    Treats the entire content of a JSON file as one document.
    """
    logger.info(f"Loading data from {data_dir}...")
    documents = []
    
    files = list(data_dir.glob("*.json"))
    if not files:
        logger.warning(f"No JSON files found in {data_dir}")
        return []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Extract text recursively from the structured JSON
            text_parts = []
            
            def extract_text_recursive(structure):
                for block in structure:
                    heading = block.get('heading', '')
                    if heading:
                        text_parts.append(heading)
                    
                    content = block.get('content', [])
                    if isinstance(content, list):
                        text_parts.extend([c for c in content if isinstance(c, str)])
                    elif isinstance(content, str):
                        text_parts.append(content)
                        
                    subsections = block.get('subsections', [])
                    if subsections:
                        extract_text_recursive(subsections)

            if 'structure' in data:
                extract_text_recursive(data['structure'])
                full_text = " ".join(text_parts)
                if full_text.strip():
                    documents.append(full_text)
            else:
                 # Fallback if structure key is missing
                 documents.append(json.dumps(data))

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

    logger.info(f"Loaded {len(documents)} documents.")
    return documents

def preprocess_text(documents: List[str]) -> List[str]:
    """
    Clean and preprocess the text data.
    - Remove entities (ORG, PERSON, GPE) using spaCy
    - Remove numbers/years via regex
    - Lemmatize
    """
    logger.info("Loading spaCy model...")
    try:
        # Disable parser for speed, but keep NER and Tagger
        nlp = spacy.load("en_core_web_sm", disable=["parser"]) 
    except OSError:
        logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        sys.exit(1)

    logger.info("Preprocessing documents...")
    
    # Increase max length for large files if necessary
    nlp.max_length = 3000000 

    # Compile regex patterns
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    fy_pattern = re.compile(r'\bFY\d{2,4}\b', re.IGNORECASE)
    page_pattern = re.compile(r'\bPage \d+\b', re.IGNORECASE)
    digit_pattern = re.compile(r'\b\d+\b')

    # Truncate extremely large documents to avoid stalling (max 50k chars ~ 10k words)
    # This keeps the core content while avoiding memory/time sinks on outliers
    regex_cleaned_docs = []
    cleaned_count = 0
    for doc_text in documents:
        if len(doc_text) > 50000:
            doc_text = doc_text[:50000]
            
        doc_text = year_pattern.sub('', doc_text)
        doc_text = fy_pattern.sub('', doc_text)
        doc_text = page_pattern.sub('', doc_text)
        doc_text = digit_pattern.sub('', doc_text)
        regex_cleaned_docs.append(doc_text)
        cleaned_count += 1
        if cleaned_count % 100 == 0:
             logger.info(f"Regex cleaned {cleaned_count}/{len(documents)} docs")

    # 2. NLP Processing (Batch with pipe)
    logger.info("Performing NLP processing (using nlp.pipe)...")
    cleaned_docs = []
    
    batch_size = 5
    total = len(regex_cleaned_docs)
    
    # Using nlp.pipe for efficiency
    for i, doc in enumerate(nlp.pipe(regex_cleaned_docs, batch_size=batch_size, n_process=1)):
        if (i + 1) % 5 == 0:
            logger.info(f"NLP processed {i + 1}/{total} docs")

        tokens = []
        for token in doc:
            # Filter unwanted entities
            if token.ent_type_ in ['ORG', 'PERSON', 'GPE']:
                continue
            
            if token.is_stop or token.is_punct or token.is_space:
                continue
                
            tokens.append(token.lemma_.lower())
            
        cleaned_docs.append(" ".join(tokens))
        
    return cleaned_docs

def run_topic_modeling(cleaned_docs: List[str], n_topics: int = 15):
    """
    Run LDA topic modeling using sklearn.
    """
    logger.info("Vectorizing text...")
    
    # Adjust min_df for small datasets (e.g. testing)
    n_docs = len(cleaned_docs)
    min_df_val = 10
    if n_docs < 20:
        min_df_val = 2
        
    vectorizer = CountVectorizer(
        max_df=0.5, 
        min_df=min_df_val, 
        stop_words='english',
        token_pattern=r'[a-zA-Z]{3,}' # Only words with 3+ chars
    )
    
    dtm = vectorizer.fit_transform(cleaned_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    logger.info(f"Vocabulary size after filtering: {len(feature_names)}")
    
    logger.info(f"Running LDA with {n_topics} components...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        n_jobs=-1, # Use all cores
        learning_method='batch' # 'online' might be faster but 'batch' is robust
    )
    
    lda.fit(dtm)
    
    return lda, feature_names, dtm, vectorizer

def save_results(lda_model, feature_names, dtm, vectorizer, filename: str = "exploratory_themes.txt", n_top_words: int = 15):
    """
    Format and save the topics to a file. Also generates an HTML visualization.
    """
    logger.info(f"Saving text results to {filename}...")
    
    output_lines = []
    output_lines.append("Exploratory LDA Topic Modeling Results")
    output_lines.append("======================================\n")
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        
        line = f"Topic {topic_idx + 1}: {top_features}"
        output_lines.append(line)
        print(line) 

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    # Visualization using pyLDAvis
    try:
        import pyLDAvis
        import numpy as np
        logger.info("Generating interactive visualization...")
        
        # Manually prepare data for pyLDAvis since .sklearn submodule might be missing
        # Normalize topic-term distributions
        topic_term_dists = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
        
        # Get document-topic distributions
        doc_topic_dists = lda_model.transform(dtm)
        
        # Get document lengths
        doc_lengths = np.array(dtm.sum(axis=1)).flatten()
        
        # Get term frequencies
        term_frequency = np.array(dtm.sum(axis=0)).flatten()
        
        # Prepare the visualization data
        vis_data = pyLDAvis.prepare(
            topic_term_dists=topic_term_dists,
            doc_topic_dists=doc_topic_dists,
            doc_lengths=doc_lengths,
            vocab=vectorizer.get_feature_names_out(),
            term_frequency=term_frequency,
            mds='tsne'
        )
        
        html_filename = filename.replace(".txt", "_visualization.html")
        pyLDAvis.save_html(vis_data, html_filename)
        logger.info(f"Visualization saved to {html_filename}")
        
    except ImportError:
        logger.warning("pyLDAvis or numpy not installed. Skipping visualization.")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of documents for testing", default=None)
    parser.add_argument("--topics", type=int, help="Number of topics", default=15)
    args = parser.parse_args()

    # Configuration
    DATA_DIR = Path("brsr_reports/outputs") 
    OUTPUT_FILE = "exploratory_themes.txt"
    N_TOPICS = args.topics
    
    # Check if dir exists
    if not DATA_DIR.exists():
        potential_path = Path(r"c:\Users\ishan\Desktop\BSBR reports\pdf-to-structured-reports\brsr_reports\outputs")
        if potential_path.exists():
            DATA_DIR = potential_path
        else:
            logger.error(f"Data directory {DATA_DIR} not found.")
            return

    # 1. Load Data
    raw_docs = load_data(DATA_DIR)
    if not raw_docs:
        return
        
    if args.limit:
        logger.info(f"Limiting to {args.limit} documents for testing.")
        raw_docs = raw_docs[:args.limit]

    # 2. Preprocess
    clean_docs = preprocess_text(raw_docs)

    # 3. Model
    lda_model, feature_names, dtm, vectorizer = run_topic_modeling(clean_docs, n_topics=N_TOPICS)

    # 4. Save
    save_results(lda_model, feature_names, dtm, vectorizer, filename=OUTPUT_FILE)

if __name__ == "__main__":
    main()
