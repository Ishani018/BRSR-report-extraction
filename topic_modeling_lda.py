"""
Topic modeling using LDA (gensim) on a single BRSR JSON file.

This script processes one JSON file, splits it into sections/paragraphs,
runs LDA topic modeling, and exports results to CSV + visualization.
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import re

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from topic_modeling_utils import split_json_into_documents, extract_text_from_json, clean_text  # noqa: E402
from config.config import LOG_FORMAT, LOG_LEVEL  # noqa: E402

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
except ImportError as e:
    print(f"ERROR: Failed to import required packages: {e}")
    print("Install with: pip install gensim pyLDAvis")
    sys.exit(1)

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )





def run_lda(
    documents: List[Dict[str, Any]],
    num_topics: int = 5,
    min_length: int = 100
) -> tuple:
    """
    Run LDA topic modeling on documents.
    
    Args:
        documents: List of document dictionaries with 'text' key
        num_topics: Number of topics to extract
        min_length: Minimum document length
        
    Returns:
        Tuple of (model, dictionary, corpus, topics, topic_distributions)
    """
    logger.info(f"Running LDA with {num_topics} topics on {len(documents)} documents...")
    
    # Extract and preprocess texts
    texts = []
    valid_docs = []
    for doc in documents:
        text = doc['text']
        if len(text.strip()) >= min_length:
            # Clean text using centralized function
            cleaned_text = clean_text(text, remove_stopwords=True)
            tokens = cleaned_text.split()
            if len(tokens) >= 5:  # Minimum tokens per document
                texts.append(tokens)
                valid_docs.append(doc)
    
    if len(texts) < num_topics:
        logger.warning(
            f"Only {len(texts)} documents available, but {num_topics} topics requested. "
            f"Using {min(len(texts), num_topics)} topics instead."
        )
        num_topics = max(1, min(len(texts), num_topics))
    
    if len(texts) < 2:
        logger.error(f"Not enough documents for topic modeling. Found {len(texts)} documents.")
        return None, None, None, None, None
    
    # Create dictionary and corpus
    logger.info("Creating dictionary and corpus...")
    dictionary = corpora.Dictionary(texts)
    
    # Filter extreme values
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    
    # Create corpus (bow format)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    logger.info(f"Dictionary size: {len(dictionary)} unique tokens")
    logger.info(f"Corpus size: {len(corpus)} documents")
    
    # Train LDA model
    logger.info("Training LDA model...")
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True,
        iterations=50
    )
    
    # Get topic distributions for each document
    topics = []
    topic_distributions = []
    for doc_bow in corpus:
        # When per_word_topics=True, model[doc_bow] returns a tuple:
        # (topic_distribution, word_topics, phi_values)
        result = model[doc_bow]
        
        # Extract topic distribution (first element)
        if isinstance(result, tuple) and len(result) >= 1:
            doc_topics = result[0]  # topic_distribution is first element
        else:
            doc_topics = result
        
        # doc_topics is a list of tuples: [(topic_id, probability), ...]
        # Get dominant topic (highest probability)
        if doc_topics and isinstance(doc_topics, list) and len(doc_topics) > 0:
            # Get the tuple with highest probability
            dominant_tuple = max(doc_topics, key=lambda x: x[1] if isinstance(x, tuple) and len(x) >= 2 else 0)
            if isinstance(dominant_tuple, tuple) and len(dominant_tuple) >= 2:
                dominant_topic = dominant_tuple[0]  # Extract topic_id from tuple
            else:
                dominant_topic = 0
        else:
            dominant_topic = 0
        topics.append(dominant_topic)
        
        # Get topic distribution
        dist = [0.0] * num_topics
        if doc_topics and isinstance(doc_topics, list):
            for topic_tuple in doc_topics:
                if isinstance(topic_tuple, tuple) and len(topic_tuple) >= 2:
                    topic_id, prob = topic_tuple[0], topic_tuple[1]
                    if isinstance(topic_id, int) and 0 <= topic_id < num_topics:
                        dist[topic_id] = float(prob)
        topic_distributions.append(dist)
    
    # Calculate coherence score
    try:
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        logger.info(f"Model coherence score: {coherence_score:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate coherence score: {e}")
        coherence_score = None
    
    logger.info(f"LDA completed. Discovered {num_topics} topics.")
    
    return model, dictionary, corpus, topics, topic_distributions, coherence_score


def export_results(
    json_path: Path,
    documents: List[Dict[str, Any]],
    model: Any,
    dictionary: Any,
    corpus: Any,
    topics: List[int],
    topic_distributions: List[List[float]],
    coherence_score: float,
    num_topics: int,
    output_dir: Path
):
    """
    Export topic modeling results to CSV files and create visualization.
    
    Args:
        json_path: Path to input JSON file
        documents: List of document dictionaries
        model: Trained LDA model
        dictionary: Gensim dictionary
        corpus: Gensim corpus
        topics: List of topic assignments for each document
        topic_distributions: List of topic distributions for each document
        coherence_score: Coherence score (if available)
        num_topics: Number of topics
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_stem = json_path.stem
    
    logger.info(f"Exporting results to {output_dir}...")
    
    # Filter documents to match topics length
    valid_documents = [doc for doc in documents if len(doc['text'].strip()) >= 100]
    valid_documents = [doc for doc in valid_documents if len(clean_text(doc['text'], remove_stopwords=True).split()) >= 5]
    
    # 1. Export section-topics CSV
    section_topics_data = []
    for i, doc in enumerate(valid_documents):
        topic_id = topics[i] if i < len(topics) else 0
        dist = topic_distributions[i] if i < len(topic_distributions) else [0.0] * num_topics
        section_topics_data.append({
            'section_heading': doc.get('section_heading', f'Document {i+1}'),
            'section_level': doc.get('section_level', 0),
            'section_path': doc.get('section_path', ''),
            'dominant_topic': topic_id,
            'topic_probability': dist[topic_id] if dist and topic_id < len(dist) else 0.0,
            'topic_distribution': str(dist),
            'text_length': len(doc['text'])
        })
    
    section_topics_df = pd.DataFrame(section_topics_data)
    section_topics_path = output_dir / 'section_topics.csv'
    section_topics_df.to_csv(section_topics_path, index=False)
    logger.info(f"Exported section-topics to {section_topics_path}")
    
    # 2. Export topic keywords CSV
    topic_keywords_data = []
    try:
        for topic_id in range(num_topics):
            # Get top words for this topic
            words_probs = model.show_topic(topic_id, topn=10)
            keywords = ', '.join([word for word, _ in words_probs])
            
            # Count documents in this topic
            doc_count = topics.count(topic_id) if topics else 0
            
            topic_keywords_data.append({
                'topic_id': topic_id,
                'top_keywords': keywords,
                'num_documents': doc_count
            })
    except Exception as e:
        logger.warning(f"Could not extract topic keywords: {e}")
    
    topic_keywords_df = pd.DataFrame(topic_keywords_data)
    topic_keywords_path = output_dir / 'topic_keywords.csv'
    topic_keywords_df.to_csv(topic_keywords_path, index=False)
    logger.info(f"Exported topic-keywords to {topic_keywords_path}")
    
    # 3. Create visualization using pyLDAvis
    try:
        vis_path = output_dir / 'visualization.html'
        vis = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis, str(vis_path))
        logger.info(f"Exported visualization to {vis_path}")
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")
        # Create a simple HTML placeholder
        with open(vis_path, 'w', encoding='utf-8') as f:
            f.write(f"""
            <html>
            <head><title>LDA Visualization</title></head>
            <body>
            <h1>LDA Topic Modeling Results</h1>
            <p>Visualization generation failed. Please check topic_keywords.csv for topic details.</p>
            <p>Input file: {json_path.name}</p>
            <p>Number of topics: {num_topics}</p>
            <p>Number of documents: {len(documents)}</p>
            </body>
            </html>
            """)
    
    # 4. Export summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LDA Topic Modeling Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {json_path.name}\n")
        f.write(f"Processing date: {datetime.now().isoformat()}\n\n")
        f.write(f"Model: LDA (gensim)\n")
        f.write(f"Number of topics: {num_topics}\n")
        f.write(f"Number of documents: {len(documents)}\n")
        f.write(f"Dictionary size: {len(dictionary)} unique tokens\n")
        if coherence_score is not None:
            f.write(f"Coherence score (c_v): {coherence_score:.4f}\n")
        f.write("\nTopic Distribution:\n")
        for topic_id in range(num_topics):
            count = topics.count(topic_id) if topics else 0
            f.write(f"  Topic {topic_id}: {count} documents\n")
    
    logger.info(f"Exported summary to {summary_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run LDA topic modeling on a single BRSR JSON file'
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to JSON file to process'
    )
    parser.add_argument(
        '--num_topics',
        type=int,
        default=5,
        help='Number of topics to extract (default: 5)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='sections',
        choices=['sections', 'paragraphs', 'both'],
        help='Strategy for splitting documents (default: sections)'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=100,
        help='Minimum document length in characters (default: 100)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: brsr_reports/topic_modeling_results/lda/{json_filename}/)'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate JSON file
    json_path = Path(args.json_file)
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)
    
    logger.info(f"Processing JSON file: {json_path}")
    
    # Load JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        sys.exit(1)
    
    # Split into documents
    documents = split_json_into_documents(
        json_data,
        strategy=args.strategy,
        min_length=args.min_length
    )
    
    if len(documents) < 2:
        logger.error(f"Not enough documents for topic modeling. Found {len(documents)} documents.")
        sys.exit(1)
    
    # Run LDA
    try:
        result = run_lda(
            documents,
            num_topics=args.num_topics,
            min_length=args.min_length
        )
        
        if result[0] is None:
            logger.error("LDA modeling failed")
            sys.exit(1)
        
        model, dictionary, corpus, topics, topic_distributions, coherence_score = result
    except Exception as e:
        logger.error(f"Error running LDA: {e}", exc_info=True)
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_base = BASE_DIR / 'brsr_reports' / 'topic_modeling_results' / 'lda'
        output_dir = output_base / json_path.stem
    
    # Export results
    try:
        export_results(
            json_path,
            documents,
            model,
            dictionary,
            corpus,
            topics,
            topic_distributions,
            coherence_score,
            args.num_topics,
            output_dir
        )
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"Topic modeling complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
