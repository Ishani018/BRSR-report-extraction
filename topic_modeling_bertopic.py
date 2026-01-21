"""
Topic modeling using BERTopic on a single BRSR JSON file.

This script processes one JSON file, splits it into sections/paragraphs,
runs BERTopic topic modeling, and exports results to CSV + visualization.
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from topic_modeling_utils import split_json_into_documents, extract_text_from_json, advanced_preprocess  # noqa: E402
from config.config import LOG_FORMAT, LOG_LEVEL  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer

try:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
except ImportError as e:
    print(f"ERROR: Failed to import bertopic: {e}")
    print("\nThis might be due to NumPy compatibility issues.")
    print("Try: pip install 'numpy<2'")
    print("Or: pip uninstall bertopic -y && pip install bertopic")
    sys.exit(1)

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_bertopic(
    documents: List[Dict[str, Any]],
    num_topics: int = 5,
    min_length: int = 100
) -> tuple:
    """
    Run BERTopic topic modeling on documents.
    
    Args:
        documents: List of document dictionaries with 'text' key
        num_topics: Number of topics to extract (if None, auto-detect)
        min_length: Minimum document length
        
    Returns:
        Tuple of (model, topics, probabilities)
    """
    logger.info(f"Running BERTopic with {num_topics} topics on {len(documents)} documents...")
    
    # Extract texts
    # Extract texts and apply advanced preprocessing
    texts = []
    for doc in documents:
        if len(doc['text'].strip()) >= min_length:
            processed = advanced_preprocess(doc['text'])
            if len(processed.split()) >= 3: # Keep docs with at least 3 tokens
                texts.append(processed)
    
    if len(texts) < 2:
        logger.error(f"Not enough documents for topic modeling. Found {len(texts)} documents.")
        return None, None, None
    
    # Initialize BERTopic with parameters tuned for smaller document sets
    # HDBSCAN min_cluster_size must be smaller than number of documents
    # For small sets, use 2-3 as min_cluster_size
    min_cluster_size = max(2, min(5, len(texts) // 10))
    
    # Reduce dimensions for UMAP to work better with small sets
    n_components = min(5, max(2, num_topics if num_topics else 5))
    n_neighbors = min(5, max(2, len(texts) - 1))
    
    # Create HDBSCAN model tuned for smaller document sets
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Create UMAP model for dimensionality reduction
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # Configure CountVectorizer for N-grams (Bigrams/Trigrams)
    # We already removed stopwords in advanced_preprocess, so we don't need to pass them here.
    # We want unigrams, bigrams, and trigrams.
    vectorizer_model = CountVectorizer(ngram_range=(1, 3))
    
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=num_topics if num_topics else 'auto',
        verbose=True,
        calculate_probabilities=True
    )
    
    # Fit the model
    topics, probabilities = model.fit_transform(texts)
    
    # Get unique topics (excluding outliers with topic -1)
    unique_topics = [t for t in set(topics) if t != -1]
    
    # If no topics found (all outliers), try to force topic reduction
    if len(unique_topics) == 0 and num_topics and num_topics > 0:
        logger.warning("No topics found (all documents are outliers). Trying to force topic reduction...")
        try:
            # Force topic reduction to create topics from outliers
            topics_outliers, new_topics, new_probs = model.reduce_topics(texts, topics, probabilities, nr_topics=num_topics)
            topics = new_topics
            probabilities = new_probs
            unique_topics = [t for t in set(topics) if t != -1]
            logger.info(f"After topic reduction: Discovered {len(unique_topics)} topics.")
        except Exception as e:
            logger.warning(f"Topic reduction failed: {e}. Continuing with original topics.")
    
    logger.info(f"BERTopic completed. Discovered {len(unique_topics)} topics (plus outliers).")
    
    return model, topics, probabilities


def export_results(
    json_path: Path,
    documents: List[Dict[str, Any]],
    model: Any,
    topics: List[int],
    probabilities: List[List[float]],
    num_topics: int,
    output_dir: Path
):
    """
    Export topic modeling results to CSV files and create visualization.
    
    Args:
        json_path: Path to input JSON file
        documents: List of document dictionaries
        model: Trained BERTopic model
        topics: List of topic assignments for each document
        probabilities: List of topic probabilities for each document
        num_topics: Number of topics (or None for auto)
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_stem = json_path.stem
    
    logger.info(f"Exporting results to {output_dir}...")
    
    # Filter documents to match topics length
    valid_documents = [doc for doc in documents if len(doc['text'].strip()) >= 100]
    
    # 1. Export section-topics CSV
    section_topics_data = []
    for i, doc in enumerate(valid_documents):
        topic_id = topics[i] if i < len(topics) else -1
        prob_dist = probabilities[i] if i < len(probabilities) and probabilities[i] is not None else []
        # Get topic probability (handle numpy arrays properly)
        topic_prob = 0.0
        if prob_dist is not None and len(prob_dist) > 0:
            try:
                import numpy as np
                if isinstance(prob_dist, np.ndarray):
                    if topic_id >= 0 and topic_id < len(prob_dist):
                        topic_prob = float(prob_dist[topic_id])
                elif isinstance(prob_dist, list) and topic_id >= 0 and topic_id < len(prob_dist):
                    topic_prob = float(prob_dist[topic_id])
            except (IndexError, TypeError, ValueError):
                topic_prob = 0.0
        
        section_topics_data.append({
            'section_heading': doc.get('section_heading', f'Document {i+1}'),
            'section_level': doc.get('section_level', 0),
            'section_path': doc.get('section_path', ''),
            'dominant_topic': topic_id,
            'topic_probability': topic_prob,
            'text_length': len(doc['text'])
        })
    
    section_topics_df = pd.DataFrame(section_topics_data)
    section_topics_path = output_dir / 'section_topics.csv'
    section_topics_df.to_csv(section_topics_path, index=False)
    logger.info(f"Exported section-topics to {section_topics_path}")
    
    # 2. Export topic keywords CSV
    topic_keywords_data = []
    try:
        # Get topic info from model
        topic_info = model.get_topic_info()
        
        # Get all unique topics from actual assignments (not just from topic_info)
        all_topics = sorted([t for t in set(topics) if t != -1])
        
        # If no topics in info but we have topics, extract them directly
        if len(all_topics) > 0:
            for topic_id in all_topics:
                try:
                    # Get top words for this topic
                    words_probs = model.get_topic(topic_id)
                    if words_probs:
                        keywords = ', '.join([word for word, _ in words_probs[:10]])
                    else:
                        keywords = "N/A"
                except Exception as e:
                    logger.warning(f"Could not get keywords for topic {topic_id}: {e}")
                    keywords = "N/A"
                
                # Count documents in this topic
                doc_count = topics.count(topic_id) if topics else 0
                
                topic_keywords_data.append({
                    'topic_id': topic_id,
                    'top_keywords': keywords,
                    'num_documents': doc_count
                })
        
        # Also try topic_info if available and different
        if len(topic_keywords_data) == 0 and hasattr(model, 'get_topic_info'):
            try:
                topic_info_df = model.get_topic_info()
                for _, row in topic_info_df.iterrows():
                    topic_id = row['Topic']
                    if topic_id == -1:
                        continue  # Skip outliers
                    
                    try:
                        words_probs = model.get_topic(topic_id)
                        if words_probs:
                            keywords = ', '.join([word for word, _ in words_probs[:10]])
                        else:
                            keywords = "N/A"
                    except Exception as e:
                        logger.warning(f"Could not get keywords for topic {topic_id}: {e}")
                        keywords = "N/A"
                    
                    doc_count = topics.count(topic_id) if topics else 0
                    
                    topic_keywords_data.append({
                        'topic_id': topic_id,
                        'top_keywords': keywords,
                        'num_documents': doc_count
                    })
            except Exception as e:
                logger.warning(f"Could not use topic_info: {e}")
        
        # Sort by topic_id
        topic_keywords_data.sort(key=lambda x: x['topic_id'])
    except Exception as e:
        logger.warning(f"Could not extract topic keywords: {e}")
    
    topic_keywords_df = pd.DataFrame(topic_keywords_data)
    topic_keywords_path = output_dir / 'topic_keywords.csv'
    topic_keywords_df.to_csv(topic_keywords_path, index=False)
    logger.info(f"Exported topic-keywords to {topic_keywords_path}")
    
    # 3. Create visualizations
    try:
        # Intertopic distance map (visualization methods are on the model instance)
        try:
            vis_intertopic = model.visualize_topics(topics=topics)
            if vis_intertopic:
                vis_path = output_dir / 'visualization_intertopic.html'
                vis_intertopic.write_html(str(vis_path))
                logger.info(f"Exported intertopic distance map to {vis_path}")
        except Exception as e:
            logger.warning(f"Could not create intertopic distance map: {e}")
        
        # Bar chart
        try:
            vis_barchart = model.visualize_barchart(topics=topics)
            if vis_barchart:
                vis_path = output_dir / 'visualization_barchart.html'
                vis_barchart.write_html(str(vis_path))
                logger.info(f"Exported topic bar chart to {vis_path}")
        except Exception as e:
            logger.warning(f"Could not create bar chart: {e}")
        
        # Combined visualization page
        vis_path = output_dir / 'visualization.html'
        with open(vis_path, 'w', encoding='utf-8') as f:
            f.write("""
            <html>
            <head>
                <title>BERTopic Visualization</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    iframe { width: 100%; height: 800px; border: 1px solid #ccc; margin: 20px 0; }
                </style>
            </head>
            <body>
                <h1>BERTopic Topic Modeling Results</h1>
                <p>Input file: """ + json_path.name + """</p>
                <p>Number of topics: """ + str(len([t for t in set(topics) if t != -1])) + """</p>
                <p>Number of documents: """ + str(len(documents)) + """</p>
                <h2>Intertopic Distance Map</h2>
                <iframe src="visualization_intertopic.html"></iframe>
                <h2>Topic Bar Chart</h2>
                <iframe src="visualization_barchart.html"></iframe>
            </body>
            </html>
            """)
        logger.info(f"Exported combined visualization to {vis_path}")
    except Exception as e:
        logger.warning(f"Could not create visualization files: {e}")
    
    # 4. Export summary
    summary_path = output_dir / 'summary.txt'
    unique_topics = [t for t in set(topics) if t != -1]
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("BERTopic Topic Modeling Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {json_path.name}\n")
        f.write(f"Processing date: {datetime.now().isoformat()}\n\n")
        f.write(f"Model: BERTopic\n")
        f.write(f"Number of topics (requested): {num_topics if num_topics else 'auto'}\n")
        f.write(f"Number of topics (discovered): {len(unique_topics)}\n")
        f.write(f"Number of documents: {len(documents)}\n")
        f.write(f"Number of outliers (topic -1): {topics.count(-1) if topics else 0}\n\n")
        f.write("Topic Distribution:\n")
        for topic_id in sorted(unique_topics):
            count = topics.count(topic_id) if topics else 0
            f.write(f"  Topic {topic_id}: {count} documents\n")
        if topics and -1 in topics:
            count = topics.count(-1)
            f.write(f"  Outliers (-1): {count} documents\n")
    
    logger.info(f"Exported summary to {summary_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run BERTopic topic modeling on a single BRSR JSON file'
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
        help='Number of topics to extract (default: 5, use 0 for auto-detect)'
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
        help='Output directory (default: brsr_reports/topic_modeling_results/bertopic/{json_filename}/)'
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
    
    # Run BERTopic
    try:
        num_topics = args.num_topics if args.num_topics > 0 else None
        model, topics, probabilities = run_bertopic(
            documents,
            num_topics=num_topics,
            min_length=args.min_length
        )
        
        if model is None or topics is None:
            logger.error("BERTopic modeling failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error running BERTopic: {e}", exc_info=True)
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_base = BASE_DIR / 'brsr_reports' / 'topic_modeling_results' / 'bertopic'
        output_dir = output_base / json_path.stem
    
    # Export results
    try:
        export_results(
            json_path,
            documents,
            model,
            topics,
            probabilities,
            args.num_topics,
            output_dir
        )
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"Topic modeling complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
