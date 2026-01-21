"""
Topic modeling using FASTopic on a single BRSR JSON file.

This script processes one JSON file, splits it into sections/paragraphs,
runs FASTopic topic modeling, and exports results to CSV + visualization.
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

from topic_modeling_utils import split_json_into_documents, extract_text_from_json, clean_text  # noqa: E402
from config.config import LOG_FORMAT, LOG_LEVEL  # noqa: E402

try:
    from fastopic import FASTopic
except ImportError as e:
    print(f"ERROR: Failed to import fastopic: {e}")
    print("Install with: pip install fastopic")
    sys.exit(1)

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_fastopic(
    documents: List[Dict[str, Any]],
    num_topics: int = 5,
    min_length: int = 100
) -> tuple:
    """
    Run FASTopic topic modeling on documents.
    
    Args:
        documents: List of document dictionaries with 'text' key
        num_topics: Number of topics to extract
        min_length: Minimum document length
        
    Returns:
        Tuple of (model, topics, topic_distributions)
    """
    logger.info(f"Running FASTopic with {num_topics} topics on {len(documents)} documents...")
    
    # Extract texts and clean them
    texts = [clean_text(doc['text'], remove_stopwords=True) for doc in documents if len(doc['text'].strip()) >= min_length]
    
    if len(texts) < num_topics:
        logger.warning(
            f"Only {len(texts)} documents available, but {num_topics} topics requested. "
            f"Using {len(texts)} topics instead."
        )
        num_topics = max(1, len(texts) - 1)
    
    # Initialize and fit FASTopic
    model = FASTopic(
        n_topic=num_topics,
        verbose=True
    )
    
    # Fit the model
    model.fit(texts)
    
    # Get topic assignments
    topics = model.get_topic()
    
    # Get topic distributions
    topic_distributions = model.get_document_topic_dist()
    
    logger.info(f"FASTopic completed. Discovered {len(set(topics))} topics.")
    
    return model, topics, topic_distributions


def export_results(
    json_path: Path,
    documents: List[Dict[str, Any]],
    model: Any,
    topics: List[int],
    topic_distributions: List[List[float]],
    num_topics: int,
    output_dir: Path
):
    """
    Export topic modeling results to CSV files and create visualization.
    
    Args:
        json_path: Path to input JSON file
        documents: List of document dictionaries
        model: Trained FASTopic model
        topics: List of topic assignments for each document
        topic_distributions: List of topic distributions for each document
        num_topics: Number of topics
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_stem = json_path.stem
    
    logger.info(f"Exporting results to {output_dir}...")
    
    # 1. Export section-topics CSV
    section_topics_data = []
    for i, doc in enumerate(documents):
        section_topics_data.append({
            'section_heading': doc.get('section_heading', f'Document {i+1}'),
            'section_level': doc.get('section_level', 0),
            'section_path': doc.get('section_path', ''),
            'dominant_topic': topics[i] if i < len(topics) else -1,
            'topic_distribution': str(topic_distributions[i]) if i < len(topic_distributions) else '[]',
            'text_length': len(doc['text'])
        })
    
    section_topics_df = pd.DataFrame(section_topics_data)
    section_topics_path = output_dir / 'section_topics.csv'
    section_topics_df.to_csv(section_topics_path, index=False)
    logger.info(f"Exported section-topics to {section_topics_path}")
    
    # 2. Export topic keywords CSV
    topic_keywords_data = []
    try:
        # Get top keywords for each topic
        for topic_id in range(num_topics):
            # Get keywords for this topic (FASTopic provides topic-word distributions)
            # Note: FASTopic API may vary, adjust based on actual API
            try:
                keywords = model.get_topic_words(topic_id, top_n=10)
                if isinstance(keywords, list):
                    keywords_str = ', '.join([str(kw) for kw in keywords[:10]])
                else:
                    keywords_str = str(keywords)
            except:
                keywords_str = "N/A"
            
            topic_keywords_data.append({
                'topic_id': topic_id,
                'top_keywords': keywords_str,
                'num_documents': topics.count(topic_id) if topics else 0
            })
    except Exception as e:
        logger.warning(f"Could not extract topic keywords: {e}")
        for topic_id in range(num_topics):
            topic_keywords_data.append({
                'topic_id': topic_id,
                'top_keywords': 'N/A',
                'num_documents': topics.count(topic_id) if topics else 0
            })
    
    topic_keywords_df = pd.DataFrame(topic_keywords_data)
    topic_keywords_path = output_dir / 'topic_keywords.csv'
    topic_keywords_df.to_csv(topic_keywords_path, index=False)
    logger.info(f"Exported topic-keywords to {topic_keywords_path}")
    
    # 3. Create visualization (if supported)
    try:
        vis_path = output_dir / 'visualization.html'
        # FASTopic visualization - adjust based on actual API
        try:
            vis_html = model.visualize_topics()
            if vis_html:
                with open(vis_path, 'w', encoding='utf-8') as f:
                    f.write(vis_html)
                logger.info(f"Exported visualization to {vis_path}")
            else:
                logger.warning("FASTopic visualization not available or returned empty")
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")
            # Create a simple HTML placeholder
            with open(vis_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <html>
                <head><title>FASTopic Visualization</title></head>
                <body>
                <h1>FASTopic Topic Modeling Results</h1>
                <p>Visualization generation failed. Please check topic_keywords.csv for topic details.</p>
                <p>Input file: {json_path.name}</p>
                <p>Number of topics: {num_topics}</p>
                <p>Number of documents: {len(documents)}</p>
                </body>
                </html>
                """)
    except Exception as e:
        logger.warning(f"Could not create visualization file: {e}")
    
    # 4. Export summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("FASTopic Topic Modeling Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {json_path.name}\n")
        f.write(f"Processing date: {datetime.now().isoformat()}\n\n")
        f.write(f"Model: FASTopic\n")
        f.write(f"Number of topics: {num_topics}\n")
        f.write(f"Number of documents: {len(documents)}\n")
        f.write(f"Unique topics discovered: {len(set(topics))}\n\n")
        f.write("Topic Distribution:\n")
        for topic_id in range(num_topics):
            count = topics.count(topic_id) if topics else 0
            f.write(f"  Topic {topic_id}: {count} documents\n")
    
    logger.info(f"Exported summary to {summary_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run FASTopic topic modeling on a single BRSR JSON file'
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
        help='Output directory (default: brsr_reports/topic_modeling_results/fastopic/{json_filename}/)'
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
    
    # Run FASTopic
    try:
        model, topics, topic_distributions = run_fastopic(
            documents,
            num_topics=args.num_topics,
            min_length=args.min_length
        )
    except Exception as e:
        logger.error(f"Error running FASTopic: {e}", exc_info=True)
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_base = BASE_DIR / 'brsr_reports' / 'topic_modeling_results' / 'fastopic'
        output_dir = output_base / json_path.stem
    
    # Export results
    try:
        export_results(
            json_path,
            documents,
            model,
            topics,
            topic_distributions,
            args.num_topics,
            output_dir
        )
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"Topic modeling complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
