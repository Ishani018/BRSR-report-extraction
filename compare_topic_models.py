"""
Compare BERTopic and Top2Vec Results
------------------------------------
Generates a comparison report showing metrics from both topic modeling approaches
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

def load_results():
    """Load results from both pipelines"""
    results = {}
    
    # Load BERTopic Optimized Results
    try:
        bertopic_path = Path('esg_topics_optimized')
        results['bertopic'] = {
            'topics': pd.read_csv(bertopic_path / 'esg_topic_keywords.csv'),
            'documents': pd.read_csv(bertopic_path / 'document_topics.csv'),
            'exists': True
        }
        print(f"[SUCCESS] Loaded BERTopic results from {bertopic_path}")
    except Exception as e:
        print(f"[WARNING] Could not load BERTopic results: {e}")
        results['bertopic'] = {'exists': False}
    
    # Load Top2Vec Results
    try:
        top2vec_path = Path('esg_topics_top2vec')
        results['top2vec'] = {
            'topics': pd.read_csv(top2vec_path / 'topic_keywords.csv'),
            'documents': pd.read_csv(top2vec_path / 'document_topics.csv'),
            'stats': pd.read_csv(top2vec_path / 'model_stats.csv'),
            'exists': True
        }
        print(f"[SUCCESS] Loaded Top2Vec results from {top2vec_path}")
    except Exception as e:
        print(f"[WARNING] Could not load Top2Vec results: {e}")
        results['top2vec'] = {'exists': False}
    
    return results

def calculate_metrics(results):
    """Calculate comparison metrics"""
    metrics = {}
    
    # BERTopic Metrics
    if results['bertopic']['exists']:
        bt_docs = results['bertopic']['documents']
        bt_topics = results['bertopic']['topics']
        
        total_docs = len(bt_docs)
        outliers = len(bt_docs[bt_docs['Topic'] == -1])
        num_topics = len(bt_topics)
        
        metrics['bertopic'] = {
            'total_documents': total_docs,
            'num_topics': num_topics,
            'outlier_count': outliers,
            'outlier_percentage': (outliers / total_docs * 100) if total_docs > 0 else 0,
            'avg_topic_size': bt_topics['Count'].mean(),
            'median_topic_size': bt_topics['Count'].median(),
            'largest_topic': bt_topics['Count'].max(),
            'smallest_topic': bt_topics['Count'].min()
        }
    
    # Top2Vec Metrics
    if results['top2vec']['exists']:
        t2v_stats = results['top2vec']['stats'].iloc[0]
        
        metrics['top2vec'] = {
            'total_documents': int(t2v_stats['total_documents']),
            'num_topics': int(t2v_stats['num_topics']),
            'outlier_count': 0,  # Top2Vec assigns all docs
            'outlier_percentage': 0.0,
            'avg_topic_size': float(t2v_stats['avg_topic_size']),
            'median_topic_size': float(t2v_stats['median_topic_size']),
            'largest_topic': int(t2v_stats['largest_topic_size']),
            'smallest_topic': int(t2v_stats['smallest_topic_size'])
        }
    
    return metrics

def create_comparison_visualization(metrics):
    """Create comparison charts"""
    
    # Prepare data for comparison
    metric_names = ['Number of Topics', 'Outlier %', 'Avg Topic Size', 'Largest Topic']
    bertopic_vals = []
    top2vec_vals = []
    
    if 'bertopic' in metrics:
        bertopic_vals = [
            metrics['bertopic']['num_topics'],
            metrics['bertopic']['outlier_percentage'],
            metrics['bertopic']['avg_topic_size'],
            metrics['bertopic']['largest_topic']
        ]
    
    if 'top2vec' in metrics:
        top2vec_vals = [
            metrics['top2vec']['num_topics'],
            metrics['top2vec']['outlier_percentage'],
            metrics['top2vec']['avg_topic_size'],
            metrics['top2vec']['largest_topic']
        ]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Number of Topics', 'Outlier Percentage', 
                       'Average Topic Size', 'Largest Topic Size'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = {'BERTopic': '#3498db', 'Top2Vec': '#e74c3c'}
    
    # Add bars for each metric
    for idx, (metric_name, bt_val, t2v_val) in enumerate(zip(metric_names, bertopic_vals, top2vec_vals)):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        fig.add_trace(
            go.Bar(name='BERTopic', x=['BERTopic'], y=[bt_val], marker_color=colors['BERTopic'], showlegend=(idx==0)),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(name='Top2Vec', x=['Top2Vec'], y=[t2v_val], marker_color=colors['Top2Vec'], showlegend=(idx==0)),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        title_text="BERTopic vs Top2Vec - Performance Comparison",
        showlegend=True
    )
    
    return fig

def generate_report(metrics):
    """Generate HTML report"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Topic Modeling Comparison Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .metric-container {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-card h3 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }}
            .comparison-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                margin: 20px 0;
            }}
            .comparison-table th, .comparison-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .comparison-table th {{
                background-color: #34495e;
                color: white;
            }}
            .comparison-table tr:hover {{
                background-color: #f5f5f5;
            }}
            .winner {{
                background-color: #d4edda;
                font-weight: bold;
            }}
            .recommendation {{
                background: #fff3cd;
                padding: 20px;
                border-left: 4px solid #ffc107;
                margin: 20px 0;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <h1>ESG Topic Modeling Comparison Report</h1>
        <p><strong>Date:</strong> {date}</p>
        
        <h2>Overview</h2>
        <p>This report compares the performance of optimized BERTopic and Top2Vec on your ESG corpus.</p>
        
        <h2>Key Metrics Comparison</h2>
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th>BERTopic (Optimized)</th>
                <th>Top2Vec</th>
                <th>Winner</th>
            </tr>
            <tr>
                <td>Total Documents</td>
                <td>{bt_total_docs}</td>
                <td>{t2v_total_docs}</td>
                <td>-</td>
            </tr>
            <tr class="{topics_winner_class}">
                <td>Number of Topics</td>
                <td>{bt_num_topics}</td>
                <td>{t2v_num_topics}</td>
                <td>{topics_winner}</td>
            </tr>
            <tr class="{outlier_winner_class}">
                <td>Outlier Percentage</td>
                <td>{bt_outlier_pct:.1f}%</td>
                <td>{t2v_outlier_pct:.1f}%</td>
                <td>{outlier_winner}</td>
            </tr>
            <tr>
                <td>Average Topic Size</td>
                <td>{bt_avg_size:.1f}</td>
                <td>{t2v_avg_size:.1f}</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Largest Topic</td>
                <td>{bt_largest}</td>
                <td>{t2v_largest}</td>
                <td>-</td>
            </tr>
        </table>
        
        <h2>Analysis</h2>
        <div class="recommendation">
            <h3>Recommendations:</h3>
            {recommendations}
        </div>
        
        <h2>Next Steps</h2>
        <ul>
            <li>Review the topic keywords in both approaches</li>
            <li>Check if topics align with known ESG categories (E, S, G)</li>
            <li>Consider which topic granularity works best for your use case</li>
            <li>If outliers are still high in BERTopic, Top2Vec may be better</li>
        </ul>
    </body>
    </html>
    """
    
    # Determine winners
    bt_metrics = metrics.get('bertopic', {})
    t2v_metrics = metrics.get('top2vec', {})
    
    # Topics winner (closer to 120 is better)
    topics_target = 120
    bt_topic_diff = abs(bt_metrics.get('num_topics', 999) - topics_target)
    t2v_topic_diff = abs(t2v_metrics.get('num_topics', 999) - topics_target)
    topics_winner = "BERTopic" if bt_topic_diff < t2v_topic_diff else "Top2Vec"
    topics_winner_class = "winner" if topics_winner == "BERTopic" else ""
    
    # Outlier winner (lower is better)
    outlier_winner = "BERTopic" if bt_metrics.get('outlier_percentage', 100) < t2v_metrics.get('outlier_percentage', 100) else "Top2Vec"
    outlier_winner_class = "winner" if outlier_winner == "BERTopic" else ""
    
    # Generate recommendations
    recommendations = []
    
    if bt_metrics.get('outlier_percentage', 0) < 15:
        recommendations.append("✓ BERTopic outlier rate is acceptable (<15%)")
    else:
        recommendations.append("⚠ BERTopic outlier rate is still high. Top2Vec may be better.")
    
    if bt_metrics.get('num_topics', 0) >= 100 and bt_metrics.get('num_topics', 0) <= 150:
        recommendations.append("✓ BERTopic topic count is in optimal range (100-150)")
    elif bt_metrics.get('num_topics', 0) > 150:
        recommendations.append("⚠ BERTopic still has too many topics. Consider increasing min_cluster_size.")
    
    if t2v_metrics.get('num_topics', 0) < bt_metrics.get('num_topics', 999):
        recommendations.append("✓ Top2Vec produces fewer, more general topics")
    
    recommendations_html = "<ul>" + "".join(f"<li>{r}</li>" for r in recommendations) + "</ul>"
    
    # Fill template
    from datetime import datetime
    report_html = html_content.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        bt_total_docs=bt_metrics.get('total_documents', 'N/A'),
        t2v_total_docs=t2v_metrics.get('total_documents', 'N/A'),
        bt_num_topics=bt_metrics.get('num_topics', 'N/A'),
        t2v_num_topics=t2v_metrics.get('num_topics', 'N/A'),
        bt_outlier_pct=bt_metrics.get('outlier_percentage', 0),
        t2v_outlier_pct=t2v_metrics.get('outlier_percentage', 0),
        bt_avg_size=bt_metrics.get('avg_topic_size', 0),
        t2v_avg_size=t2v_metrics.get('avg_topic_size', 0),
        bt_largest=bt_metrics.get('largest_topic', 'N/A'),
        t2v_largest=t2v_metrics.get('largest_topic', 'N/A'),
        topics_winner=topics_winner,
        topics_winner_class=topics_winner_class,
        outlier_winner=outlier_winner,
        outlier_winner_class=outlier_winner_class,
        recommendations=recommendations_html
    )
    
    return report_html

def main():
    print("="*80)
    print("Topic Modeling Comparison Report Generator")
    print("="*80)
    
    # Load results
    print("\nLoading results from both pipelines...")
    results = load_results()
    
    if not results['bertopic']['exists'] and not results['top2vec']['exists']:
        print("\n[ERROR] No results found! Please run at least one pipeline first.")
        print("  - BERTopic: python esg_bertopic_optimized.py")
        print("  - Top2Vec: python esg_top2vec_pipeline.py")
        return
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    # Generate HTML report
    print("\nGenerating report...")
    report_html = generate_report(metrics)
    
    # Save report
    output_file = 'comparison_report.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_html)
    
    print(f"\n[SUCCESS] Comparison report saved to: {output_file}")
    
    # Also create visualization if both exist
    if results['bertopic']['exists'] and results['top2vec']['exists']:
        print("\nGenerating comparison charts...")
        fig = create_comparison_visualization(metrics)
        viz_file = 'comparison_charts.html'
        fig.write_html(viz_file)
        print(f"[SUCCESS] Comparison charts saved to: {viz_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if 'bertopic' in metrics:
        print("\nBERTopic (Optimized):")
        print(f"  Topics: {metrics['bertopic']['num_topics']}")
        print(f"  Outliers: {metrics['bertopic']['outlier_percentage']:.1f}%")
        print(f"  Avg Topic Size: {metrics['bertopic']['avg_topic_size']:.1f}")
    
    if 'top2vec' in metrics:
        print("\nTop2Vec:")
        print(f"  Topics: {metrics['top2vec']['num_topics']}")
        print(f"  Outliers: {metrics['top2vec']['outlier_percentage']:.1f}%")
        print(f"  Avg Topic Size: {metrics['top2vec']['avg_topic_size']:.1f}")
    
    print("\n" + "="*80)
    print(f"Open {output_file} in your browser to view the full report!")
    print("="*80)

if __name__ == "__main__":
    main()
