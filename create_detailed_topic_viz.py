"""
Create comprehensive topic visualization with individual bar charts
Shows all topics with their keyword distributions
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_comprehensive_topic_visualization():
    """Create visualization showing all topics with individual bar charts"""
    
    print("Loading topic data...")
    topics_df = pd.read_csv('esg_topics_output/topics_with_readable_names.csv')
    
    # Sort by document count
    topics_df = topics_df.sort_values('Count', ascending=False)
    
    # Get top 50 topics (or all if less than 50)
    n_topics = min(50, len(topics_df))
    top_topics = topics_df.head(n_topics)
    
    print(f"Creating visualization for top {n_topics} topics...")
    
    # Create subplots - one per topic
    # Use a grid layout
    rows = n_topics
    fig = make_subplots(
        rows=rows, 
        cols=1,
        subplot_titles=[f"{row['Readable_Name']} ({row['Count']} docs)" 
                       for _, row in top_topics.iterrows()],
        vertical_spacing=0.02,
        specs=[[{"type": "bar"}] for _ in range(rows)]
    )
    
    # Add bar chart for each topic
    for idx, (_, topic_row) in enumerate(top_topics.iterrows(), 1):
        # Parse keywords
        keywords_str = topic_row['Top_Keywords']
        keywords = [k.strip() for k in keywords_str.split(',')[:10]]  # Top 10 keywords
        
        # Create importance scores (descending)
        scores = list(range(len(keywords), 0, -1))
        
        fig.add_trace(
            go.Bar(
                x=scores,
                y=keywords,
                orientation='h',
                marker=dict(color=f'rgba(58, 71, 80, 0.6)'),
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Importance: %{x}<extra></extra>'
            ),
            row=idx,
            col=1
        )
        
        # Update axes for this subplot
        fig.update_xaxes(title_text="Importance", row=idx, col=1, showgrid=True)
        fig.update_yaxes(title_text="", row=idx, col=1)
    
    # Update overall layout
    fig.update_layout(
        height=400 * n_topics,  # Scale height based on number of topics
        title_text=f"Top {n_topics} ESG Topics - Keyword Distributions<br><sub>Scroll to browse all topics</sub>",
        showlegend=False,
        font=dict(size=10)
    )
    
    output_path = 'esg_topics_output/all_topics_detailed.html'
    fig.write_html(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Height: {400 * n_topics}px (very tall - scroll to see all topics)")
    
    return output_path

if __name__ == "__main__":
    print("="*80)
    print("Creating Comprehensive Topic Visualization")
    print("="*80)
    create_comprehensive_topic_visualization()
    print("\n[SUCCESS] Created detailed topic visualization!")
    print("\nThis shows all topics with individual keyword bar charts.")
    print("Open the file and scroll down to browse all topics.")
