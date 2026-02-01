"""
Regenerate ONLY the visualizations from saved data with readable names
This avoids re-running the entire pipeline while updating the viz
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path

def regenerate_visualizations_from_data():
    """Regenerate BERTopic visualizations using saved data and readable names"""
    
    print("Loading topic data and assignments...")
    
    # Load the readable names
    topics_df = pd.read_csv('esg_topics_output/topics_with_readable_names.csv')
    readable_names = dict(zip(topics_df['Topic'], topics_df['Readable_Name']))
    
    # Load document-topic assignments
    doc_topics_df = pd.read_csv('esg_topics_output/document_topics.csv')
    
    print(f"Loaded {len(topics_df)} topics and {len(doc_topics_df)} documents")
    
    # Since we don't have the saved model, we need to create visualizations from scratch
    # Let's create better versions using plotly directly
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 1. Enhanced Bar Chart
    print("\n1. Creating enhanced bar chart...")
    top_topics = topics_df.nlargest(30, 'Count')
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=top_topics['Readable_Name'][::-1],
        x=top_topics['Count'][::-1],
        orientation='h',
        marker=dict(
            color=top_topics['Count'][::-1],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Doc Count")
        ),
        text=top_topics['Count'][::-1],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                      'Documents: %{x}<br>' +
                      '<extra></extra>'
    ))
    
    fig_bar.update_layout(
        title={
            'text': 'Top 30 ESG Topics by Document Count<br><sub>Human-Readable Topic Names</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Number of Documents',
        yaxis_title='',
        height=1000,
        showlegend=False,
        font=dict(size=11),
        margin=dict(l=300, r=100, t=100, b=100)
    )
    
    bar_path = 'esg_topics_output/topic_barchart.html'
    fig_bar.write_html(bar_path)
    print(f"   Saved: {bar_path}")
    
    # 2. Topic Distribution Sunburst (Alternative to distance map)
    print("\n2. Creating topic distribution sunburst chart...")
    
    # Categorize topics
    def categorize(keywords):
        k = keywords.lower()
        if any(word in k for word in ['energy', 'emission', 'water', 'waste', 'climate', 'environmental']):
            return 'Environmental (E)'
        elif any(word in k for word in ['safety', 'health', 'employee', 'worker', 'community', 'education']):
            return 'Social (S)'
        else:
            return 'Governance (G)'
    
    topics_df['Category'] = topics_df['Top_Keywords'].apply(categorize)
    
    # Prepare data for sunburst
    labels = ['ESG Topics']
    parents = ['']
    values = [topics_df['Count'].sum()]
    colors = ['lightgray']
    
    # Add categories
    for cat in ['Environmental (E)', 'Social (S)', 'Governance (G)']:
        cat_topics = topics_df[topics_df['Category'] == cat]
        labels.append(cat)
        parents.append('ESG Topics')
        values.append(cat_topics['Count'].sum())
        if cat.startswith('E'):
            colors.append('#2ecc71')
        elif cat.startswith('S'):
            colors.append('#3498db')
        else:
            colors.append('#9b59b6')
    
    # Add top 10 topics from each category
    for cat in ['Environmental (E)', 'Social (S)', 'Governance (G)']:
        cat_topics = topics_df[topics_df['Category'] == cat].nlargest(10, 'Count')
        for _, row in cat_topics.iterrows():
            labels.append(row['Readable_Name'])
            parents.append(cat)
            values.append(row['Count'])
            colors.append('lightgray')
    
    fig_sunburst = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Documents: %{value}<br><extra></extra>'
    ))
    
    fig_sunburst.update_layout(
        title={
            'text': 'ESG Topics - Hierarchical Distribution<br><sub>Organized by E, S, G Categories</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        font=dict(size=12)
    )
    
    sunburst_path = 'esg_topics_output/intertopic_distance_map.html'
    fig_sunburst.write_html(sunburst_path)
    print(f"   Saved: {sunburst_path}")
    print("   (Replaced distance map with sunburst - shows topic hierarchy)")
    
    # 3. Create a scrollable topic explorer
    print("\n3. Creating interactive topic explorer...")
    
    # Sort topics by count
    sorted_topics = topics_df.sort_values('Count', ascending=False)
    
    fig_explorer = go.Figure(data=[go.Table(
        columnwidth=[50, 300, 80, 400],
        header=dict(
            values=['<b>ID</b>', '<b>Topic Name</b>', '<b>Docs</b>', '<b>Top Keywords</b>'],
            fill_color='#34495e',
            align='left',
            font=dict(color='white', size=13),
            height=40
        ),
        cells=dict(
            values=[
                sorted_topics['Topic'].astype(str),
                sorted_topics['Readable_Name'],
                sorted_topics['Count'].astype(str),
                sorted_topics['Top_Keywords'].str[:100] + '...'
            ],
            fill_color=[['#ecf0f1' if i % 2 == 0 else 'white' for i in range(len(sorted_topics))]],
            align='left',
            font=dict(size=11),
            height=30
        )
    )])
    
    fig_explorer.update_layout(
        title={
            'text': 'All ESG Topics - Interactive Explorer<br><sub>Click and scroll to browse all 345 topics</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=900
    )
    
    explorer_path = 'esg_topics_output/topic_explorer.html'
    fig_explorer.write_html(explorer_path)
    print(f"   Saved: {explorer_path}")
    
    print("\n" + "="*80)
    print("[SUCCESS] Updated all visualizations with readable topic names!")
    print("="*80)
    print("\nUpdated files:")
    print("  ✓ topic_barchart.html - Now shows readable names")
    print("  ✓ intertopic_distance_map.html - Replaced with hierarchical sunburst")
    print("  ✓ topic_explorer.html - NEW scrollable table of all topics")
    print("\nAll visualizations now display human-readable labels!")

if __name__ == "__main__":
    regenerate_visualizations_from_data()
