"""
Create enhanced visualizations with readable topic names
Uses the document-topic assignments and readable names to create new visualizations
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

def create_barchart_visualization():
    """Create bar chart of top topics with readable names"""
    print("Creating bar chart visualization...")
    
    # Load data
    topics_df = pd.read_csv('esg_topics_output/topics_with_readable_names.csv')
    
    # Sort by count and get top 30
    topics_df = topics_df.sort_values('Count', ascending=False).head(30)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=topics_df['Readable_Name'][::-1],  # Reverse for better display
        x=topics_df['Count'][::-1],
        orientation='h',
        marker=dict(
            color=topics_df['Count'][::-1],
            colorscale='Viridis',
            showscale=True
        ),
        text=topics_df['Count'][::-1],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Documents: %{x}<br><extra></extra>'
    ))
    
    fig.update_layout(
        title='Top 30 ESG Topics by Document Count',
        xaxis_title='Number of Documents',
        yaxis_title='Topic',
        height=900,
        showlegend=False,
        font=dict(size=11)
    )
    
    output_path = 'esg_topics_output/topic_barchart_readable.html'
    fig.write_html(output_path)
    print(f"Saved: {output_path}")
    return output_path

def create_topic_summary_table():
    """Create an HTML table with all topics"""
    print("\nCreating topic summary table...")
    
    topics_df = pd.read_csv('esg_topics_output/topics_with_readable_names.csv')
    topics_df = topics_df.sort_values('Count', ascending=False)
    
    # Create styled table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Topic ID</b>', '<b>Topic Name</b>', '<b>Document Count</b>', '<b>Top Keywords</b>'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[
                topics_df['Topic'].astype(str),
                topics_df['Readable_Name'],
                topics_df['Count'].astype(str),
                topics_df['Top_Keywords'].str[:80] + '...'  # Truncate for display
            ],
            fill_color='lavender',
            align='left',
            font=dict(size=11),
            height=25
        )
    )])
    
    fig.update_layout(
        title='All ESG Topics - Complete Summary',
        height=800
    )
    
    output_path = 'esg_topics_output/topics_summary_table.html'
    fig.write_html(output_path)
    print(f"Saved: {output_path}")
    return output_path

def create_category_breakdown():
    """Create visualization showing ESG category breakdown"""
    print("\nCreating ESG category breakdown...")
    
    topics_df = pd.read_csv('esg_topics_output/topics_with_readable_names.csv')
    
    # Categorize topics based on keywords
    def categorize_topic(keywords):
        keywords_lower = keywords.lower()
        
        # Environmental keywords
        env_keywords = ['energy', 'emission', 'water', 'waste', 'climate', 'environmental', 
                       'renewable', 'solar', 'biodiversity', 'pollution', 'carbon']
        # Social keywords
        social_keywords = ['safety', 'health', 'employee', 'worker', 'community', 'education',
                          'student', 'wage', 'human right', 'diversity', 'talent', 'training']
        # Governance keywords
        gov_keywords = ['governance', 'risk', 'compliance', 'assurance', 'board', 'audit',
                       'regulation', 'policy', 'stakeholder', 'privacy', 'cyber']
        
        env_score = sum(1 for kw in env_keywords if kw in keywords_lower)
        social_score = sum(1 for kw in social_keywords if kw in keywords_lower)
        gov_score = sum(1 for kw in gov_keywords if kw in keywords_lower)
        
        if env_score >= social_score and env_score >= gov_score:
            return 'Environmental (E)'
        elif social_score >= gov_score:
            return 'Social (S)'
        else:
            return 'Governance (G)'
    
    topics_df['ESG_Category'] = topics_df['Top_Keywords'].apply(categorize_topic)
    
    # Calculate totals by category
    category_counts = topics_df.groupby('ESG_Category')['Count'].sum().reset_index()
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=category_counts['ESG_Category'],
        values=category_counts['Count'],
        marker=dict(colors=['#2ecc71', '#3498db', '#9b59b6']),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Documents: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='ESG Topic Distribution - Document Count by Category',
        height=600
    )
    
    output_path = 'esg_topics_output/esg_category_breakdown.html'
    fig.write_html(output_path)
    print(f"Saved: {output_path}")
    return output_path

def main():
    print("="*80)
    print("Creating Enhanced Visualizations with Readable Names")
    print("="*80)
    
    created_files = []
    
    try:
        file_path = create_barchart_visualization()
        created_files.append(file_path)
    except Exception as e:
        print(f"Error creating bar chart: {e}")
    
    try:
        file_path = create_topic_summary_table()
        created_files.append(file_path)
    except Exception as e:
        print(f"Error creating summary table: {e}")
    
    try:
        file_path = create_category_breakdown()
        created_files.append(file_path)
    except Exception as e:
        print(f"Error creating category breakdown: {e}")
    
    print("\n" + "="*80)
    print("[SUCCESS] Created enhanced visualizations!")
    print("="*80)
    print("\nNew visualization files:")
    for file_path in created_files:
        print(f"  - {file_path}")
    
    print("\nThese visualizations use the human-readable topic names!")

if __name__ == "__main__":
    main()
