"""
Update BERTopic visualizations with human-readable topic names
Loads the saved model and regenerates visualizations with better labels
"""

import pandas as pd
from bertopic import BERTopic
import os

def main():
    print("Loading readable topic names...")
    topics_df = pd.read_csv('esg_topics_output/topics_with_readable_names.csv')
    
    # Create mapping: topic_id -> readable_name
    topic_mapping = {}
    for _, row in topics_df.iterrows():
        topic_id = row['Topic']
        readable_name = row['Readable_Name']
        topic_mapping[topic_id] = readable_name
    
    print(f"Loaded {len(topic_mapping)} topic labels")
    
    # Load the saved BERTopic model
    print("\nLoading BERTopic model...")
    model_path = 'esg_topics_output/bertopic_model'
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("The pipeline may not have saved the model. Checking for alternative paths...")
        # Try alternative location
        alt_path = 'bertopic_model'
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"Found model at: {alt_path}")
        else:
            print("Model not found. Cannot update visualizations without the saved model.")
            print("\nTo save the model, add this line in esg_bertopic_pipeline.py after fitting:")
            print("    topic_model.save('esg_topics_output/bertopic_model')")
            return
    
    topic_model = BERTopic.load(model_path)
    print("Model loaded successfully!")
    
    # Update topic labels in the model
    print("\nUpdating topic labels...")
    new_labels = []
    for topic_id in sorted(topic_mapping.keys()):
        if topic_id in topic_mapping:
            new_labels.append(topic_mapping[topic_id])
        else:
            new_labels.append(f"Topic {topic_id}")
    
    # Set custom labels
    topic_model.set_topic_labels(topic_mapping)
    print(f"Updated labels for {len(topic_mapping)} topics")
    
    # Regenerate visualizations
    print("\n" + "="*80)
    print("Regenerating Visualizations with Readable Names")
    print("="*80)
    
    try:
        print("\n1. Generating topic bar chart...")
        fig_barchart = topic_model.visualize_barchart(top_n_topics=30, custom_labels=True)
        barchart_path = 'esg_topics_output/topic_barchart_readable.html'
        fig_barchart.write_html(barchart_path)
        print(f"   Saved: {barchart_path}")
    except Exception as e:
        print(f"   Error: {e}")
    
    try:
        print("\n2. Generating intertopic distance map...")
        fig_distance = topic_model.visualize_topics(custom_labels=True)
        distance_path = 'esg_topics_output/intertopic_distance_map_readable.html'
        fig_distance.write_html(distance_path)
        print(f"   Saved: {distance_path}")
    except Exception as e:
        print(f"   Error: {e}")
    
    try:
        print("\n3. Generating topic hierarchy...")
        fig_hierarchy = topic_model.visualize_hierarchy(custom_labels=True)
        hierarchy_path = 'esg_topics_output/topic_hierarchy_readable.html'
        fig_hierarchy.write_html(hierarchy_path)
        print(f"   Saved: {hierarchy_path}")
    except Exception as e:
        print(f"   Error: {e}")
    
    try:
        print("\n4. Generating heatmap...")
        fig_heatmap = topic_model.visualize_heatmap(custom_labels=True, n_clusters=20)
        heatmap_path = 'esg_topics_output/topic_heatmap_readable.html'
        fig_heatmap.write_html(heatmap_path)
        print(f"   Saved: {heatmap_path}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*80)
    print("[SUCCESS] Visualizations updated with readable names!")
    print("="*80)
    print("\nNew visualization files created:")
    print("  - topic_barchart_readable.html")
    print("  - intertopic_distance_map_readable.html")
    print("  - topic_hierarchy_readable.html")
    print("  - topic_heatmap_readable.html")

if __name__ == "__main__":
    main()
