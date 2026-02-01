"""
Generate human-readable topic names from BERTopic keywords
Reads esg_topic_keywords.csv and creates a version with better labels
"""

import pandas as pd
import re

def generate_readable_name(topic_id, keywords_str):
    """Generate human-readable topic name from keywords"""
    
    # Manual mapping for common ESG topics based on keywords
    topic_mappings = {
        # Environmental
        'energy_emission_reduce': 'Energy Efficiency & Emissions Reduction',
        'waste_total_category': 'Waste Management & Recycling',
        'water_withdrawal_consumption': 'Water Management & Conservation',
        'emission_water_initiative': 'Environmental Sustainability Initiatives',
        'plastic_sustainable_source': 'Sustainable Materials & Plastic Management',
        
        # Social
        'hazard_safety_work': 'Occupational Health & Safety',
        'student_school_child': 'Education & Community Development',
        'wage_pay_human_right': 'Wages & Human Rights',
        'talent_woman_diversity': 'Talent Management & Diversity',
        'able_worker_permanent': 'Workforce Inclusion & Disability',
        
        # Governance
        'risk_opportunity_implication': 'Risk & Opportunity Management',
        'assurance_perform_engagement': 'External Assurance & Verification',
        'appointment_regulation_appoint': 'Regulatory Compliance & Appointments',
        'datum_privacy_cyber': 'Data Privacy & Cyber Security',
        'stakeholder_identify_key': 'Stakeholder Identification & Engagement',
        
        # Compliance & Reporting
        'complaint_pende_close': 'Complaint Handling & Grievance Resolution',
        'grievance_mechanism_brief': 'Grievance Redressal Mechanisms',
        'diligence_human_right': 'Human Rights Due Diligence',
        
        # Business Operations
        'export_contribution_percentage': 'Export Operations & Markets',
        'sell_entity_business': 'Business Activities & Revenue',
        'procurement_vulnerable_marginalize': 'Preferential Procurement & Social Inclusion',
        'input_grievance_community': 'Community Engagement & Input Sourcing',
        
        # Specific Areas
        'vehicle_tyre_market': 'Automotive & Chemical Markets',
        'treatment_level_specify': 'Wastewater Treatment & Discharge',
        'joint_venture_associate': 'Joint Ventures & Subsidiaries',
        'asset_liability_recognise': 'Financial Assets & Liabilities',
    }
    
    # Extract key terms from keywords
    keywords = [k.strip() for k in keywords_str.lower().split(',')]
    
    # Try to match against known patterns
    for pattern, name in topic_mappings.items():
        pattern_words = pattern.split('_')
        if any(pw in ' '.join(keywords[:5]) for pw in pattern_words):
            return name
    
    # Fallback: capitalize and clean up keywords
    top_keywords = keywords[:3]
    cleaned = []
    for kw in top_keywords:
        # Remove underscores, clean up
        kw_clean = kw.replace('_', ' ').strip()
        # Skip if it's gibberish or reversed text
        if not re.match(r'^[a-z\s]+$', kw_clean) or len(kw_clean) < 3:
            continue
        cleaned.append(kw_clean.title())
    
    if cleaned:
        return ' & '.join(cleaned[:2])
    else:
        return f"Topic {topic_id}"

def main():
    print("Reading topic keywords...")
    df = pd.read_csv('esg_topics_output/esg_topic_keywords.csv')
    
    print(f"Processing {len(df)} topics...")
    
    # Generate readable names
    readable_names = []
    for idx, row in df.iterrows():
        topic_id = row['Topic']
        keywords = row['Top_Keywords']
        
        readable_name = generate_readable_name(topic_id, keywords)
        readable_names.append(readable_name)
    
    # Add to dataframe
    df['Readable_Name'] = readable_names
    
    # Reorder columns
    df = df[['Topic', 'Count', 'Readable_Name', 'Name', 'Top_Keywords']]
    
    # Save enhanced version
    output_file = 'esg_topics_output/topics_with_readable_names.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Saved enhanced topics to: {output_file}")
    
    # Print top 30 for preview
    print("\n" + "="*100)
    print("TOP 30 TOPICS WITH READABLE NAMES")
    print("="*100)
    
    for idx, row in df.head(30).iterrows():
        print(f"\n[Topic {row['Topic']}]: {row['Readable_Name']}")
        print(f"   Documents: {row['Count']}")
        print(f"   Keywords: {row['Top_Keywords'][:100]}...")

if __name__ == "__main__":
    main()
