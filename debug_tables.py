"""Debug script to see what tables are being extracted"""
from pathlib import Path
from extract_tables import extract_tables, score_table_quality
import logging

logging.basicConfig(level=logging.INFO)

pdf_path = Path(r'data/(17) Adani Green Energy Ltd.-20251230T103344Z-1-001/(17) Adani Green Energy Ltd/17_Adani Green Energy Ltd._2019_20.pdf')

print("Extracting tables from:", pdf_path.name)
print("="*80)

tables = extract_tables(pdf_path)

print(f"\nFound {len(tables)} raw tables")
print("\nTable Quality Scores:")
print("-" * 80)

for i, table in enumerate(tables[:15]):  # First 15 tables
    quality = score_table_quality(table.dataframe)
    print(f"Table {i+1} (Page {table.page_number}): Quality={quality:.3f}, Shape={table.dataframe.shape}")
    if quality > 0:
        print(f"  First few rows:")
        print(table.dataframe.head(3).to_string()[:200])
    print()
