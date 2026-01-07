"""Check for tables in financial statement pages"""
import camelot
from pathlib import Path

pdf_path = Path(r'data/(17) Adani Green Energy Ltd.-20251230T103344Z-1-001/(17) Adani Green Energy Ltd/17_Adani Green Energy Ltd._2019_20.pdf')

# Try extracting from later pages where financial statements usually are
page_ranges = ['80-100', '100-120', '120-142']

for page_range in page_ranges:
    print(f"\nChecking pages {page_range}:")
    print("-" * 60)
    
    tables_lattice = camelot.read_pdf(str(pdf_path), pages=page_range, flavor='lattice', suppress_stdout=True)
    print(f"  Lattice: {len(tables_lattice)} tables")
    
    tables_stream = camelot.read_pdf(str(pdf_path), pages=page_range, flavor='stream', suppress_stdout=True)
    print(f"  Stream: {len(tables_stream)} tables")
    
    if len(tables_lattice) > 0:
        print(f"  Sample table shape: {tables_lattice[0].df.shape}")
