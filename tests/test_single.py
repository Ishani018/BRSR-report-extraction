"""Quick test script to process one PDF"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import process_single_pdf

pdf_path = Path(r'data/(17) Adani Green Energy Ltd.-20251230T103344Z-1-001/(17) Adani Green Energy Ltd/17_Adani Green Energy Ltd._2019_20.pdf')

print("Processing:", pdf_path.name)
print("="*80)

result = process_single_pdf(pdf_path)

print("\n" + "="*80)
print("STATUS:", result['status'])
if result['status'] == 'success':
    output_dir = Path(result['output_directory'])
    all_files = list(output_dir.rglob("*.*"))
    print("Total files created:", len(all_files))
    
    # Show main files
    print("\nMain report files:")
    for f in sorted(output_dir.glob("*.*")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    # Show section files
    sections_dir = output_dir / "sections"
    if sections_dir.exists():
        print("\nSection files:")
        for f in sorted(sections_dir.glob("*.*")):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
print("="*80)
