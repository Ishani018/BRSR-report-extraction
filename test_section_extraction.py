"""
Test script for section extraction on a single PDF.
"""
from pathlib import Path
from main import process_single_pdf

# Test with Adani Green Energy 2019 report
pdf_path = Path(r"data\(17) Adani Green Energy Ltd.-20251230T103344Z-1-001\(17) Adani Green Energy Ltd\17_Adani Green Energy Ltd._2019_20.pdf")

print("Testing section extraction...")
print("="*80)

result = process_single_pdf(pdf_path)

print("\n" + "="*80)
if result["status"] == "success":
    print("STATUS: success")
    output_dir = Path(result["output_directory"])
    
    # Check what was created
    all_files = list(output_dir.rglob("*.*"))
    print(f"Files created: {len(all_files)}")
    
    # Check section files specifically
    section_dir = output_dir / "sections"
    if section_dir.exists():
        section_files = list(section_dir.glob("*.docx"))
        print(f"Section DOCX files: {len(section_files)}")
        for f in section_files:
            print(f"  - {f.name}")
        
        # Show metadata
        metadata_file = section_dir / "sections_metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
            print("\nSection detection results:")
            for section_key, data in metadata.items():
                if data.get('extracted'):
                    boundary = data['boundary']
                    stats = data['content_stats']
                    print(f"  ✓ {section_key}:")
                    print(f"      Pages: {boundary['start_page']}-{boundary['end_page']}")
                    print(f"      Confidence: {boundary['confidence']:.2f}")
                    print(f"      Characters: {stats['character_count']:,}")
                else:
                    print(f"  ✗ {section_key}: Not found")
else:
    print(f"STATUS: failed - {result.get('error')}")
print("="*80)
