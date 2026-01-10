"""
Debug script to see what headings are in the first 20 pages.
"""
import pdfplumber
from pathlib import Path

pdf_path = Path(r"data\(17) Adani Green Energy Ltd.-20251230T103344Z-1-001\(17) Adani Green Energy Ltd\17_Adani Green Energy Ltd._2019_20.pdf")

print("Analyzing first 20 pages for potential headings...")
print("="*80)

with pdfplumber.open(pdf_path) as pdf:
    for page_num in range(min(20, len(pdf.pages))):
        page = pdf.pages[page_num]
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        
        if not words:
            continue
        
        # Calculate median font size
        font_sizes = [w.get('height', 10) for w in words]
        median_font = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10
        
        # Find large text at top of page
        prominent_text = []
        for word in words:
            if word['top'] < 250 and word.get('height', 0) > median_font * 1.1:
                prominent_text.append((word['text'], word['top'], word.get('height', 0)))
        
        if prominent_text:
            # Group by y-position to form lines
            prominent_text.sort(key=lambda x: x[1])
            current_line = [prominent_text[0][0]]
            current_y = prominent_text[0][1]
            
            for text, y, size in prominent_text[1:]:
                if abs(y - current_y) <= 3:
                    current_line.append(text)
                else:
                    if current_line:
                        line_text = ' '.join(current_line)
                        if len(line_text) < 120:
                            print(f"\nPage {page_num + 1}:")
                            print(f"  '{line_text}'")
                    current_line = [text]
                    current_y = y
            
            if current_line:
                line_text = ' '.join(current_line)
                if len(line_text) < 120:
                    print(f"\nPage {page_num + 1}:")
                    print(f"  '{line_text}'")

print("\n" + "="*80)
