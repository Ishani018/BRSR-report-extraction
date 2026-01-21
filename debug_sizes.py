import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_sizes():
    data_dir = Path("brsr_reports/outputs")
    if not data_dir.exists():
        data_dir = Path(r"c:\Users\ishan\Desktop\BSBR reports\pdf-to-structured-reports\brsr_reports\outputs")
    
    files = list(data_dir.glob("*.json"))
    docs = []
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            # simplistic extraction for checking size
            text = str(data)
            docs.append((f.name, len(text)))
        except:
            pass
            
    docs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Total documents: {len(docs)}")
    print("Top 10 largest documents (by char count):")
    for name, length in docs[:10]:
        print(f"{name}: {length:,} chars")
        
    avg = sum(d[1] for d in docs) / len(docs)
    print(f"Average length: {avg:,.0f} chars")

if __name__ == "__main__":
    check_sizes()
