"""Check extracted financial data"""
import json

with open('outputs/(17) Adani Green Energy Ltd/2019/metadata.json', encoding='utf-8') as f:
    data = json.load(f)

fd = data.get('financial_data', {})

print("Financial Data Summary:")
print("="*60)
print(f"Balance sheets identified: {len(fd.get('balance_sheet', []))}")
print(f"Income statements identified: {len(fd.get('income_statement', []))}")
print(f"Cash flows identified: {len(fd.get('cash_flow', []))}")

tm = fd.get('text_metrics', {})
print(f"\nText metrics extracted: {len(tm)} types")
for key in list(tm.keys())[:8]:
    matches = tm[key]
    print(f"  {key}: {len(matches)} matches")
    if matches:
        print(f"    Example: {matches[0].get('raw', 'N/A')}")
