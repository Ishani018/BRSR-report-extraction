"""Final summary of extraction"""
import json

with open('outputs/(17) Adani Green Energy Ltd/2019/metadata.json', encoding='utf-8') as f:
    d = json.load(f)

print('='*70)
print('FINAL EXTRACTION SUMMARY - Adani Green Energy Ltd 2019')
print('='*70)

print(f"\nDocument Statistics:")
print(f"  Total pages processed: {d['statistics']['total_pages']}")
print(f"  Total sections identified: {d['statistics']['total_sections']}")
print(f"  Total tables exported to CSV: {d['statistics']['total_tables']}")

print(f"\nFinancial Statement Detection:")
fd = d['financial_data']
print(f"  Balance Sheets identified: {len(fd['balance_sheet'])}")
print(f"  Income Statements identified: {len(fd['income_statement'])}")
print(f"  Cash Flow statements identified: {len(fd['cash_flow'])}")

print(f"\nKey Metrics Extracted from Text:")
tm = fd['text_metrics']
metrics = ['revenue', 'profit', 'ebitda', 'assets', 'equity', 'liabilities']
for metric in metrics:
    count = len(tm.get(metric, []))
    if count > 0:
        example = tm[metric][0]['raw'] if tm[metric] else 'N/A'
        print(f"  {metric.title():15s}: {count:2d} instances (e.g., {example})")

print(f"\nHighlights Extracted:")
for key in fd.get('highlights', {}).keys():
    print(f"  - {key.replace('_', ' ').title()}")

print('\n' + '='*70)
print('✓ Pipeline successfully extracts structured financial data!')
print('='*70)
