"""Check actual table on page 62"""
import camelot

tables = camelot.read_pdf(
    r'data/(17) Adani Green Energy Ltd.-20251230T103344Z-1-001/(17) Adani Green Energy Ltd/17_Adani Green Energy Ltd._2019_20.pdf',
    pages='62',
    flavor='stream',
    suppress_stdout=True
)

print(f'Tables on page 62: {len(tables)}')

if len(tables) > 0:
    df = tables[0].df
    print(f'\nFirst table shape: {df.shape}')
    print('\nFirst 10 rows:')
    print(df.head(10).to_string())
    
    print('\n\nColumn types:')
    for col in df.columns:
        print(f'  Column {col}: {df[col].dtype}')
