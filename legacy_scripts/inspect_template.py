import pandas as pd

try:
    xls = pd.ExcelFile('Payment and Emergency Tool Input Template.xlsx')
    with open('template_structure.txt', 'w') as f:
        f.write(f"Sheets: {xls.sheet_names}\n")
        for s in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=s)
            f.write(f"\nSheet: {s}\n")
            f.write(f"Columns: {list(df.columns)}\n")
            # f.write(f"First row: {df.iloc[0].to_dict() if not df.empty else 'Empty'}\n")
except Exception as e:
    with open('template_structure.txt', 'w') as f:
        f.write(f"Error: {e}")
