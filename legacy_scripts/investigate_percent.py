import pandas as pd
import sys

# Set encoding
sys.stdout.reconfigure(encoding='utf-8')

file_path = r'c:\Users\shobhit.sharma\Downloads\Uzio-ADP-Payment-and-Emergency-Data-Review-main\Uzio-ADP-Payment-and-Emergency-Data-Review-main\UZIO_vs_ADP_Payment_Report_ADP (1).xlsx'
sheet_name = 'ADP Payment Data'

with open('investigate_percent_reading.txt', 'w', encoding='utf-8') as f:
    f.write(f"Reading '{sheet_name}' from {file_path}...\n")
    try:
        # Read as object
        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=object)
        
        # Look for Marek Cieply (ID 7VGB6ZA62)
        # Based on previous analysis, we know the ID.
        # Find column that looks like Percentage
        pct_cols = [c for c in df.columns if 'percent' in str(c).lower()]
        f.write(f"Percentage columns found: {pct_cols}\n")
        
        if pct_cols:
            col = pct_cols[0]
            # Filter for Marek
            mask = df.apply(lambda x: x.astype(str).str.contains('7VGB6ZA62', case=False).any(), axis=1)
            rows = df[mask]
            
            f.write(f"\n--- Rows for 7VGB6ZA62 ---\n")
            f.write(rows.to_string() + "\n")
            
            f.write(f"\n--- Raw Values for column '{col}' ---\n")
            vals = rows[col].tolist()
            for v in vals:
                f.write(f"Value: {v} | Type: {type(v)}\n")
                
    except Exception as e:
        f.write(f"Error: {e}")

print("Investigation complete. See investigate_percent_reading.txt")
