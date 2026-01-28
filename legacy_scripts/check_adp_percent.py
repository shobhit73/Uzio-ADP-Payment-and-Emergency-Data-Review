import pandas as pd
import sys

# Set encoding
sys.stdout.reconfigure(encoding='utf-8')

adp_file = r'c:\Users\shobhit.sharma\Downloads\Uzio-ADP-Payment-and-Emergency-Data-Review-main\Uzio-ADP-Payment-and-Emergency-Data-Review-main\ADP_DD_Details.csv'

with open('check_adp_percent.txt', 'w', encoding='utf-8') as f:
    try:
        # Read all as string to see raw format
        df = pd.read_csv(adp_file, dtype=str)
        
        # Filter for Marek Cieply (ID 7VGB6ZA62 or just Name 'Cieply, Marek')
        # Based on previous analysis, ID was 7VGB6ZA62 
        # But wait, ADP ID might be different or mapped. 
        # Let's search by name "Marek" or just print rows where DEPOSIT PERCENT is not null
        
        mask = df.astype(str).apply(lambda x: x.str.contains('Marek', case=False)).any(axis=1)
        marek_rows = df[mask]
        
        f.write("--- Rows for Marek ---\n")
        f.write(marek_rows.to_string() + "\n\n")
        
        f.write("--- Sample Deposit Percent Values ---\n")
        pct_col = [c for c in df.columns if 'percent' in c.lower()][0]
        f.write(f"Column: {pct_col}\n")
        f.write(df[pct_col].dropna().head(10).to_string())

    except Exception as e:
        f.write(f"Error: {e}")

print("Check complete. See check_adp_percent.txt")
