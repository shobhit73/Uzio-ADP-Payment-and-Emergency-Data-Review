import pandas as pd
import sys

# Set encoding
sys.stdout.reconfigure(encoding='utf-8')

adp_file = r'c:\Users\shobhit.sharma\Downloads\Uzio-ADP-Payment-and-Emergency-Data-Review-main\Uzio-ADP-Payment-and-Emergency-Data-Review-main\ADP_DD_Details.csv'
uzio_file = r'c:\Users\shobhit.sharma\Downloads\Uzio-ADP-Payment-and-Emergency-Data-Review-main\Uzio-ADP-Payment-and-Emergency-Data-Review-main\KDL_Payment_Data_Uzio.xlsx'

with open('raw_data_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(f"--- Analyzing ADP File: {adp_file} ---\n")
    try:
        # Read CSV with some robustness
        adp_df = pd.read_csv(adp_file, dtype=str)
        f.write(f"Columns: {list(adp_df.columns)}\n\n")
        
        # Look for relevant ADP columns
        prio_cols = [c for c in adp_df.columns if 'priority' in c.lower()]
        acct_cols = [c for c in adp_df.columns if 'account' in c.lower() or 'type' in c.lower()]
        
        f.write(f"Priority-like cols: {prio_cols}\n")
        if prio_cols:
            for c in prio_cols:
                f.write(f"First 10 values of '{c}':\n{adp_df[c].head(10).to_string()}\n")
                f.write(f"Unique values in '{c}': {adp_df[c].dropna().unique()}\n")
        
        f.write(f"\nAccount/Type-like cols: {acct_cols}\n")
        if acct_cols:
            for c in acct_cols:
                 f.write(f"First 10 values of '{c}':\n{adp_df[c].head(10).to_string()}\n")

    except Exception as e:
        f.write(f"Error reading ADP CSV: {e}\n")

    f.write(f"\n\n--- Analyzing Uzio File: {uzio_file} ---\n")
    try:
        # Read Excel
        uzio_df = pd.read_excel(uzio_file, dtype=str)
        f.write(f"Columns: {list(uzio_df.columns)}\n\n")

        # Look for relevant Uzio columns
        prio_cols = [c for c in uzio_df.columns if 'priority' in c.lower()]
        acct_cols = [c for c in uzio_df.columns if 'account' in c.lower() or 'type' in c.lower()]

        f.write(f"Priority-like cols: {prio_cols}\n")
        if prio_cols:
             for c in prio_cols:
                f.write(f"First 10 values of '{c}':\n{uzio_df[c].head(10).to_string()}\n")
                f.write(f"Unique values in '{c}': {uzio_df[c].dropna().unique()}\n")
        
        f.write(f"\nAccount/Type-like cols: {acct_cols}\n")
        if acct_cols:
            for c in acct_cols:
                 f.write(f"First 10 values of '{c}':\n{uzio_df[c].head(10).to_string()}\n")

    except Exception as e:
         f.write(f"Error reading Uzio Excel: {e}\n")

print("Analysis complete. Check raw_data_analysis.txt")
