import pandas as pd
import sys

# Set encoding
sys.stdout.reconfigure(encoding='utf-8')

uzio_file = r'c:\Users\shobhit.sharma\Downloads\Uzio-ADP-Payment-and-Emergency-Data-Review-main\Uzio-ADP-Payment-and-Emergency-Data-Review-main\KDL_Payment_Data_Uzio.xlsx'

with open('check_priority.txt', 'w', encoding='utf-8') as f:
    try:
        # Read with header=1 (0-indexed, so second row)
        df = pd.read_excel(uzio_file, header=1, dtype=str)
        
        if 'Priority' in df.columns:
            valid_prio = df['Priority'].dropna()
            non_blank = valid_prio[valid_prio.str.strip() != ""]
            
            f.write(f"Total rows: {len(df)}\n")
            f.write(f"Non-blank Priority count: {len(non_blank)}\n")
            if len(non_blank) > 0:
                f.write(f"Sample values: {non_blank.head().tolist()}\n")
            else:
                f.write("Priority column exists but is completely empty/blank.\n")
        else:
            f.write("Priority column NOT found with header=1.\n")
            f.write(f"Columns found: {list(df.columns)}\n")

    except Exception as e:
        f.write(f"Error: {e}")

print("Check complete. See check_priority.txt")
