import pandas as pd
import sys

# Set encoding
sys.stdout.reconfigure(encoding='utf-8')

uzio_file = r'c:\Users\shobhit.sharma\Downloads\Uzio-ADP-Payment-and-Emergency-Data-Review-main\Uzio-ADP-Payment-and-Emergency-Data-Review-main\KDL_Payment_Data_Uzio.xlsx'

with open('uzio_structure.txt', 'w', encoding='utf-8') as f:
    f.write(f"--- Inspecting Uzio File: {uzio_file} ---\n")
    try:
        # Read first 10 rows without header
        df = pd.read_excel(uzio_file, header=None, nrows=10, dtype=str)
        f.write(df.to_string())
    except Exception as e:
        f.write(f"Error: {e}")

print("Inspection complete. Check uzio_structure.txt")
