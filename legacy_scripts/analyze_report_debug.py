import pandas as pd
import sys

# Set encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

file_path = r'c:\Users\shobhit.sharma\Downloads\Uzio-ADP-Payment-and-Emergency-Data-Review-main\Uzio-ADP-Payment-and-Emergency-Data-Review-main\UZIO_vs_ADP_Payment_Report_ADP (1).xlsx'

with open('analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Reading file: {file_path}...\n")
    try:
        xls = pd.ExcelFile(file_path)
        f.write('--- Sheets ---\n')
        f.write(str(xls.sheet_names) + '\n')
        
        f.write('\n--- Summary ---\n')
        f.write(pd.read_excel(xls, 'Summary').to_string() + '\n')
        
        f.write('\n--- Field Summary (Non-Zero Mismatches) ---\n')
        fs = pd.read_excel(xls, 'Field_Summary_By_Status')
        cols_of_interest = ['MISMATCH', 'UZIO_MISSING_VALUE', 'ADP_MISSING_VALUE']
        existing_cols = [c for c in cols_of_interest if c in fs.columns]
        
        if existing_cols:
            mismatch_fs = fs[fs[existing_cols].sum(axis=1) > 0]
            f.write(mismatch_fs.to_string() + '\n')
        else:
            f.write("No mismatch columns found in Field Summary.\n")

        f.write('\n--- Mismatch Samples ---\n')
        details = pd.read_excel(xls, 'Comparison_Detail_AllFields')
        mismatches = details[details['ADP_SourceOfTruth_Status'].isin(['MISMATCH', 'UZIO_MISSING_VALUE', 'ADP_MISSING_VALUE'])]
        if not mismatches.empty:
            f.write(mismatches.head(100).to_string() + '\n')
        else:
            f.write("No mismatches found in Comparison Detail.\n")

    except Exception as e:
        f.write(f"Error: {e}\n")
        print(f"Error: {e}")

print("Analysis complete. Check analysis_results.txt")
