import pandas as pd
import io
import openpyxl
from app import run_comparison

# Create a dummy Excel file in memory with necessary sheets
output = io.BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    # 1. Uzio Data
    # EMP001: Match
    # EMP002: Value missing in ADP
    # EMP003: Value missing in Uzio
    # EMP004: Mismatch
    # EMP005: Only in Uzio
    uzio_data = pd.DataFrame({
        'Employee ID': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
        'First Name': ['John', 'Jane', '', 'Bob', 'Alice'],
        # Add required columns for payment/emergency filter
        'Routing Number': ['123456789'] * 5, 
        'Account Number': ['111'] * 5,
        'Paycheck Distribution': ['Amount'] * 5,
        'Paycheck Percentage': [''] * 5,
        'Paycheck Amount': ['100'] * 5,
    })
    uzio_data.to_excel(writer, sheet_name='Uzio Data', index=False)

    # 2. ADP Payment Data
    # EMP006: Only in ADP
    adp_data = pd.DataFrame({
        'Associate ID': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP006'],
        'First Name': ['John', '', 'Jane', 'Robert', 'Eve'],
        # Add required columns for normalization
        'Deposit Type': ['Amount'] * 5,
        'Deposit Amount': ['100'] * 5,
        'Deposit Percent': [''] * 5,
        'Deposit Percentage': [''] * 5,
        'Priority #': ['1'] * 5,
        'Routing Number': ['123456789'] * 5,
        'Account Number': ['111'] * 5,
    })
    adp_data.to_excel(writer, sheet_name='ADP Payment Data', index=False)

    # 3. ADP Emergency Contact Data (Required by app, but can be empty/minimal)
    adp_ec = pd.DataFrame({
        'Associate ID': [],
        'Contact Name': [],
        'Mobile Phone': [],
        'Relationship': []
    })
    adp_ec.to_excel(writer, sheet_name='ADP Emergency Contact Data', index=False)

    # 4. Payment_Mapping
    mapping = pd.DataFrame({
        'UZIO Column': ['First Name', 'Routing Number', 'Account Number'],
        'ADP Column': ['First Name', 'Routing Number', 'Account Number']
    })
    mapping.to_excel(writer, sheet_name='Payment_Mapping', index=False)

    # 5. Emergency_Mapping (Required)
    ec_mapping = pd.DataFrame({
        'UZIO Column': [],
        'ADP Column': []
    })
    ec_mapping.to_excel(writer, sheet_name='Emergency_Mapping', index=False)

output.seek(0)
file_bytes = output.getvalue()

try:
    results = run_comparison(file_bytes)
    print("Run comparison successful.")
    
    # Parse the output payment report to check status strings
    payment_report_bytes = results['payment']
    payment_xls = pd.ExcelFile(io.BytesIO(payment_report_bytes))
    
    details = pd.read_excel(payment_xls, sheet_name='Comparison_Detail_AllFields')
    statuses = details['ADP_SourceOfTruth_Status'].unique()
    
    print("\nFound Statuses:")
    for s in statuses:
        print(f"- {s}")

    expected_statuses = {
        "Data Match",
        "Data Mismatch",
        "Value missing in Uzio (ADP has value)",
        "Value missing in ADP (Uzio has value)",
        "Employee ID Not Found in ADP",
        "Employee ID Not Found in Uzio"
    }
    
    found_unexpected = False
    for s in statuses:
        if s not in expected_statuses:
             # Ignore these if they didn't occur in our dummy data, 
             # but flagging anything that LOOKS like the old statuses
             if "MISSING" in s.upper() and "ADP_COLUMN" in s: # Old style
                 print(f"ERROR: Found old status format: {s}")
                 found_unexpected = True
             if s == "OK":
                 print(f"ERROR: Found old status format: {s}")
                 found_unexpected = True
                 
    # Verify specific cases
    # EMP001 John vs John -> Data Match
    row_001 = details[details['Employee ID'] == 'EMP001']
    if not row_001.empty:
        status = row_001.iloc[0]['ADP_SourceOfTruth_Status']
        if status != "Data Match":
             print(f"FAILURE: EMP001 expected 'Data Match', got '{status}'")
        else:
            print("SUCCESS: EMP001 -> Data Match")

    # EMP004 Bob vs Robert -> Data Mismatch
    row_004 = details[(details['Employee ID'] == 'EMP004') & (details['Field'] == 'First Name')]
    if not row_004.empty:
        status = row_004.iloc[0]['ADP_SourceOfTruth_Status']
        if status != "Data Mismatch":
             print(f"FAILURE: EMP004 expected 'Data Mismatch', got '{status}'")
        else:
            print("SUCCESS: EMP004 -> Data Mismatch")
            
    # EMP005 (Only Uzio) -> Employee ID Not Found in ADP
    # Note: Logic in app.py handles 'only in Uzio' at the row generation level.
    # It seems to generate rows for all keys in union.
    row_005 = details[details['Employee ID'] == 'EMP005']
    if not row_005.empty:
        # It's likely found in Uzio but not ADP, so 'Employee ID Not Found in ADP'
        # UNLESS the mapping loop didn't find an ADP index.
        status = row_005.iloc[0]['ADP_SourceOfTruth_Status']
        if status != "Employee ID Not Found in ADP":
             print(f"FAILURE: EMP005 expected 'Employee ID Not Found in ADP', got '{status}'")
        else:
            print("SUCCESS: EMP005 -> Employee ID Not Found in ADP")

    # EMP006 (Only ADP) -> Employee ID Not Found in Uzio
    row_006 = details[details['Employee ID'] == 'EMP006']
    if not row_006.empty:
        status = row_006.iloc[0]['ADP_SourceOfTruth_Status']
        if status != "Employee ID Not Found in Uzio":
             print(f"FAILURE: EMP006 expected 'Employee ID Not Found in Uzio', got '{status}'")
        else:
            print("SUCCESS: EMP006 -> Employee ID Not Found in Uzio")

except Exception as e:
    print(f"Comparison failed with error: {e}")
    import traceback
    traceback.print_exc()
