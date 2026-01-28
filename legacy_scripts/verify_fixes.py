
import pandas as pd
import numpy as np
import sys
import app

# Set encoding
sys.stdout.reconfigure(encoding='utf-8')

with open('verification_results.txt', 'w', encoding='utf-8') as f:
    f.write("--- Verifying 'norm_value' Fixes ---\n")
    
    # 1. Percentage
    val = "20.00 %"
    res = app.norm_value(val, "Deposit Percent")
    f.write(f"Percentage Test: '{val}' -> {res} (Expected: 20.0)\n")
    
    val2 = "99.8"
    res2 = app.norm_value(val2, "Deposit Percent")
    f.write(f"Percentage Test: '{val2}' -> {res2} (Expected: 99.8)\n")

    # 2. Account Type
    acc1 = "CK1 - checking"
    res_acc1 = app.norm_value(acc1, "Account Type")
    f.write(f"Account Type: '{acc1}' -> '{res_acc1}' (Expected: 'checking')\n")
    
    acc2 = "SV2  -  savings"
    res_acc2 = app.norm_value(acc2, "Account Type")
    f.write(f"Account Type: '{acc2}' -> '{res_acc2}' (Expected: 'savings')\n")
    
    acc3 = "Checking"
    res_acc3 = app.norm_value(acc3, "Account Type")
    f.write(f"Account Type: '{acc3}' -> '{res_acc3}' (Expected: 'checking')\n")

    f.write("\n--- Verifying 'normalize_uzio_payment_priority' ---\n")
    
    data = {
        "Employee ID": ["A1", "B2", "B2"],
        "Priority": [np.nan, np.nan, np.nan],
        "Amount": [100, 50, 50]
    }
    df = pd.DataFrame(data)
    f.write("Input DF:\n" + df.to_string() + "\n")
    
    norm_df = app.normalize_uzio_payment_priority(df, "Employee ID")
    f.write("Output DF:\n" + norm_df.to_string() + "\n")
    
    prio_A1 = norm_df.loc[0, "Priority"]
    f.write(f"A1 Priority: {prio_A1} (Expected: 1)\n")
    prio_B2_0 = norm_df.loc[1, "Priority"]
    f.write(f"B2 (row 1) Priority: {prio_B2_0} (Expected: '' or NaN)\n")

print("Verification complete. Check verification_results.txt")
