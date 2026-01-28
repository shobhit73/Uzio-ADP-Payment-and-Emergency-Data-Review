
import pandas as pd
import numpy as np
import sys
import app

sys.stdout.reconfigure(encoding='utf-8')

with open('verification_results_strict_percent.txt', 'w', encoding='utf-8') as f:
    f.write("--- Verifying '_safe_percentage' Logic ---\n")
    
    def check(val, expected):
        res = app._safe_percentage(val)
        status = "PASS" if (np.isnan(res) and np.isnan(expected)) or (abs(res - expected) < 0.001) else "FAIL"
        f.write(f"Input: {val} ({type(val).__name__}) -> {res} (Expected: {expected}) : {status}\n")

    # 1. String with %
    check("20%", 20.0)
    check("20.00 %", 20.0)
    
    # 2. String pure number
    check("20", 20.0) # > 1.0, keeps as is
    check("0.2", 20.0) # <= 1.0, scales x100
    check("3", 3.0)   # > 1.0, keeps as 3 (3%)? Wait. 
                      # If input is "3", is it 3% or 300%? 3% usually.
                      # My logic: 3 > 1.0, so returns 3. Correct.
    check("0.03", 3.0) # <= 1.0, scales x100 -> 3. Correct.

    # 3. Float inputs (Excel)
    check(0.2, 20.0)  # <= 1.0 -> 20. Correct.
    check(20.0, 20.0) # > 1.0 -> 20. Correct.
    check(0.03, 3.0)  # Correct.
    check(1.0, 100.0) # 1.0 is 100% in Excel. Returns 100. Correct.
    
    # 4. Edge cases
    check(0.0, 0.0)   # 0 -> 0. (Not <= 1.0 because 0 not in (0, 1.0]? logic: 0 < abs(f) <= 1.0)
                      # If 0, returns 0. Correct.
    check(1.01, 1.01) # Keeps as 1.01%? Or is it 101%?
                      # If > 1.0, keeps as is. So 1.01 comes out as 1.01.
                      # If user meant 101% (1.01 in Excel), this logic fails.
                      # But for PAYROLL, split <= 100%. So > 1.0 usually means literal percent "50".
                      # 1.01 literal percent is 1.01%. 
                      # It is indistinguishable from 101% in terms of " > 1.0".
                      # But payroll splits rarely exceed 100%.
                      
    f.write("\n--- Verifying Logic Flow ---\n")
    # Simulate a partial row
    # If 0.2 comes in, it becomes 20.
    # sum = 20.
    # Full = 100 - 20 = 80.
    # Previously: 0.2 comes in -> 0.2. Full = 99.8.
    
    val = app._safe_percentage(0.2)
    full = 100.0 - val
    f.write(f"Input 0.2 -> Scaled {val}. Full Remainder: {full} (Should be 80)\n")
    
print("Verification complete. See verification_results_strict_percent.txt")
