
import pandas as pd
import numpy as np
import sys
import app

# Set encoding
sys.stdout.reconfigure(encoding='utf-8')

with open('verification_results_fuzzy.txt', 'w', encoding='utf-8') as f:
    f.write("--- Verifying 'is_fuzzy_match' Logic ---\n")
    
    # helper
    def check(v1, v2, expect_bool):
        res = app.is_fuzzy_match(v1, v2)
        status = "PASS" if res == expect_bool else "FAIL"
        f.write(f"Match({v1}, {v2}) -> {res} (Expect {expect_bool}) : {status}\n")

    # 1. Exact strings
    check("checking", "checking", True)
    check("Checking", "checking", False) # is_fuzzy_match expects normalized input? Yes, called after norm_value
    
    # 2. Float exact
    check(20.0, 20.0, True)
    check(20, 20.0, True)
    
    # 3. Tolerance
    check(99.8, 99.8000001, True)
    check(99.8, 99.81, True) # diff is 0.01. Strict < 0.01 fails? abs(0.01) < 0.01 is False. 99.81 - 99.8 = 0.01000...
    
    # 4. Scaling (The main fix)
    check(20, 0.2, True)      # Uzio=20, ADP=0.2
    check(0.2, 20, True)      # Uzio=0.2, ADP=20
    check(3, 0.03, True)      # Uzio=3, ADP=0.03
    check(0.03, 3, True)

    # 5. Non-matches
    check(20, 0.5, False)
    check("foo", "bar", False)
    check(20, "", False)

print("Verification complete. See verification_results_fuzzy.txt")
