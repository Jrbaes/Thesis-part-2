import pandas as pd
import os

csv_path = 'merged_clinical_dietary_anthro_leftjoin.csv'
if not os.path.exists(csv_path):
    csv_path = 'Thesis-part-2/merged_clinical_dietary_anthro_leftjoin.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {csv_path}")
    sentinels = [7777777, 8888888, 9999999, 77777, 88888, 99999, 7777, 8888, 9999]
    cols_with_sentinels = {}
    for col in df.select_dtypes(include=['number']).columns:
        found = [s for s in sentinels if (df[col] == s).any()]
        if found:
             cols_with_sentinels[col] = found
    
    if cols_with_sentinels:
        for col, vals in cols_with_sentinels.items():
            print(f"{col}: {vals}")
    else:
        print("No sentinel values found.")
else:
    print("CSV file not found")
