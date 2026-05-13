import pandas as pd
import numpy as np
import math, os

# Try to load the merged dataset
paths = [
    r'c:\Jon\College\Thesis\merged_clinical_leftjoin.csv',
    r'c:\Jon\College\Thesis\V2.2.1.1\merged_clinical_leftjoin.csv',
]
df = None
for p in paths:
    if os.path.exists(p):
        df = pd.read_csv(p, low_memory=False)
        print(f"Loaded from: {p}, shape: {df.shape}")
        break

if df is None:
    # Try looking in subdirectories
    for root, dirs, files in os.walk(r'c:\Jon\College\Thesis'):
        for f in files:
            if 'merged' in f.lower() and f.endswith('.csv'):
                p = os.path.join(root, f)
                df = pd.read_csv(p, low_memory=False)
                print(f"Loaded from: {p}, shape: {df.shape}")
                break
        if df is not None:
            break

if df is None:
    print("Could not find dataset")
else:
    # Show epwt_fg columns and Total_ columns ranges
    epwt_cols = [c for c in df.columns if c.startswith('epwt_fg')]
    total_cols = [c for c in df.columns if c.startswith('Total_')]
    fg_cols = [c for c in df.columns if c.startswith('fg') and not c.startswith('epwt_fg')]
    
    print(f"\nepwt_fg columns: {epwt_cols}")
    print(f"Total_ columns: {total_cols}")
    print(f"fg_ columns: {fg_cols[:20]}")
    
    for col in epwt_cols + total_cols:
        if col in df.columns:
            col_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(col_data) > 0:
                p95 = col_data.quantile(0.95)
                p99 = col_data.quantile(0.99)
                max_val = col_data.max()
                mean_val = col_data.mean()
                print(f"{col}: mean={mean_val:.2f}, p95={p95:.2f}, p99={p99:.2f}, max={max_val:.2f}")
            else:
                print(f"{col}: No numeric data")
