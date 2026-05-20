# Generated from: EXP 0 Create Merge Dataset.ipynb
# Converted at: 2026-05-20T02:09:16.289Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # EXP 0 Create Merge Dataset
# 
# Creates `merged_clinical_leftjoin.csv` from 2015 datasets using a left join anchored on the clinical dataset.
# Join keys: `hhnum`, `member_code`.


from pathlib import Path
import pandas as pd

# Resolve root so this notebook works in FINAL or its parent context.
CWD = Path.cwd()
if (CWD / 'Datasets2015').exists():
    ROOT = CWD
elif (CWD / 'FINAL' / 'Datasets2015').exists():
    ROOT = CWD / 'FINAL'
elif (CWD.parent / 'Datasets2015').exists():
    ROOT = CWD.parent
else:
    raise FileNotFoundError(
        "Could not locate Datasets2015. Run this notebook from FINAL or its parent folder."
    )
DS = ROOT / 'Datasets2015'

clinical_path = next((DS / 'Clinical').glob('*data-set*.csv'))
dietary_path = next((DS / 'Dietary').glob('*data-set*.csv'))
anthro_path = next((DS / 'Anthropometric').glob('*data-set*.csv'))

clinical = pd.read_csv(clinical_path, low_memory=False)
dietary = pd.read_csv(dietary_path, low_memory=False)
anthro = pd.read_csv(anthro_path, low_memory=False)

requested_keys = ['hhnum', 'member_code']

if 'hhnum' not in clinical.columns:
    raise KeyError("Clinical dataset must contain 'hhnum'.")

# Anthropometric has both IDs in this project and can be joined at person level.
anthro_join_keys = [k for k in requested_keys if k in anthro.columns and k in clinical.columns]
if len(anthro_join_keys) < 2:
    raise KeyError("Anthropometric dataset must contain both 'hhnum' and 'member_code'.")

# Dietary is household-level in this dataset; fallback to hhnum when member_code is absent.
dietary_join_keys = [k for k in requested_keys if k in dietary.columns and k in clinical.columns]
if not dietary_join_keys:
    raise KeyError("Dietary dataset must contain at least 'hhnum' for joining.")

# Deduplicate right-hand datasets on their actual join granularity.
dietary = dietary.drop_duplicates(subset=dietary_join_keys, keep='first')
anthro = anthro.drop_duplicates(subset=anthro_join_keys, keep='first')

dietary_overlap = [c for c in dietary.columns if c in clinical.columns and c not in dietary_join_keys]
if dietary_overlap:
    dietary = dietary.rename(columns={c: f'{c}_dietary' for c in dietary_overlap})

merged = clinical.merge(dietary, on=dietary_join_keys, how='left')

anthro_overlap = [c for c in anthro.columns if c in merged.columns and c not in anthro_join_keys]
if anthro_overlap:
    anthro = anthro.rename(columns={c: f'{c}_anthro' for c in anthro_overlap})

merged = merged.merge(anthro, on=anthro_join_keys, how='left')

out_path = ROOT / 'merged_clinical_leftjoin.csv'
merged.to_csv(out_path, index=False)

print(f'Using data root: {ROOT}')
print(f'Clinical source: {clinical_path.name}')
print(f'Dietary source: {dietary_path.name}')
print(f'Anthropometric source: {anthro_path.name}')
print(f'Dietary join keys used: {dietary_join_keys}')
print(f'Anthropometric join keys used: {anthro_join_keys}')
print(f'Wrote: {out_path}')
print(f'Rows: {len(merged):,} | Columns: {merged.shape[1]:,}')