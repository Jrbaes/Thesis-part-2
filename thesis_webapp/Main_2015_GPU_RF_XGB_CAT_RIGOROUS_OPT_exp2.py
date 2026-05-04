#!/usr/bin/env python
# coding: utf-8

# # Main 2015 Rigorous GPU Notebook (RF + XGBoost + CatBoost)
# 
# This notebook mirrors the exp1 preprocessing strategy and is optimized for **accuracy + recall** during hyperparameter search.
# 
# Workflow:
# 1. Data loading and target inference
# 2. EDA and manual dropping
# 3. Collinearity heatmap after manual drops
# 4. KNN imputation + standardization + one-hot encoding
# 5. Collinearity-driven dropping and post-drop heatmap (colors only)
# 6. Rigorous GPU-first optimization for Random Forest, XGBoost, CatBoost
# 7. Calibration with base, Platt, Isotonic, Venn-Abers
# 8. Final explanation and interpretation

# In[1]:


# Optional install (uncomment if needed), then restart kernel once.
get_ipython().run_line_magic('pip', 'install -q numpy pandas scipy scikit-learn joblib xgboost catboost venn-abers seaborn matplotlib imbalanced-learn torch')
get_ipython().run_line_magic('pip', 'install -q cuml-cu13 --extra-index-url=https://pypi.nvidia.com')


# In[2]:


import torch
print("CUDA available:", torch.cuda.is_available())


# In[3]:


import json
import random
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import randint, uniform, loguniform

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 300)

xgb_available = True
cat_available = True
venn_available = True
torch_available = True
torch_cuda_available = False
cuml_available = True

try:
    from xgboost import XGBClassifier
except Exception:
    xgb_available = False
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    cat_available = False
    CatBoostClassifier = None

try:
    from venn_abers import VennAbers
except Exception:
    venn_available = False
    VennAbers = None

try:
    import torch
    torch_cuda_available = bool(torch.cuda.is_available())
except Exception:
    torch_available = False

try:
    from cuml.ensemble import RandomForestClassifier as cuRFClassifier
except Exception:
    cuml_available = False
    cuRFClassifier = None

print({
    'xgboost': xgb_available,
    'catboost': cat_available,
    'venn_abers': venn_available,
    'torch': torch_available,
    'torch_cuda': torch_cuda_available,
    'cuml_rf': cuml_available,
})


# In[4]:


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
ARTIFACT_DIR = PROJECT_ROOT / 'gpu_rf_xgb_cat_exp2_artifacts'
MODEL_DIR = ARTIFACT_DIR / 'models'

for p in [ARTIFACT_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DATASET2015_CANDIDATES = [
    PROJECT_ROOT / 'Datasets2015',
    PROJECT_ROOT.parent / 'Datasets2015',
]

AUTO_MERGED_DATA_PATH = None
DATA_CANDIDATES = [
    PROJECT_ROOT / 'merged_clinical_dietary_anthro_leftjoin.csv',
    PROJECT_ROOT / 'merged_clinical_dietary_leftjoin.csv',
    PROJECT_ROOT.parent / 'merged_clinical_dietary_leftjoin.csv',
    PROJECT_ROOT / 'merged_clinical_leftjoin.csv',
    PROJECT_ROOT.parent / 'merged_clinical_leftjoin.csv',
]

TARGET_CANDIDATES = ['hypertension', 'htn', 'target', 'label', 'outcome']
COLLINEARITY_CUTOFF = 0.70

STAGE1_EPOCHS = 120
STAGE2_EPOCHS = 300
FINAL_EPOCHS = 900

STAGE1_TRIALS_PER_MODEL = 180
STAGE2_REFINEMENTS_PER_TOP_CONFIG = 24
TOP_K_STAGE1 = 8
TOP_K_STAGE2 = 3
CV_FOLDS_STAGE1 = 5
CV_FOLDS_STAGE2 = 6

USE_GPU_WHEN_AVAILABLE = True
N_JOBS = -1

print('Artifacts:', ARTIFACT_DIR)


# In[5]:


def _normalize_join_columns(df_in):
    rename_map = {}
    col_lc = {c.lower(): c for c in df_in.columns}
    for key in ['hhnum', 'member_code']:
        if key in col_lc and col_lc[key] != key:
            rename_map[col_lc[key]] = key
    return df_in.rename(columns=rename_map)

def _find_anthropometric_dataset_path():
    for base in DATASET2015_CANDIDATES:
        anthro_dir = base / 'Anthropometric'
        if not anthro_dir.exists():
            continue

        csv_paths = sorted([p for p in anthro_dir.glob('*.csv') if 'dictionary' not in p.name.lower()])
        preferred = [
            p for p in csv_paths
            if ('data-set' in p.name.lower()) or ('dataset' in p.name.lower())
        ]
        for p in preferred + csv_paths:
            return p
    return None

def _prepare_merged_with_anthro(base_path):
    if not base_path.exists():
        return None

    try:
        base_df = pd.read_csv(base_path)
    except Exception:
        return None

    anthro_tokens = ['weight', 'height', 'waist', 'hip', 'bmi', 'whr']
    has_anthro = any(any(tok in c.lower() for tok in anthro_tokens) for c in base_df.columns)
    if has_anthro:
        return base_path

    anthro_path = _find_anthropometric_dataset_path()
    if anthro_path is None:
        return None

    try:
        anthro_df = pd.read_csv(anthro_path)
    except Exception:
        return None

    base_df = _normalize_join_columns(base_df)
    anthro_df = _normalize_join_columns(anthro_df)

    join_keys = [k for k in ['hhnum', 'member_code'] if k in base_df.columns and k in anthro_df.columns]
    if not join_keys:
        return None

    anthro_df = anthro_df.drop_duplicates(subset=join_keys, keep='first')

    overlap = [c for c in anthro_df.columns if c in base_df.columns and c not in join_keys]
    if overlap:
        anthro_df = anthro_df.rename(columns={c: f'{c}_anthro' for c in overlap})

    merged_df = base_df.merge(anthro_df, on=join_keys, how='left')
    out_path = PROJECT_ROOT / 'merged_clinical_dietary_anthro_leftjoin.csv'
    merged_df.to_csv(out_path, index=False)

    print(f'Prepared anthropometric-augmented dataset: {out_path}')
    print(f'  source merged file: {base_path}')
    print(f'  source anthropometric file: {anthro_path}')
    print(f'  join keys: {join_keys}')
    return out_path

def resolve_data_path(candidates):
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f'No dataset found. Checked: {candidates}')

def infer_target_column(df, candidates):
    lc = {c.lower(): c for c in df.columns}
    for t in candidates:
        if t.lower() in lc:
            return lc[t.lower()]
    for c in df.columns:
        c_lc = c.lower()
        if any(t.lower() in c_lc for t in candidates):
            return c
    return None

def infer_bp_columns(df):
    sbp_aliases = ['ave_sbp', 'sbp', 'systolic', 'sysbp', 's_bp']
    dbp_aliases = ['ave_dbp', 'dbp', 'diastolic', 'diabp', 'd_bp']

    lc = {c.lower(): c for c in df.columns}

    sbp_col = None
    for a in sbp_aliases:
        if a in lc:
            sbp_col = lc[a]
            break
    if sbp_col is None:
        for c in df.columns:
            c_lc = c.lower()
            if any(a in c_lc for a in sbp_aliases):
                sbp_col = c
                break

    dbp_col = None
    for a in dbp_aliases:
        if a in lc:
            dbp_col = lc[a]
            break
    if dbp_col is None:
        for c in df.columns:
            c_lc = c.lower()
            if any(a in c_lc for a in dbp_aliases):
                dbp_col = c
                break

    return sbp_col, dbp_col

def find_first_column_case_insensitive(columns, candidates):
    lc = {c.lower(): c for c in columns}
    for cand in candidates:
        cand_lc = cand.lower()
        if cand_lc in lc:
            return lc[cand_lc]
    for c in columns:
        c_lc = c.lower()
        if any(cand.lower() in c_lc for cand in candidates):
            return c
    return None

def to_numeric_clean(series):
    s = pd.to_numeric(series, errors='coerce')
    return s.where(~s.isin([9, 99, 888888, 999999]), np.nan)

def build_smoking_level_feature(df_in):
    used_cols = []
    smoking_level_col = find_first_column_case_insensitive(df_in.columns, ['smoking_level'])
    smoke_status_col = find_first_column_case_insensitive(df_in.columns, ['smoke_status'])
    current_smoking_col = find_first_column_case_insensitive(df_in.columns, ['current_smoking', 'currentsmoking'])
    ever_smoke_col = find_first_column_case_insensitive(df_in.columns, ['ever_smk'])

    if smoking_level_col is not None:
        s = to_numeric_clean(df_in[smoking_level_col]).clip(lower=0, upper=3)
        used_cols.append(smoking_level_col)
        return s.astype(float), sorted(set(used_cols))

    idx = df_in.index
    smoke = pd.Series(np.nan, index=idx, dtype=float)

    if smoke_status_col is not None:
        status = to_numeric_clean(df_in[smoke_status_col])
        used_cols.append(smoke_status_col)
        smoke.loc[status == 0] = 0
        smoke.loc[status == 2] = 1
        smoke.loc[status == 1] = 2
        if current_smoking_col is not None:
            current = to_numeric_clean(df_in[current_smoking_col])
            used_cols.append(current_smoking_col)
            smoke.loc[(status == 1) & (current == 3)] = 3
        return smoke.astype(float), sorted(set(used_cols))

    if current_smoking_col is not None:
        current = to_numeric_clean(df_in[current_smoking_col])
        used_cols.append(current_smoking_col)
        smoke.loc[current == 0] = 0
        smoke.loc[current.isin([1, 2])] = 2
        smoke.loc[current == 3] = 3
        if ever_smoke_col is not None:
            ever = to_numeric_clean(df_in[ever_smoke_col])
            used_cols.append(ever_smoke_col)
            smoke.loc[(current == 0) & (ever > 0)] = 1
        return smoke.astype(float), sorted(set(used_cols))

    if ever_smoke_col is not None:
        ever = to_numeric_clean(df_in[ever_smoke_col])
        used_cols.append(ever_smoke_col)
        smoke.loc[ever == 0] = 0
        smoke.loc[ever > 0] = 1
        return smoke.astype(float), sorted(set(used_cols))

    return None, []

def build_alcohol_level_feature(df_in):
    used_cols = []
    alcohol_level_col = find_first_column_case_insensitive(df_in.columns, ['alcohol_level'])
    alcohol_status_col = find_first_column_case_insensitive(df_in.columns, ['alcohol_status'])
    alcohol_ever_col = find_first_column_case_insensitive(df_in.columns, ['alcohol'])
    current_alcohol_col = find_first_column_case_insensitive(df_in.columns, ['con_alcohol'])
    drink30_col = find_first_column_case_insensitive(df_in.columns, ['drnk_30days'])
    binge_col = find_first_column_case_insensitive(df_in.columns, ['binge_drink', 'binge_drinking'])

    if alcohol_level_col is not None:
        a = to_numeric_clean(df_in[alcohol_level_col]).clip(lower=0, upper=3)
        used_cols.append(alcohol_level_col)
        return a.astype(float), sorted(set(used_cols))

    idx = df_in.index
    alcohol = pd.Series(np.nan, index=idx, dtype=float)

    if alcohol_status_col is not None:
        status = to_numeric_clean(df_in[alcohol_status_col])
        used_cols.append(alcohol_status_col)
        alcohol.loc[status == 0] = 0
        alcohol.loc[status == 2] = 1
        alcohol.loc[status == 1] = 2
        if binge_col is not None:
            binge = to_numeric_clean(df_in[binge_col])
            used_cols.append(binge_col)
            alcohol.loc[(status == 1) & (binge == 1)] = 3
        return alcohol.astype(float), sorted(set(used_cols))

    alcohol.loc[:] = 0
    if alcohol_ever_col is not None:
        ever = to_numeric_clean(df_in[alcohol_ever_col])
        used_cols.append(alcohol_ever_col)
        alcohol.loc[ever > 0] = 1
    if current_alcohol_col is not None:
        current = to_numeric_clean(df_in[current_alcohol_col])
        used_cols.append(current_alcohol_col)
        alcohol.loc[current == 1] = np.maximum(alcohol.loc[current == 1], 2)
    if drink30_col is not None:
        d30 = to_numeric_clean(df_in[drink30_col])
        used_cols.append(drink30_col)
        alcohol.loc[d30 == 1] = np.maximum(alcohol.loc[d30 == 1], 2)
    if binge_col is not None:
        binge = to_numeric_clean(df_in[binge_col])
        used_cols.append(binge_col)
        alcohol.loc[binge == 1] = 3

    if used_cols:
        return alcohol.astype(float), sorted(set(used_cols))

    return None, []

def build_bmi_feature(df_in):
    weight_col = find_first_column_case_insensitive(df_in.columns, ['weight'])
    height_col = find_first_column_case_insensitive(df_in.columns, ['height'])
    if weight_col is None or height_col is None:
        return None, []
    w = pd.to_numeric(df_in[weight_col], errors='coerce')
    h = pd.to_numeric(df_in[height_col], errors='coerce')
    h_m = h.copy()
    if pd.notna(h_m.median(skipna=True)) and float(h_m.median(skipna=True)) > 3.0:
        h_m = h_m / 100.0
    bmi = w / (h_m ** 2)
    bmi = bmi.replace([np.inf, -np.inf], np.nan)
    return bmi.astype(float), [weight_col, height_col]

def build_whr_feature(df_in):
    waist_col = find_first_column_case_insensitive(df_in.columns, ['waist'])
    hip_col = find_first_column_case_insensitive(df_in.columns, ['hip'])
    if waist_col is None or hip_col is None:
        return None, []
    waist = pd.to_numeric(df_in[waist_col], errors='coerce')
    hip = pd.to_numeric(df_in[hip_col], errors='coerce').replace(0, np.nan)
    whr = (waist / hip).replace([np.inf, -np.inf], np.nan)
    return whr.astype(float), [waist_col, hip_col]


# In[6]:


AUTO_MERGED_DATA_PATH = None
for merged_candidate in [
    PROJECT_ROOT / 'merged_clinical_dietary_leftjoin.csv',
    PROJECT_ROOT.parent / 'merged_clinical_dietary_leftjoin.csv',
]:
    AUTO_MERGED_DATA_PATH = _prepare_merged_with_anthro(merged_candidate)
    if AUTO_MERGED_DATA_PATH is not None:
        break

effective_candidates = [
    AUTO_MERGED_DATA_PATH,
    *DATA_CANDIDATES,
]
effective_candidates = [p for p in effective_candidates if p is not None]

data_path = resolve_data_path(effective_candidates)
df = pd.read_csv(data_path)
target_col = infer_target_column(df, TARGET_CANDIDATES)
TARGET_DEFINED_FROM_BP = False
TARGET_SOURCE_COLUMNS = []

if target_col is None:
    sbp_col, dbp_col = infer_bp_columns(df)
    if sbp_col is not None and dbp_col is not None:
        sbp = pd.to_numeric(df[sbp_col], errors='coerce')
        dbp = pd.to_numeric(df[dbp_col], errors='coerce')
        df['Hypertension'] = (((sbp >= 130) | (dbp >= 80)).fillna(False)).astype(int)
        target_col = 'Hypertension'
        TARGET_DEFINED_FROM_BP = True
        TARGET_SOURCE_COLUMNS = [sbp_col, dbp_col]
        print(f'Target column created from: {sbp_col}, {dbp_col}')
    else:
        raise ValueError('Could not infer target and could not derive Hypertension from SBP/DBP (130/80 OR rule).')

df = df.dropna(subset=[target_col]).copy()
y_raw = df[target_col]
if y_raw.nunique() != 2:
    raise ValueError(f'Target must be binary. Found {y_raw.nunique()} classes.')

if y_raw.dtype == 'O':
    y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), index=y_raw.index, name=target_col)
else:
    y = pd.Series(y_raw.astype(int), index=y_raw.index, name=target_col)

X = df.drop(columns=[target_col]).copy()

smoking_feature, smoking_sources = build_smoking_level_feature(X)
if smoking_feature is not None:
    X['fe_smoking_level'] = smoking_feature

alcohol_feature, alcohol_sources = build_alcohol_level_feature(X)
if alcohol_feature is not None:
    X['fe_alcohol_level'] = alcohol_feature

bmi_feature, bmi_sources = build_bmi_feature(X)
if bmi_feature is not None:
    X['bmi'] = bmi_feature

whr_feature, whr_sources = build_whr_feature(X)
if whr_feature is not None:
    X['whr'] = whr_feature

behavior_raw_candidates = [
    'current_smoking', 'currentsmoking', 'ever_smk', 'smoke_status', 'smoking_level',
    'alcohol', 'con_alcohol', 'drnk_30days', 'drnk_30d_num', 'alcohol_status',
    'binge_drink', 'binge_drinking', 'alcohol_level',
]
x_lc = {c.lower(): c for c in X.columns}
behavior_drop = sorted({x_lc[c.lower()] for c in behavior_raw_candidates if c.lower() in x_lc})
if behavior_drop:
    X = X.drop(columns=behavior_drop, errors='ignore')

anthro_source_drop = sorted({
    c for c in set((bmi_sources or []) + (whr_sources or []))
    if c in X.columns and c.lower() not in {'bmi', 'whr'}
})
if anthro_source_drop:
    X = X.drop(columns=anthro_source_drop, errors='ignore')

NON_REMOVABLE_BASE_ALIASES = ['age', 'sex']

manual_non_predictive = [
    'regcode', 'provcode', 'provhuc', 'psc', 'csc', 'rhc', 'psurec', 'strrec',
    'wgts', 'fwgt', 'finalwgt', 'finalwgt1', 'finalwgt4',
    'fwgth_natl_var', 'fwgth_prov', 'fwgth_natl2_var',
    'fwgti_natl_var', 'fwgti_prov', 'fwgti_natl2_var', 'fwgti_prov2',
    'rep_natl', 'rep_prov', 'ms_psucode', 'enns_year', 'wrkplace',
    'interview_status', 'intdate', 'enumcode',
    'hhnum', 'member_code',
    'ave_sbp', 'ave_dbp', 'sbp', 'dbp', 'systolic', 'diastolic', 'sysbp', 'diabp',
    'blood_pressure',
    'height', 'weight', 'waist', 'hip',
]
x_lc = {c.lower(): c for c in X.columns}
manual_drop = sorted({x_lc[c.lower()] for c in manual_non_predictive if c.lower() in x_lc})

protected_base_cols = []
for col in X.columns:
    col_lc = col.lower()
    if any(alias in col_lc for alias in NON_REMOVABLE_BASE_ALIASES):
        protected_base_cols.append(col)
protected_base_cols = sorted(set(protected_base_cols))

if manual_drop:
    protected_drop = [c for c in manual_drop if c in protected_base_cols]
    if protected_drop:
        print('Manual-drop protection triggered for non-removable base features:', protected_drop)
    manual_drop = [c for c in manual_drop if c not in protected_base_cols]
    X = X.drop(columns=manual_drop, errors='ignore')

base_removed_non_removable = []
for alias in NON_REMOVABLE_BASE_ALIASES:
    if not any(alias in c.lower() for c in X.columns):
        base_removed_non_removable.append(alias)

RETRAIN_REQUIRED_NON_REMOVABLE = len(base_removed_non_removable) > 0

print(f'Loaded: {data_path}')
print(f'Auto merged dataset: {AUTO_MERGED_DATA_PATH}')
print(f'Target: {target_col}')
print(f'Target defined from BP fallback: {TARGET_DEFINED_FROM_BP}')
print(f'Raw rows: {len(df)}')
print(f'Features after manual preprocessing: {X.shape[1]}')
print(f'Manual dropped columns: {len(manual_drop)}')
print(f'Behavior raw dropped columns: {len(behavior_drop)}')
print(f'Anthropometric source dropped columns: {len(anthro_source_drop)}')
print('Protected non-removable base columns:', protected_base_cols)
print('Missing non-removable base aliases after base preprocessing:', base_removed_non_removable)
print('Class balance:', y.value_counts(normalize=True).to_dict())


# In[13]:


# Rigorous EDA block (paper-ready): target relationships + artifact folders
EDA_DIR = ARTIFACT_DIR / 'eda'
EDA_IMG_DIR = EDA_DIR / 'images'
EDA_TABLE_DIR = EDA_DIR / 'tables'
for p in [EDA_DIR, EDA_IMG_DIR, EDA_TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print(f'EDA directory: {EDA_DIR}')
print(f'Image output directory: {EDA_IMG_DIR}')
print(f'Table output directory: {EDA_TABLE_DIR}')
print(f'Sample size: {len(X):,}')
print('Target distribution (count):')
print(y.value_counts(dropna=False).to_string())
print('Target distribution (proportion):')
print(y.value_counts(normalize=True, dropna=False).round(4).to_string())

target_count_df = (
    y.value_counts(dropna=False)
    .rename_axis('target')
    .reset_index(name='count')
    .sort_values('target')
    .reset_index(drop=True)
)
target_count_df['label'] = target_count_df['target'].map({0: 'No HTN', 1: 'HTN'}).fillna(target_count_df['target'].astype(str))
target_count_df['proportion'] = target_count_df['count'] / target_count_df['count'].sum()
target_count_df['percent'] = target_count_df['proportion'] * 100.0

target_count_path = EDA_TABLE_DIR / 'target_distribution_counts.csv'
target_count_df.to_csv(target_count_path, index=False)
print(f'Saved target distribution table: {target_count_path}')
print(target_count_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bar_colors = ['#1976d2', '#c62828']

axes[0].bar(target_count_df['label'], target_count_df['count'], color=bar_colors[:len(target_count_df)])
axes[0].set_title('Target Distribution (Counts)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)
for idx, row in target_count_df.iterrows():
    axes[0].text(idx, row['count'], f"{int(row['count']):,}", ha='center', va='bottom', fontsize=10)

axes[1].bar(target_count_df['label'], target_count_df['percent'], color=bar_colors[:len(target_count_df)])
axes[1].set_title('Target Distribution (Percent)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Percent')
axes[1].grid(axis='y', alpha=0.3)
for idx, row in target_count_df.iterrows():
    axes[1].text(idx, row['percent'], f"{row['percent']:.2f}%", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
target_dist_fig_path = EDA_IMG_DIR / 'target_distribution_counts_and_percent.png'
plt.savefig(target_dist_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved target distribution figure: {target_dist_fig_path}')

num_cols_eda = X.select_dtypes(include=[np.number]).columns.tolist()
print(f'Numeric features for target-association analysis: {len(num_cols_eda)}')

def _safe_smd(x0: pd.Series, x1: pd.Series) -> float:
    x0 = pd.to_numeric(x0, errors='coerce').dropna()
    x1 = pd.to_numeric(x1, errors='coerce').dropna()
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    m0, m1 = x0.mean(), x1.mean()
    v0, v1 = x0.var(ddof=1), x1.var(ddof=1)
    pooled = np.sqrt((v0 + v1) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return float((m1 - m0) / pooled)

target_numeric = pd.to_numeric(y, errors='coerce')
rows = []
for c in num_cols_eda:
    s = pd.to_numeric(X[c], errors='coerce')
    valid = s.notna() & target_numeric.notna()
    n_valid = int(valid.sum())
    if n_valid < 5:
        continue

    corr_spearman = s[valid].corr(target_numeric[valid], method='spearman')
    grp0 = s[valid & (target_numeric == 0)]
    grp1 = s[valid & (target_numeric == 1)]

    rows.append(
        {
            'feature': c,
            'n_valid': n_valid,
            'spearman_corr_target': float(corr_spearman) if pd.notna(corr_spearman) else np.nan,
            'abs_spearman_corr_target': abs(float(corr_spearman)) if pd.notna(corr_spearman) else np.nan,
            'mean_target0': float(grp0.mean()) if len(grp0) else np.nan,
            'mean_target1': float(grp1.mean()) if len(grp1) else np.nan,
            'std_mean_diff_target1_vs_0': _safe_smd(grp0, grp1),
        }
    )

target_relation_df = pd.DataFrame(rows).sort_values('abs_spearman_corr_target', ascending=False).reset_index(drop=True)
target_relation_path = EDA_TABLE_DIR / 'target_relationship_numeric_features.csv'
target_relation_df.to_csv(target_relation_path, index=False)

print(f'Saved target relationship table: {target_relation_path}')
print('Top 20 numeric features by |Spearman correlation| with target:')
print(target_relation_df[['feature', 'spearman_corr_target', 'abs_spearman_corr_target', 'std_mean_diff_target1_vs_0']].head(20).to_string(index=False))

if not target_relation_df.empty:
    plot_df = target_relation_df.head(20).iloc[::-1].copy()
    plt.figure(figsize=(10, 8))
    colors = ['#c62828' if v >= 0 else '#1565c0' for v in plot_df['spearman_corr_target']]
    plt.barh(plot_df['feature'], plot_df['spearman_corr_target'], color=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.title('Top Numeric Variable Relationships with Target (Spearman)')
    plt.xlabel('Spearman Correlation with Target')
    plt.ylabel('Feature')
    plt.tight_layout()
    target_rel_fig_path = EDA_IMG_DIR / 'target_relationship_top20_spearman.png'
    plt.savefig(target_rel_fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved target-relationship figure: {target_rel_fig_path}')


# In[8]:


# Supplementary EDA heatmap after manual dropping (colors only, no correlation values).
if 'EDA_DIR' not in globals():
    EDA_DIR = ARTIFACT_DIR / 'eda'
    EDA_IMG_DIR = EDA_DIR / 'images'
    EDA_TABLE_DIR = EDA_DIR / 'tables'
for p in [EDA_DIR, EDA_IMG_DIR, EDA_TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

num_cols_manual = X.select_dtypes(include=[np.number]).columns.tolist()
print('Numeric columns for supplementary manual-drop heatmap:', len(num_cols_manual))

if len(num_cols_manual) > 1:
    corr_manual = X[num_cols_manual].corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_manual, cmap='coolwarm', center=0, annot=False, cbar=True)
    plt.title('Correlation Heatmap After Manual Dropping (Colors Only)')
    plt.tight_layout()
    manual_heatmap_path = EDA_IMG_DIR / 'heatmap_after_manual_dropping.png'
    plt.savefig(manual_heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved supplementary manual-drop heatmap: {manual_heatmap_path}')
else:
    print('Not enough numeric columns to plot manual-drop heatmap.')


# In[15]:


X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_tmp
)

split_target_rows = []
for split_name, y_split in [('Train', y_train), ('Validation', y_valid), ('Test', y_test)]:
    split_counts = y_split.value_counts(dropna=False).sort_index()
    split_total = int(split_counts.sum())
    for target_value, count in split_counts.items():
        split_target_rows.append(
            {
                'split': split_name,
                'target': target_value,
                'label': {0: 'No HTN', 1: 'HTN'}.get(target_value, str(target_value)),
                'count': int(count),
                'percent': float(count / split_total * 100.0),
            }
        )

split_target_df = pd.DataFrame(split_target_rows)
split_target_path = EDA_TABLE_DIR / 'target_distribution_by_split.csv'
split_target_df.to_csv(split_target_path, index=False)
print(f'Saved split target table: {split_target_path}')
print(split_target_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=split_target_df, x='split', y='count', hue='label', palette=['#1976d2', '#c62828'], ax=axes[0])
axes[0].set_title('Target Counts by Data Split', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Split')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.0f', padding=2, fontsize=9)

sns.barplot(data=split_target_df, x='split', y='percent', hue='label', palette=['#1976d2', '#c62828'], ax=axes[1])
axes[1].set_title('Target Percent by Data Split', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Split')
axes[1].set_ylabel('Percent')
axes[1].grid(axis='y', alpha=0.3)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.2f%%', padding=2, fontsize=9)

handles, labels = axes[1].get_legend_handles_labels()
axes[0].legend_.remove()
axes[1].legend(handles, labels, title='Class', loc='upper right')
plt.tight_layout()
split_fig_path = EDA_IMG_DIR / 'target_distribution_by_split.png'
plt.savefig(split_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved split target figure: {split_fig_path}')

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

numeric_pipe = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
])

categorical_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', make_ohe()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipe, num_cols),
        ('cat', categorical_pipe, cat_cols),
    ],
    remainder='drop',
)

X_train_proc = preprocessor.fit_transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)
X_test_proc = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
X_train_proc = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
X_valid_proc = pd.DataFrame(X_valid_proc, columns=feature_names, index=X_valid.index)
X_test_proc = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)

# Ensure EDA folders exist even if the standalone EDA cell was skipped.
if 'EDA_DIR' not in globals():
    EDA_DIR = ARTIFACT_DIR / 'eda'
    EDA_IMG_DIR = EDA_DIR / 'images'
    EDA_TABLE_DIR = EDA_DIR / 'tables'
for p in [EDA_DIR, EDA_IMG_DIR, EDA_TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def high_collinearity_pairs(corr_df: pd.DataFrame, cutoff: float = 0.70) -> pd.DataFrame:
    abs_corr = corr_df.abs()
    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
    pairs = []
    for c in upper.columns:
        rows = upper.index[upper[c] > cutoff].tolist()
        for r in rows:
            pairs.append(
                {
                    'feature_1': r,
                    'feature_2': c,
                    'corr': float(corr_df.loc[r, c]),
                    'abs_corr': float(abs_corr.loc[r, c]),
                }
            )
    if not pairs:
        return pd.DataFrame(columns=['feature_1', 'feature_2', 'corr', 'abs_corr'])
    return pd.DataFrame(pairs).sort_values('abs_corr', ascending=False).reset_index(drop=True)

# Heatmap BEFORE collinearity culling (colors only).
corr_pre_cull = X_train_proc.corr(numeric_only=True)
if corr_pre_cull.shape[1] > 1:
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_pre_cull, cmap='coolwarm', center=0, annot=False, cbar=True)
    plt.title('Correlation Heatmap Before Collinearity Culling (Colors Only)')
    plt.tight_layout()
    pre_heatmap_path = EDA_IMG_DIR / 'heatmap_before_collinearity_culling.png'
    plt.savefig(pre_heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved pre-culling heatmap: {pre_heatmap_path}')
else:
    print('Not enough processed features to plot pre-culling heatmap.')

high_corr_pre_df = high_collinearity_pairs(corr_pre_cull, cutoff=COLLINEARITY_CUTOFF)
high_corr_pre_path = EDA_TABLE_DIR / f'collinearity_pairs_before_culling_gt_{COLLINEARITY_CUTOFF:.2f}.csv'
high_corr_pre_df.to_csv(high_corr_pre_path, index=False)
print(f'Saved pre-culling high-collinearity table: {high_corr_pre_path}')
print(f'Pre-culling |corr| > {COLLINEARITY_CUTOFF:.2f} pair count: {len(high_corr_pre_df)}')
if not high_corr_pre_df.empty:
    print(high_corr_pre_df.head(30).to_string(index=False))

# Protect only canonical age/sex features (not every alias like age_anthro/agemos/sex_anthro).
NON_REMOVABLE_PROCESSED_ALIASES = ['age', 'sex']
canonical_non_removable_processed_cols = [
    c for c in ['num__age', 'num__sex'] if c in X_train_proc.columns
]

# Fallback only if canonical names are absent.
if not canonical_non_removable_processed_cols:
    non_removable_processed_cols = []
    for c in X_train_proc.columns:
        c_lc = c.lower()
        if any(alias in c_lc for alias in NON_REMOVABLE_PROCESSED_ALIASES):
            non_removable_processed_cols.append(c)
    non_removable_processed_cols = sorted(set(non_removable_processed_cols))
else:
    non_removable_processed_cols = sorted(set(canonical_non_removable_processed_cols))

protected_cols = sorted(set([c for c in ['num__bmi', 'num__whr'] if c in X_train_proc.columns] + non_removable_processed_cols))

def collinearity_filter(df_in, cutoff=0.70, protected=None):
    protected_set = set(protected or [])
    corr = df_in.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = []
    for c in upper.columns:
        if c in protected_set:
            continue
        if (upper[c] > cutoff).any():
            drop_cols.append(c)
    keep_cols = [c for c in df_in.columns if c not in drop_cols]
    return keep_cols, drop_cols

keep_cols, dropped_cols = collinearity_filter(
    X_train_proc, cutoff=COLLINEARITY_CUTOFF, protected=protected_cols
)

missing_non_removable_after_filter = [c for c in non_removable_processed_cols if c not in keep_cols]
if missing_non_removable_after_filter:
    keep_cols = keep_cols + missing_non_removable_after_filter
    dropped_cols = [c for c in dropped_cols if c not in set(missing_non_removable_after_filter)]
    RETRAIN_REQUIRED_NON_REMOVABLE = True

X_train_final = X_train_proc[keep_cols].copy()
X_valid_final = X_valid_proc[keep_cols].copy()
X_test_final = X_test_proc[keep_cols].copy()

missing_non_removable_final = []
for c in non_removable_processed_cols:
    if c not in X_train_final.columns:
        missing_non_removable_final.append(c)

if missing_non_removable_final:
    RETRAIN_REQUIRED_NON_REMOVABLE = True
    raise ValueError(
        f'Non-removable processed features missing from final matrix: {missing_non_removable_final}. '
        'Please verify source columns for age/sex are present before training.'
    )

print('Train/Valid/Test shapes:', X_train_final.shape, X_valid_final.shape, X_test_final.shape)
print('Protected cols:', protected_cols)
print('Non-removable processed cols:', non_removable_processed_cols)
print('Missing non-removable after collinearity filter (recovered):', missing_non_removable_after_filter)
print(f'Collinearity dropped: {len(dropped_cols)}')
print('Retrain required due to non-removable check:', RETRAIN_REQUIRED_NON_REMOVABLE)


# In[ ]:


# Heatmap AFTER collinearity-based culling (colors only) + confirmation table for paper.
if 'EDA_DIR' not in globals():
    EDA_DIR = ARTIFACT_DIR / 'eda'
    EDA_IMG_DIR = EDA_DIR / 'images'
    EDA_TABLE_DIR = EDA_DIR / 'tables'
for p in [EDA_DIR, EDA_IMG_DIR, EDA_TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

if 'high_collinearity_pairs' not in globals():
    def high_collinearity_pairs(corr_df: pd.DataFrame, cutoff: float = 0.70) -> pd.DataFrame:
        abs_corr = corr_df.abs()
        upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
        pairs = []
        for c in upper.columns:
            rows = upper.index[upper[c] > cutoff].tolist()
            for r in rows:
                pairs.append(
                    {
                        'feature_1': r,
                        'feature_2': c,
                        'corr': float(corr_df.loc[r, c]),
                        'abs_corr': float(abs_corr.loc[r, c]),
                    }
                )
        if not pairs:
            return pd.DataFrame(columns=['feature_1', 'feature_2', 'corr', 'abs_corr'])
        return pd.DataFrame(pairs).sort_values('abs_corr', ascending=False).reset_index(drop=True)

if X_train_final.shape[1] > 1:
    corr_final = X_train_final.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_final, cmap='coolwarm', center=0, annot=False, cbar=True)
    plt.title('Correlation Heatmap After Collinearity Culling (Colors Only)')
    plt.tight_layout()
    post_heatmap_path = EDA_IMG_DIR / 'heatmap_after_collinearity_culling.png'
    plt.savefig(post_heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved post-culling heatmap: {post_heatmap_path}')

    high_corr_post_df = high_collinearity_pairs(corr_final, cutoff=COLLINEARITY_CUTOFF)
    high_corr_post_path = EDA_TABLE_DIR / f'collinearity_pairs_after_culling_gt_{COLLINEARITY_CUTOFF:.2f}.csv'
    high_corr_post_df.to_csv(high_corr_post_path, index=False)
    print(f'Saved post-culling high-collinearity table: {high_corr_post_path}')
    print(f'Post-culling |corr| > {COLLINEARITY_CUTOFF:.2f} pair count: {len(high_corr_post_df)}')
    if not high_corr_post_df.empty:
        print(high_corr_post_df.head(30).to_string(index=False))
else:
    print('Not enough final features to plot post-collinearity heatmap.')


# In[ ]:


# Save fitted preprocessor to disk
import joblib
preprocessor_save_path = ARTIFACT_DIR / 'preprocessor.joblib'
joblib.dump(preprocessor, preprocessor_save_path)
print(f'Saved preprocessor to: {preprocessor_save_path}')
print(f'Preprocessor type: {type(preprocessor)}')
if hasattr(preprocessor, 'feature_names_in_'):
    print(f'Input features ({len(preprocessor.feature_names_in_)}): {list(preprocessor.feature_names_in_)}')
out_names = preprocessor.get_feature_names_out()
print(f'Output features ({len(out_names)}): {list(out_names)}')


# In[ ]:


MODEL_SPACES = {
    'random_forest': {
        'max_depth': [None, 6, 8, 12, 16, 22],
        'min_samples_split': randint(2, 24),
        'min_samples_leaf': randint(1, 12),
        'max_features': ['sqrt', 'log2', 0.4, 0.6, 0.8],
    },
    'xgboost': {
        'learning_rate': uniform(0.01, 0.24),
        'max_depth': randint(3, 11),
        'subsample': uniform(0.60, 0.40),
        'colsample_bytree': uniform(0.60, 0.40),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0.0, 2.0),
        'reg_lambda': loguniform(1e-3, 30),
    },
    'catboost': {
        'learning_rate': uniform(0.01, 0.24),
        'depth': randint(4, 11),
        'l2_leaf_reg': loguniform(1.0, 30.0),
        'random_strength': uniform(0.0, 2.0),
    },
}

AVAILABLE_MODELS = ['random_forest']
if xgb_available:
    AVAILABLE_MODELS.append('xgboost')
if cat_available:
    AVAILABLE_MODELS.append('catboost')

print('Models:', AVAILABLE_MODELS)


# In[ ]:


def safe_predict_proba(model, X):
    p = model.predict_proba(X)
    if hasattr(p, 'get'):
        p = p.get()
    p = np.asarray(p)
    if p.ndim == 2:
        return np.clip(p[:, 1], 1e-6, 1 - 1e-6)
    return np.clip(p.reshape(-1), 1e-6, 1 - 1e-6)

def metric_pack(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    auc_val = 0.5 if np.unique(y_true).size < 2 else roc_auc_score(y_true, y_prob)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': auc_val,
        'logloss': log_loss(y_true, y_prob, labels=[0, 1]),
    }

def optimization_score(metrics):
    # Primary objective requested: maximize accuracy and recall.
    # Secondary terms stabilize selection and avoid degenerate thresholds.
    return (
        0.60 * float(metrics['accuracy_mean'])
        + 0.40 * float(metrics['recall_mean'])
        + 0.05 * float(metrics['f1_mean'])
        - 0.08 * float(metrics['logloss_mean'])
        - 0.03 * float(metrics['accuracy_std'])
        - 0.03 * float(metrics['recall_std'])
    )

def build_model(model_name, params, epoch_budget):
    p = deepcopy(params)
    use_gpu = bool(USE_GPU_WHEN_AVAILABLE and torch_cuda_available)

    if model_name == 'random_forest':
        if use_gpu and cuml_available:
            return cuRFClassifier(
                n_estimators=int(epoch_budget),
                max_depth=16 if p['max_depth'] is None else int(p['max_depth']),
                max_features=1.0 if isinstance(p['max_features'], str) else float(p['max_features']),
                random_state=RANDOM_SEED,
            )
        return RandomForestClassifier(
            n_estimators=int(epoch_budget),
            max_depth=p['max_depth'],
            min_samples_split=int(p['min_samples_split']),
            min_samples_leaf=int(p['min_samples_leaf']),
            max_features=p['max_features'],
            n_jobs=N_JOBS,
            random_state=RANDOM_SEED,
            class_weight='balanced_subsample',
        )

    if model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=int(epoch_budget),
            learning_rate=float(p['learning_rate']),
            max_depth=int(p['max_depth']),
            subsample=float(p['subsample']),
            colsample_bytree=float(p['colsample_bytree']),
            min_child_weight=float(p['min_child_weight']),
            gamma=float(p['gamma']),
            reg_lambda=float(p['reg_lambda']),
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            tree_method='hist',
            device='cuda' if use_gpu else 'cpu',
            verbosity=0,
        )

    if model_name == 'catboost':
        kwargs = {
            'iterations': int(epoch_budget),
            'learning_rate': float(p['learning_rate']),
            'depth': int(p['depth']),
            'l2_leaf_reg': float(p['l2_leaf_reg']),
            'random_strength': float(p['random_strength']),
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'random_seed': RANDOM_SEED,
            'thread_count': N_JOBS,
            'verbose': False,
        }
        if use_gpu:
            kwargs['task_type'] = 'GPU'
            kwargs['devices'] = '0'
        return CatBoostClassifier(**kwargs)

    raise ValueError(f'Unsupported model: {model_name}')

def fit_model(model, Xtr, ytr, Xva=None, yva=None):
    name = type(model).__name__.lower()
    try:
        if 'catboost' in name and Xva is not None:
            model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=30, verbose=False)
        elif 'xgb' in name and Xva is not None:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            model.fit(Xtr, ytr)
    except Exception as e:
        msg = str(e).lower()
        if 'gpu' in msg or 'cuda' in msg or 'device' in msg:
            if 'xgb' in name:
                model.set_params(device='cpu')
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)] if Xva is not None else None, verbose=False)
            elif 'catboost' in name:
                model.set_params(task_type='CPU')
                model.fit(Xtr, ytr, eval_set=(Xva, yva) if Xva is not None else None, verbose=False)
            else:
                model.fit(Xtr, ytr)
        else:
            raise
    return model

def evaluate_params_cv(model_name, params, X_data, y_data, epoch_budget, n_splits=5):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    threshold_grid = np.round(np.arange(0.35, 0.70, 0.05), 2)
    rows = []

    y_array = np.asarray(y_data).astype(int)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_data, y_array), start=1):
        Xtr = X_data.iloc[tr_idx]
        Xva = X_data.iloc[va_idx]
        ytr = y_data.iloc[tr_idx]
        yva = y_data.iloc[va_idx]

        model = build_model(model_name, params, epoch_budget=epoch_budget)
        model = fit_model(model, Xtr, ytr, Xva, yva)
        p_val = safe_predict_proba(model, Xva)

        best_obj = -np.inf
        best_met = None
        best_thr = 0.5

        for thr in threshold_grid:
            met = metric_pack(yva, p_val, threshold=float(thr))
            # Fold objective centered on accuracy + recall.
            obj = 0.60 * met['accuracy'] + 0.40 * met['recall']
            if obj > best_obj:
                best_obj = obj
                best_met = met
                best_thr = float(thr)

        best_met['fold'] = int(fold)
        best_met['best_threshold'] = float(best_thr)
        rows.append(best_met)

    fold_df = pd.DataFrame(rows)
    summary = {
        'accuracy_mean': float(fold_df['accuracy'].mean()),
        'accuracy_std': float(fold_df['accuracy'].std(ddof=0)),
        'recall_mean': float(fold_df['recall'].mean()),
        'recall_std': float(fold_df['recall'].std(ddof=0)),
        'precision_mean': float(fold_df['precision'].mean()),
        'precision_std': float(fold_df['precision'].std(ddof=0)),
        'f1_mean': float(fold_df['f1'].mean()),
        'f1_std': float(fold_df['f1'].std(ddof=0)),
        'auc_mean': float(fold_df['auc'].mean()),
        'auc_std': float(fold_df['auc'].std(ddof=0)),
        'logloss_mean': float(fold_df['logloss'].mean()),
        'logloss_std': float(fold_df['logloss'].std(ddof=0)),
        'threshold_mean': float(fold_df['best_threshold'].mean()),
    }
    summary['stage_score'] = optimization_score(summary)
    return summary

def refine_candidates(base_params, n_refine=10):
    out = []
    for b in base_params:
        out.append(deepcopy(b))
        for _ in range(n_refine):
            c = {}
            for k, v in b.items():
                if isinstance(v, (int, np.integer)):
                    c[k] = max(1, int(round(v * np.random.uniform(0.7, 1.3))))
                elif isinstance(v, (float, np.floating)):
                    c[k] = max(1e-6, float(v * np.random.uniform(0.7, 1.3)))
                else:
                    c[k] = v
            out.append(c)
    uniq = []
    seen = set()
    for p in out:
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


# In[ ]:


stage1_results = {}
stage1_top_params = {}

for model_name in AVAILABLE_MODELS:
    trials = list(ParameterSampler(
        MODEL_SPACES[model_name],
        n_iter=STAGE1_TRIALS_PER_MODEL,
        random_state=RANDOM_SEED
    ))

    rows = []
    print(f'Stage 1 -> {model_name} | trials={len(trials)} | folds={CV_FOLDS_STAGE1}')

    for i, params in enumerate(trials, start=1):
        try:
            cv_met = evaluate_params_cv(
                model_name, params, X_train_final, y_train,
                epoch_budget=STAGE1_EPOCHS, n_splits=CV_FOLDS_STAGE1
            )
            rows.append({'trial': i, 'params': params, **cv_met})
        except Exception as e:
            rows.append({'trial': i, 'params': params, 'stage_score': -999.0, 'error': str(e)})

        if i % 15 == 0 or i == len(trials):
            print(f'  completed {i}/{len(trials)}')

    df_stage = pd.DataFrame(rows).sort_values(['stage_score', 'accuracy_mean', 'recall_mean'], ascending=False).reset_index(drop=True)
    stage1_results[model_name] = df_stage
    stage1_top_params[model_name] = df_stage.head(TOP_K_STAGE1)['params'].tolist()

stage1_summary = []
for m, df_m in stage1_results.items():
    top = df_m.iloc[0]
    stage1_summary.append({
        'model': m,
        'best_accuracy_cv': top.get('accuracy_mean', np.nan),
        'best_recall_cv': top.get('recall_mean', np.nan),
        'best_stage_score': top.get('stage_score', np.nan),
    })

pd.DataFrame(stage1_summary).sort_values('best_stage_score', ascending=False)


# In[ ]:


stage2_results = {}
best_configs = {}

for model_name in AVAILABLE_MODELS:
    candidates = refine_candidates(stage1_top_params.get(model_name, []), n_refine=STAGE2_REFINEMENTS_PER_TOP_CONFIG)
    rows = []
    print(f'Stage 2 -> {model_name} | candidates={len(candidates)} | folds={CV_FOLDS_STAGE2}')

    for i, params in enumerate(candidates, start=1):
        try:
            cv_met = evaluate_params_cv(
                model_name, params, X_train_final, y_train,
                epoch_budget=STAGE2_EPOCHS, n_splits=CV_FOLDS_STAGE2
            )
            rows.append({'trial': i, 'params': params, **cv_met})
        except Exception as e:
            rows.append({'trial': i, 'params': params, 'stage_score': -999.0, 'error': str(e)})

        if i % 15 == 0 or i == len(candidates):
            print(f'  completed {i}/{len(candidates)}')

    df_stage2 = pd.DataFrame(rows).sort_values(['stage_score', 'accuracy_mean', 'recall_mean'], ascending=False).reset_index(drop=True)
    stage2_results[model_name] = df_stage2
    best_configs[model_name] = df_stage2.head(TOP_K_STAGE2)['params'].tolist()

stage2_summary = []
for m, df_m in stage2_results.items():
    top = df_m.iloc[0]
    stage2_summary.append({
        'model': m,
        'best_accuracy_cv': top.get('accuracy_mean', np.nan),
        'best_recall_cv': top.get('recall_mean', np.nan),
        'best_threshold_cv': top.get('threshold_mean', np.nan),
        'best_stage_score': top.get('stage_score', np.nan),
    })

pd.DataFrame(stage2_summary).sort_values('best_stage_score', ascending=False)


# In[ ]:


# CPU-based models: AdaBoost, KNN, Naive Bayes, Logistic Regression (lighter than GPU search)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Keep CPU optimization intentionally lighter than GPU optimization.
CPU_STAGE1_TRIALS_PER_MODEL = max(36, STAGE1_TRIALS_PER_MODEL // 4)
CPU_STAGE2_REFINEMENTS_PER_TOP_CONFIG = max(4, STAGE2_REFINEMENTS_PER_TOP_CONFIG // 3)
CPU_TOP_K_STAGE1 = min(4, TOP_K_STAGE1)
CPU_TOP_K_STAGE2 = min(2, TOP_K_STAGE2)
CPU_CV_FOLDS_STAGE1 = 3
CPU_CV_FOLDS_STAGE2 = 4
CPU_THRESHOLD_GRID = np.array([0.40, 0.50, 0.60], dtype=float)

CPU_MODEL_SPACES = {
    'adaboost': {
        'n_estimators': randint(30, 150),
        'learning_rate': uniform(0.03, 0.50),
        'base_estimator__max_depth': randint(1, 6),
    },
    'knn': {
        'n_neighbors': randint(5, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    },
    'naive_bayes': {
        'var_smoothing': loguniform(1e-10, 1e-6),
    },
    'logistic_regression': {
        'C': loguniform(1e-4, 1e2),
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [150, 250],
    },
}

print('CPU search profile:', {
    'stage1_trials_per_model': CPU_STAGE1_TRIALS_PER_MODEL,
    'stage2_refinements_per_top_config': CPU_STAGE2_REFINEMENTS_PER_TOP_CONFIG,
    'top_k_stage1': CPU_TOP_K_STAGE1,
    'top_k_stage2': CPU_TOP_K_STAGE2,
    'cv_folds_stage1': CPU_CV_FOLDS_STAGE1,
    'cv_folds_stage2': CPU_CV_FOLDS_STAGE2,
    'threshold_grid': CPU_THRESHOLD_GRID.tolist(),
})

def build_cpu_model(model_name, params):
    p = deepcopy(params)

    if model_name == 'adaboost':
        from sklearn.tree import DecisionTreeClassifier
        base_est = DecisionTreeClassifier(
            max_depth=int(p.get('base_estimator__max_depth', 3)),
            random_state=RANDOM_SEED,
        )
        return AdaBoostClassifier(
            estimator=base_est,
            n_estimators=int(p.get('n_estimators', 80)),
            learning_rate=float(p.get('learning_rate', 0.15)),
            random_state=RANDOM_SEED,
        )

    if model_name == 'knn':
        return KNeighborsClassifier(
            n_neighbors=int(p.get('n_neighbors', 7)),
            weights=p.get('weights', 'uniform'),
            metric=p.get('metric', 'euclidean'),
            n_jobs=N_JOBS,
        )

    if model_name == 'naive_bayes':
        return GaussianNB(
            var_smoothing=float(p.get('var_smoothing', 1e-9)),
        )

    if model_name == 'logistic_regression':
        return LogisticRegression(
            C=float(p.get('C', 1.0)),
            solver=p.get('solver', 'lbfgs'),
            max_iter=int(p.get('max_iter', 150)),
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            class_weight='balanced',
        )

    raise ValueError(f'Unsupported CPU model: {model_name}')

def fit_cpu_model(model, Xtr, ytr, Xva=None, yva=None):
    model.fit(Xtr, ytr)
    return model


# In[ ]:


# Stage 1: CPU Model Hyperparameter Search

cpu_stage1_results = {}
cpu_stage1_top_params = {}

CPU_AVAILABLE_MODELS = ['adaboost', 'knn', 'naive_bayes', 'logistic_regression']

for model_name in CPU_AVAILABLE_MODELS:
    trials = list(ParameterSampler(
        CPU_MODEL_SPACES[model_name],
        n_iter=CPU_STAGE1_TRIALS_PER_MODEL,
        random_state=RANDOM_SEED
    ))

    rows = []
    print(f'CPU Stage 1 -> {model_name} | trials={len(trials)} | folds={CPU_CV_FOLDS_STAGE1}')

    for i, params in enumerate(trials, start=1):
        try:
            splitter = StratifiedKFold(n_splits=CPU_CV_FOLDS_STAGE1, shuffle=True, random_state=RANDOM_SEED)
            threshold_grid = CPU_THRESHOLD_GRID
            fold_rows = []

            y_array = np.asarray(y_train).astype(int)
            for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train_final, y_array), start=1):
                Xtr = X_train_final.iloc[tr_idx]
                Xva = X_train_final.iloc[va_idx]
                ytr = y_train.iloc[tr_idx]
                yva = y_train.iloc[va_idx]

                model = build_cpu_model(model_name, params)
                model = fit_cpu_model(model, Xtr, ytr, Xva, yva)
                p_val = safe_predict_proba(model, Xva)

                best_obj = -np.inf
                best_met = None
                best_thr = 0.5

                for thr in threshold_grid:
                    met = metric_pack(yva, p_val, threshold=float(thr))
                    obj = 0.60 * met['accuracy'] + 0.40 * met['recall']
                    if obj > best_obj:
                        best_obj = obj
                        best_met = met
                        best_thr = float(thr)

                best_met['fold'] = int(fold)
                best_met['best_threshold'] = float(best_thr)
                fold_rows.append(best_met)

            fold_df = pd.DataFrame(fold_rows)
            cv_summary = {
                'accuracy_mean': float(fold_df['accuracy'].mean()),
                'accuracy_std': float(fold_df['accuracy'].std(ddof=0)),
                'recall_mean': float(fold_df['recall'].mean()),
                'recall_std': float(fold_df['recall'].std(ddof=0)),
                'precision_mean': float(fold_df['precision'].mean()),
                'precision_std': float(fold_df['precision'].std(ddof=0)),
                'f1_mean': float(fold_df['f1'].mean()),
                'f1_std': float(fold_df['f1'].std(ddof=0)),
                'auc_mean': float(fold_df['auc'].mean()),
                'auc_std': float(fold_df['auc'].std(ddof=0)),
                'logloss_mean': float(fold_df['logloss'].mean()),
                'logloss_std': float(fold_df['logloss'].std(ddof=0)),
                'threshold_mean': float(fold_df['best_threshold'].mean()),
            }
            cv_summary['stage_score'] = optimization_score(cv_summary)
            rows.append({'trial': i, 'params': params, **cv_summary})
        except Exception as e:
            rows.append({'trial': i, 'params': params, 'stage_score': -999.0, 'error': str(e)})

        if i % 10 == 0 or i == len(trials):
            print(f'  completed {i}/{len(trials)}')

    df_stage = pd.DataFrame(rows).sort_values(['stage_score', 'accuracy_mean', 'recall_mean'], ascending=False).reset_index(drop=True)
    cpu_stage1_results[model_name] = df_stage
    cpu_stage1_top_params[model_name] = df_stage.head(CPU_TOP_K_STAGE1)['params'].tolist()

cpu_stage1_summary = []
for m, df_m in cpu_stage1_results.items():
    top = df_m.iloc[0]
    cpu_stage1_summary.append({
        'model': f'cpu_{m}',
        'best_accuracy_cv': top.get('accuracy_mean', np.nan),
        'best_recall_cv': top.get('recall_mean', np.nan),
        'best_stage_score': top.get('stage_score', np.nan),
    })

print('\nCPU Stage 1 Summary:')
display(pd.DataFrame(cpu_stage1_summary).sort_values('best_stage_score', ascending=False))


# In[ ]:


# Stage 2: CPU Model Refinement

cpu_stage2_results = {}
cpu_best_configs = {}

for model_name in CPU_AVAILABLE_MODELS:
    candidates = refine_candidates(cpu_stage1_top_params.get(model_name, []), n_refine=CPU_STAGE2_REFINEMENTS_PER_TOP_CONFIG)
    rows = []
    print(f'CPU Stage 2 -> {model_name} | candidates={len(candidates)} | folds={CPU_CV_FOLDS_STAGE2}')

    for i, params in enumerate(candidates, start=1):
        try:
            splitter = StratifiedKFold(n_splits=CPU_CV_FOLDS_STAGE2, shuffle=True, random_state=RANDOM_SEED)
            threshold_grid = CPU_THRESHOLD_GRID
            fold_rows = []

            y_array = np.asarray(y_train).astype(int)
            for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train_final, y_array), start=1):
                Xtr = X_train_final.iloc[tr_idx]
                Xva = X_train_final.iloc[va_idx]
                ytr = y_train.iloc[tr_idx]
                yva = y_train.iloc[va_idx]

                model = build_cpu_model(model_name, params)
                model = fit_cpu_model(model, Xtr, ytr, Xva, yva)
                p_val = safe_predict_proba(model, Xva)

                best_obj = -np.inf
                best_met = None
                best_thr = 0.5

                for thr in threshold_grid:
                    met = metric_pack(yva, p_val, threshold=float(thr))
                    obj = 0.60 * met['accuracy'] + 0.40 * met['recall']
                    if obj > best_obj:
                        best_obj = obj
                        best_met = met
                        best_thr = float(thr)

                best_met['fold'] = int(fold)
                best_met['best_threshold'] = float(best_thr)
                fold_rows.append(best_met)

            fold_df = pd.DataFrame(fold_rows)
            cv_summary = {
                'accuracy_mean': float(fold_df['accuracy'].mean()),
                'accuracy_std': float(fold_df['accuracy'].std(ddof=0)),
                'recall_mean': float(fold_df['recall'].mean()),
                'recall_std': float(fold_df['recall'].std(ddof=0)),
                'precision_mean': float(fold_df['precision'].mean()),
                'precision_std': float(fold_df['precision'].std(ddof=0)),
                'f1_mean': float(fold_df['f1'].mean()),
                'f1_std': float(fold_df['f1'].std(ddof=0)),
                'auc_mean': float(fold_df['auc'].mean()),
                'auc_std': float(fold_df['auc'].std(ddof=0)),
                'logloss_mean': float(fold_df['logloss'].mean()),
                'logloss_std': float(fold_df['logloss'].std(ddof=0)),
                'threshold_mean': float(fold_df['best_threshold'].mean()),
            }
            cv_summary['stage_score'] = optimization_score(cv_summary)
            rows.append({'trial': i, 'params': params, **cv_summary})
        except Exception as e:
            rows.append({'trial': i, 'params': params, 'stage_score': -999.0, 'error': str(e)})

        if i % 10 == 0 or i == len(candidates):
            print(f'  completed {i}/{len(candidates)}')

    df_stage2 = pd.DataFrame(rows).sort_values(['stage_score', 'accuracy_mean', 'recall_mean'], ascending=False).reset_index(drop=True)
    cpu_stage2_results[model_name] = df_stage2
    cpu_best_configs[model_name] = df_stage2.head(CPU_TOP_K_STAGE2)['params'].tolist()

cpu_stage2_summary = []
for m, df_m in cpu_stage2_results.items():
    top = df_m.iloc[0]
    cpu_stage2_summary.append({
        'model': f'cpu_{m}',
        'best_accuracy_cv': top.get('accuracy_mean', np.nan),
        'best_recall_cv': top.get('recall_mean', np.nan),
        'best_threshold_cv': top.get('threshold_mean', np.nan),
        'best_stage_score': top.get('stage_score', np.nan),
    })

print('\nCPU Stage 2 Summary:')
display(pd.DataFrame(cpu_stage2_summary).sort_values('best_stage_score', ascending=False))


# In[ ]:


final_models = {}
selected_final = {}
final_rows = []
SAVE_ALL_FINAL_MODELS = False

# GPU Models
for model_name in AVAILABLE_MODELS:
    params_list = best_configs.get(model_name, [])
    if not params_list:
        continue

    best_model = None
    best_params = None
    best_val_score = -np.inf
    best_threshold = 0.5

    for params in params_list:
        model = build_model(model_name, params, epoch_budget=FINAL_EPOCHS)
        model = fit_model(model, X_train_final, y_train, X_valid_final, y_valid)

        p_valid = safe_predict_proba(model, X_valid_final)
        threshold_grid = np.round(np.arange(0.35, 0.70, 0.05), 2)

        local_best = -np.inf
        local_thr = 0.5
        local_met = None

        for thr in threshold_grid:
            met = metric_pack(y_valid, p_valid, threshold=float(thr))
            s = 0.60 * met['accuracy'] + 0.40 * met['recall']
            if s > local_best:
                local_best = s
                local_thr = float(thr)
                local_met = met

        if local_best > best_val_score:
            best_val_score = local_best
            best_model = model
            best_params = params
            best_threshold = local_thr
            best_valid_metrics = local_met

    final_models[model_name] = best_model
    selected_final[model_name] = {'params': best_params, 'threshold': best_threshold}

    p_test = safe_predict_proba(best_model, X_test_final)
    test_met = metric_pack(y_test, p_test, threshold=best_threshold)

    final_rows.append({
        'model': model_name,
        'valid_accuracy': best_valid_metrics['accuracy'],
        'valid_recall': best_valid_metrics['recall'],
        'valid_f1': best_valid_metrics['f1'],
        'valid_auc': best_valid_metrics['auc'],
        'valid_logloss': best_valid_metrics['logloss'],
        'selected_threshold': best_threshold,
        'test_accuracy': test_met['accuracy'],
        'test_recall': test_met['recall'],
        'test_f1': test_met['f1'],
        'test_auc': test_met['auc'],
        'test_logloss': test_met['logloss'],
        'params': best_params,
    })

    if SAVE_ALL_FINAL_MODELS:
        model_path = MODEL_DIR / f'{model_name}.joblib'
        import joblib
        joblib.dump(best_model, model_path)

# CPU Models
for model_name in CPU_AVAILABLE_MODELS:
    params_list = cpu_best_configs.get(model_name, [])
    if not params_list:
        continue

    best_model = None
    best_params = None
    best_val_score = -np.inf
    best_threshold = 0.5

    for params in params_list:
        model = build_cpu_model(model_name, params)
        model = fit_cpu_model(model, X_train_final, y_train, X_valid_final, y_valid)

        p_valid = safe_predict_proba(model, X_valid_final)
        threshold_grid = np.round(np.arange(0.35, 0.70, 0.05), 2)

        local_best = -np.inf
        local_thr = 0.5
        local_met = None

        for thr in threshold_grid:
            met = metric_pack(y_valid, p_valid, threshold=float(thr))
            s = 0.60 * met['accuracy'] + 0.40 * met['recall']
            if s > local_best:
                local_best = s
                local_thr = float(thr)
                local_met = met

        if local_best > best_val_score:
            best_val_score = local_best
            best_model = model
            best_params = params
            best_threshold = local_thr
            best_valid_metrics = local_met

    final_models[f'cpu_{model_name}'] = best_model
    selected_final[f'cpu_{model_name}'] = {'params': best_params, 'threshold': best_threshold}

    p_test = safe_predict_proba(best_model, X_test_final)
    test_met = metric_pack(y_test, p_test, threshold=best_threshold)

    final_rows.append({
        'model': f'cpu_{model_name}',
        'valid_accuracy': best_valid_metrics['accuracy'],
        'valid_recall': best_valid_metrics['recall'],
        'valid_f1': best_valid_metrics['f1'],
        'valid_auc': best_valid_metrics['auc'],
        'valid_logloss': best_valid_metrics['logloss'],
        'selected_threshold': best_threshold,
        'test_accuracy': test_met['accuracy'],
        'test_recall': test_met['recall'],
        'test_f1': test_met['f1'],
        'test_auc': test_met['auc'],
        'test_logloss': test_met['logloss'],
        'params': best_params,
    })

    if SAVE_ALL_FINAL_MODELS:
        model_path = MODEL_DIR / f'cpu_{model_name}.joblib'
        import joblib
        joblib.dump(best_model, model_path)

final_results_df = pd.DataFrame(final_rows).sort_values(['valid_accuracy', 'valid_recall'], ascending=False).reset_index(drop=True)
final_results_path = ARTIFACT_DIR / 'final_results_with_cpu.csv'
final_results_df.to_csv(final_results_path, index=False)
print('Saved:', final_results_path)
print(f'\nTotal models trained: {len(final_models)} (GPU + CPU combined)')
print('Saved all final model artifacts:', SAVE_ALL_FINAL_MODELS)
final_results_df


# In[ ]:


def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        m = idx == i
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += m.mean() * abs(acc - conf)
    return float(ece)

def fit_platt_calibrator(p_fit, y_fit):
    lr = LogisticRegression(max_iter=3000, random_state=RANDOM_SEED, n_jobs=N_JOBS)
    lr.fit(p_fit.reshape(-1, 1), y_fit)
    return lr

def fit_isotonic_calibrator(p_fit, y_fit):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_fit, y_fit)
    return iso

def fit_venn_abers_calibrator(p_fit, y_fit):
    if not venn_available:
        raise RuntimeError('venn-abers not installed')
    va = VennAbers()
    va.fit(np.column_stack([1.0 - p_fit, p_fit]), np.asarray(y_fit))
    return va

def apply_with_calibrator(method, calibrator, p_eval):
    p_eval = np.clip(np.asarray(p_eval), 1e-6, 1 - 1e-6)
    if method == 'base' or calibrator is None:
        return p_eval
    if method == 'platt':
        return np.clip(calibrator.predict_proba(p_eval.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    if method == 'isotonic':
        return np.clip(calibrator.predict(p_eval), 1e-6, 1 - 1e-6)
    if method == 'venn_abers':
        _, p1 = calibrator.predict_proba(np.column_stack([1.0 - p_eval, p_eval]))
        p1 = np.asarray(p1)
        if p1.ndim == 2:
            return np.clip(p1[:, 1], 1e-6, 1 - 1e-6)
        return np.clip(p1.reshape(-1), 1e-6, 1 - 1e-6)
    raise ValueError(method)

def fit_calibrator(method, p_fit, y_fit):
    if method == 'base':
        return None
    if method == 'platt':
        return fit_platt_calibrator(p_fit, y_fit)
    if method == 'isotonic':
        return fit_isotonic_calibrator(p_fit, y_fit)
    if method == 'venn_abers':
        return fit_venn_abers_calibrator(p_fit, y_fit)
    raise ValueError(method)

def align_features_for_model(model, X_data):
    if hasattr(model, 'feature_names_in_'):
        model_cols = list(model.feature_names_in_)
        missing = [c for c in model_cols if c not in X_data.columns]
        if missing:
            raise ValueError(f'Model expects missing features: {missing}')
        return X_data.loc[:, model_cols].copy(), model_cols
    return X_data.copy(), list(X_data.columns)

calibration_rows = []
calibrated_test_probs = {}
calibrated_model_artifacts = {}

for model_name, model in final_models.items():
    threshold = selected_final[model_name]['threshold']
    X_valid_eval, model_cols = align_features_for_model(model, X_valid_final)
    X_test_eval, _ = align_features_for_model(model, X_test_final)

    p_valid_base = safe_predict_proba(model, X_valid_eval)
    p_test_base = safe_predict_proba(model, X_test_eval)

    methods = ['base', 'platt', 'isotonic']
    if venn_available:
        methods.append('venn_abers')

    p_fit, p_eval, y_fit, y_eval = train_test_split(
        p_valid_base, y_valid.values, test_size=0.5, random_state=RANDOM_SEED, stratify=y_valid.values
    )

    for method in methods:
        try:
            calibrator = fit_calibrator(method, p_fit, y_fit)

            p_eval_cal = apply_with_calibrator(method, calibrator, p_eval)
            eval_met = metric_pack(y_eval, p_eval_cal, threshold=threshold)

            p_test_cal = apply_with_calibrator(method, calibrator, p_test_base)
            calibrated_test_probs[(model_name, method)] = p_test_cal

            calibrated_model_artifacts[(model_name, method)] = {
                'base_model_name': model_name,
                'base_model': model,
                'calibration_method': method,
                'calibrator': calibrator,
                'selected_threshold': float(threshold),
                'feature_names': model_cols,
            }

            calibration_rows.append({
                'model': model_name,
                'method': method,
                'selected_threshold': threshold,
                'cal_accuracy': eval_met['accuracy'],
                'cal_recall': eval_met['recall'],
                'cal_f1': eval_met['f1'],
                'cal_auc': eval_met['auc'],
                'cal_logloss': eval_met['logloss'],
                'cal_ece': expected_calibration_error(y_eval, p_eval_cal),
            })
        except Exception as e:
            calibration_rows.append({'model': model_name, 'method': method, 'error': str(e)})

calibration_df = pd.DataFrame(calibration_rows)
ok = calibration_df['error'].isna() if 'error' in calibration_df.columns else pd.Series([True] * len(calibration_df))

if ok.any():
    calibration_df.loc[ok, 'rank_acc'] = calibration_df.loc[ok, 'cal_accuracy'].rank(method='min', ascending=False)
    calibration_df.loc[ok, 'rank_recall'] = calibration_df.loc[ok, 'cal_recall'].rank(method='min', ascending=False)
    calibration_df.loc[ok, 'rank_logloss'] = calibration_df.loc[ok, 'cal_logloss'].rank(method='min', ascending=True)
    calibration_df.loc[ok, 'rank_ece'] = calibration_df.loc[ok, 'cal_ece'].rank(method='min', ascending=True)
    calibration_df.loc[ok, 'rank_mean'] = (
        calibration_df.loc[ok, 'rank_acc']
        + calibration_df.loc[ok, 'rank_recall']
        + calibration_df.loc[ok, 'rank_logloss']
        + calibration_df.loc[ok, 'rank_ece']
    ) / 4.0

calibration_df = calibration_df.sort_values(['rank_mean', 'cal_accuracy', 'cal_recall'], ascending=[True, False, False], na_position='last').reset_index(drop=True)
calibration_path = ARTIFACT_DIR / 'calibration_results.csv'
calibration_df.to_csv(calibration_path, index=False)
print('Saved:', calibration_path)
calibration_df.head(20)


# In[ ]:


# Paper tables: (1) all trained models, (2) models selected for calibration, (3) all calibrated models.
# This cell is robust: it uses in-memory objects when available, otherwise falls back to saved artifacts.

def _load_df_if_exists(path_obj):
    if path_obj.exists():
        return pd.read_csv(path_obj)
    return None

# Resolve artifact paths.
final_results_with_cpu_path = ARTIFACT_DIR / 'final_results_with_cpu.csv'
calibration_results_path = ARTIFACT_DIR / 'calibration_results.csv'

# Pull final-model table source.
if 'final_results_df' in globals() and isinstance(final_results_df, pd.DataFrame) and not final_results_df.empty:
    _final_df = final_results_df.copy()
else:
    _final_df = _load_df_if_exists(final_results_with_cpu_path)

if _final_df is None or _final_df.empty:
    raise ValueError(
        f'No final model results found. Expected in memory (`final_results_df`) or file: {final_results_with_cpu_path}'
    )

# Pull calibration table source.
if 'calibration_df' in globals() and isinstance(calibration_df, pd.DataFrame) and not calibration_df.empty:
    _cal_df = calibration_df.copy()
else:
    _cal_df = _load_df_if_exists(calibration_results_path)

if _cal_df is None or _cal_df.empty:
    raise ValueError(
        f'No calibration results found. Expected in memory (`calibration_df`) or file: {calibration_results_path}'
    )


def _infer_balancing_method(model_name: str, params_obj):
    m = str(model_name).lower()

    # Explicit class-weight handling in this notebook.
    if m == 'random_forest':
        return 'class_weight=balanced_subsample'
    if m == 'cpu_logistic_regression':
        return 'class_weight=balanced'

    # Parameter-level fallback if class_weight appears in serialized params.
    if isinstance(params_obj, dict) and 'class_weight' in params_obj:
        return f"class_weight={params_obj['class_weight']}"

    # No explicit sample-level rebalancing (SMOTE/ADASYN) is used in this notebook.
    return 'none'


def _family_and_compute(model_name: str):
    m = str(model_name)
    if m.startswith('cpu_'):
        return m.replace('cpu_', '', 1), 'cpu'
    return m, 'gpu_or_gpu_first'

# Table 1: all models trained (GPU + CPU) with balancing method.
trained_rows = []
for _, row in _final_df.iterrows():
    family, compute_type = _family_and_compute(row['model'])
    trained_rows.append(
        {
            'model_key': row['model'],
            'model_family': family,
            'compute_type': compute_type,
            'balancing_method': _infer_balancing_method(row['model'], row.get('params', None)),
            'selected_threshold': row.get('selected_threshold', np.nan),
            'valid_accuracy': row.get('valid_accuracy', np.nan),
            'valid_recall': row.get('valid_recall', np.nan),
            'valid_f1': row.get('valid_f1', np.nan),
            'valid_auc': row.get('valid_auc', np.nan),
            'test_accuracy': row.get('test_accuracy', np.nan),
            'test_recall': row.get('test_recall', np.nan),
            'test_f1': row.get('test_f1', np.nan),
            'test_auc': row.get('test_auc', np.nan),
        }
    )

paper_all_trained_df = pd.DataFrame(trained_rows).sort_values(
    ['compute_type', 'valid_accuracy', 'valid_recall'],
    ascending=[True, False, False],
).reset_index(drop=True)

paper_all_trained_path = ARTIFACT_DIR / 'paper_table_all_models_trained.csv'
paper_all_trained_df.to_csv(paper_all_trained_path, index=False)

# Table 2: models selected for calibration.
selected_for_calibration_models = sorted(_cal_df['model'].dropna().astype(str).unique().tolist())

# Threshold source preference: selected_final in memory; fallback to final results table.
threshold_lookup = {}
if 'selected_final' in globals() and isinstance(selected_final, dict):
    for k, v in selected_final.items():
        if isinstance(v, dict) and 'threshold' in v:
            threshold_lookup[str(k)] = v['threshold']
if not threshold_lookup and 'model' in _final_df.columns and 'selected_threshold' in _final_df.columns:
    threshold_lookup = {
        str(r['model']): r['selected_threshold']
        for _, r in _final_df[['model', 'selected_threshold']].dropna().iterrows()
    }

selected_rows = []
for model_key in selected_for_calibration_models:
    family, compute_type = _family_and_compute(model_key)
    selected_rows.append(
        {
            'model_key': model_key,
            'model_family': family,
            'compute_type': compute_type,
            'balancing_method': _infer_balancing_method(model_key, None),
            'selected_threshold': threshold_lookup.get(model_key, np.nan),
            'selected_for_calibration': True,
        }
    )

paper_selected_for_calibration_df = pd.DataFrame(selected_rows).sort_values(
    ['compute_type', 'model_family'],
    ascending=[True, True],
).reset_index(drop=True)

paper_selected_for_calibration_path = ARTIFACT_DIR / 'paper_table_models_selected_for_calibration.csv'
paper_selected_for_calibration_df.to_csv(paper_selected_for_calibration_path, index=False)

# Table 3: all calibrated models (all model-method combinations).
paper_calibrated_df = _cal_df.copy()
if 'error' in paper_calibrated_df.columns:
    paper_calibrated_df['calibration_status'] = np.where(
        paper_calibrated_df['error'].isna(),
        'ok',
        'failed',
    )
else:
    paper_calibrated_df['calibration_status'] = 'ok'

paper_calibrated_df['model_key'] = paper_calibrated_df['model'].astype(str)
paper_calibrated_df['model_family'] = paper_calibrated_df['model_key'].apply(lambda x: _family_and_compute(x)[0])
paper_calibrated_df['compute_type'] = paper_calibrated_df['model_key'].apply(lambda x: _family_and_compute(x)[1])
paper_calibrated_df['balancing_method'] = paper_calibrated_df['model_key'].apply(lambda x: _infer_balancing_method(x, None))

paper_calibrated_df = paper_calibrated_df.sort_values(
    ['calibration_status', 'rank_mean', 'cal_accuracy', 'cal_recall'],
    ascending=[True, True, False, False],
    na_position='last',
).reset_index(drop=True)

paper_calibrated_path = ARTIFACT_DIR / 'paper_table_all_calibrated_models.csv'
paper_calibrated_df.to_csv(paper_calibrated_path, index=False)

print('Saved paper tables:')
print(' - All trained models:', paper_all_trained_path)
print(' - Models selected for calibration:', paper_selected_for_calibration_path)
print(' - All calibrated models:', paper_calibrated_path)

print('\nPreview: all trained models')
display(paper_all_trained_df)
print('\nPreview: models selected for calibration')
display(paper_selected_for_calibration_df)
print('\nPreview: all calibrated models (top 30 rows)')
display(paper_calibrated_df.head(30))


# In[ ]:


import joblib

TOP_CALIBRATED_MODELS_TO_SAVE = 3

clean_cal_df = calibration_df[calibration_df['error'].isna()] if 'error' in calibration_df.columns else calibration_df
if clean_cal_df.empty:
    raise ValueError('No successful calibration rows found. Check calibration_df for failures before saving the best model.')

# Save top few calibrated model artifacts.
calibrated_dir = MODEL_DIR / 'calibrated_top_models'
calibrated_dir.mkdir(parents=True, exist_ok=True)

saved_calibrated_rows = []
for i, (_, row) in enumerate(clean_cal_df.head(TOP_CALIBRATED_MODELS_TO_SAVE).iterrows(), start=1):
    model_name = row['model']
    method = row['method']
    key = (model_name, method)

    if key not in calibrated_model_artifacts:
        continue

    artifact = calibrated_model_artifacts[key]
    safe_model_name = str(model_name).replace('/', '_').replace(' ', '_')
    out_path = calibrated_dir / f'top{i}_{safe_model_name}_{method}.joblib'

    joblib.dump(artifact, out_path)
    saved_calibrated_rows.append({
        'rank': i,
        'model': model_name,
        'method': method,
        'selected_threshold': float(row['selected_threshold']),
        'path': str(out_path),
    })

best_row = clean_cal_df.iloc[0]
best_model_name = best_row['model']
best_method = best_row['method']
best_threshold = float(best_row['selected_threshold'])
best_model_obj = final_models[best_model_name]

p_best_test = calibrated_test_probs[(best_model_name, best_method)]
best_test_metrics = metric_pack(y_test, p_best_test, threshold=best_threshold)
best_test_metrics['test_ece'] = expected_calibration_error(y_test.values, p_best_test)

best_model_path = MODEL_DIR / 'best_model.joblib'
joblib.dump(best_model_obj, best_model_path)

summary = {
    'best_model': best_model_name,
    'best_calibration_method': best_method,
    'selected_threshold': best_threshold,
    'test_accuracy': float(best_test_metrics['accuracy']),
    'test_recall': float(best_test_metrics['recall']),
    'test_precision': float(best_test_metrics['precision']),
    'test_f1': float(best_test_metrics['f1']),
    'test_auc': float(best_test_metrics['auc']),
    'test_logloss': float(best_test_metrics['logloss']),
    'test_ece': float(best_test_metrics['test_ece']),
    'best_model_joblib': str(best_model_path),
    'saved_top_calibrated_joblibs': saved_calibrated_rows,
}

summary_path = ARTIFACT_DIR / 'best_summary.json'
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(pd.DataFrame([summary]).to_string(index=False))
print('Saved model:', best_model_path)
print('Saved top calibrated artifacts dir:', calibrated_dir)
print('Saved summary:', summary_path)
if saved_calibrated_rows:
    display(pd.DataFrame(saved_calibrated_rows))


# In[ ]:


TOP_5_CALIBRATED_MODELS_TO_SAVE = 5

clean_cal_df = calibration_df[calibration_df['error'].isna()] if 'error' in calibration_df.columns else calibration_df
if clean_cal_df.empty:
    raise ValueError('No successful calibration rows found. Run the calibration cell first.')

calibrated_top5_dir = MODEL_DIR / 'calibrated_top5_models'
calibrated_top5_dir.mkdir(parents=True, exist_ok=True)

saved_top5_rows = []
for rank, (_, row) in enumerate(clean_cal_df.head(TOP_5_CALIBRATED_MODELS_TO_SAVE).iterrows(), start=1):
    model_name = row['model']
    method = row['method']
    key = (model_name, method)

    if key not in calibrated_model_artifacts:
        continue

    artifact = calibrated_model_artifacts[key]
    safe_model_name = str(model_name).replace('/', '_').replace(' ', '_')
    out_path = calibrated_top5_dir / f'top{rank}_{safe_model_name}_{method}.joblib'
    joblib.dump(artifact, out_path)

    saved_top5_rows.append({
        'rank': rank,
        'model': model_name,
        'method': method,
        'selected_threshold': float(row['selected_threshold']),
        'path': str(out_path),
    })

print('Saved top 5 calibrated model artifacts to:', calibrated_top5_dir)
display(pd.DataFrame(saved_top5_rows))


# In[22]:


# Prep for explainability-only run: load saved best model and define safe_predict_proba.
import joblib

best_summary_path = ARTIFACT_DIR / 'best_summary.json'
default_model_path = MODEL_DIR / 'calibrated_top_models' / 'top1_xgboost_base.joblib'

best_model_path = default_model_path
if best_summary_path.exists():
    with open(best_summary_path, 'r', encoding='utf-8') as f:
        best_summary_payload = json.load(f)

    raw_best_model_path = best_summary_payload.get('best_model_joblib')
    if raw_best_model_path:
        candidate_from_summary = Path(raw_best_model_path)
        if candidate_from_summary.exists():
            best_model_path = candidate_from_summary

# Fallbacks if summary path is stale or artifact naming differs.
if not best_model_path.exists():
    fallback_candidates = [
        default_model_path,
        MODEL_DIR / 'best_model.joblib',
        MODEL_DIR / 'calibrated_top_models' / 'top1_xgboost_base.joblib',
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            best_model_path = candidate
            break

if not best_model_path.exists():
    raise FileNotFoundError(
        f'Best model artifact not found. Checked: {[str(p) for p in [default_model_path, MODEL_DIR / "best_model.joblib"]]}. '
        'Run model selection/saving cells first.'
    )

loaded_artifact = joblib.load(best_model_path)


def _find_predict_proba_model(obj):
    if hasattr(obj, 'predict_proba'):
        return obj
    if isinstance(obj, dict):
        for key in ['model', 'estimator', 'classifier', 'clf', 'base_model', 'wrapped_model']:
            if key in obj:
                candidate = _find_predict_proba_model(obj[key])
                if candidate is not None:
                    return candidate
        for value in obj.values():
            candidate = _find_predict_proba_model(value)
            if candidate is not None:
                return candidate
    if isinstance(obj, (list, tuple)):
        for value in obj:
            candidate = _find_predict_proba_model(value)
            if candidate is not None:
                return candidate
    return None


best_model_obj = _find_predict_proba_model(loaded_artifact)
if best_model_obj is None:
    raise TypeError(
        f'Could not locate a predict_proba-capable model inside loaded artifact type: {type(loaded_artifact)}'
    )


def safe_predict_proba(model, X):
    p = model.predict_proba(X)
    if hasattr(p, 'get'):
        p = p.get()
    p = np.asarray(p)
    if p.ndim == 1:
        return p
    if p.shape[1] == 1:
        return p[:, 0]
    return p[:, 1]

print('Loaded best model for explanations:', type(best_model_obj).__name__)
print('Model path:', best_model_path)
if isinstance(loaded_artifact, dict):
    print('Artifact top-level keys:', list(loaded_artifact.keys())[:12])


# In[23]:


try:
    import shap
except ImportError as exc:
    raise ImportError("SHAP is not installed. Install it with `%pip install -q shap` and rerun this cell.") from exc

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError as exc:
    raise ImportError("LIME is not installed. Install it with `%pip install -q lime` and rerun this cell.") from exc

model_feature_names = None
if hasattr(best_model_obj, 'feature_names_in_'):
    model_feature_names = list(best_model_obj.feature_names_in_)
elif hasattr(best_model_obj, 'get_booster'):
    booster = best_model_obj.get_booster()
    if getattr(booster, 'feature_names', None):
        model_feature_names = list(booster.feature_names)

if model_feature_names is not None:
    missing_model_features = [c for c in model_feature_names if c not in X_train_final.columns]
    if missing_model_features:
        raise ValueError(
            f'Model expects features not present in the current final matrix: {missing_model_features}. '
            'Re-run the training cells so the model matches the latest preprocessing output.'
        )
    explain_train_source = X_train_final.loc[:, model_feature_names].copy()
    explain_test_source = X_test_final.loc[:, model_feature_names].copy()
else:
    explain_train_source = X_train_final.copy()
    explain_test_source = X_test_final.copy()

explain_dir = ARTIFACT_DIR / 'explanations'
explain_dir.mkdir(parents=True, exist_ok=True)

background_size = min(300, len(explain_train_source))
eval_size = min(200, len(explain_test_source))
shap_background = explain_train_source.sample(background_size, random_state=RANDOM_SEED)
shap_eval = explain_test_source.sample(eval_size, random_state=RANDOM_SEED)

def best_model_predict_proba_df(data_like):
    if isinstance(data_like, pd.DataFrame):
        data_df = data_like.copy()
    else:
        data_df = pd.DataFrame(data_like, columns=shap_background.columns)
    p1 = safe_predict_proba(best_model_obj, data_df)
    return np.column_stack([1.0 - p1, p1])

try:
    shap_explainer = shap.TreeExplainer(best_model_obj)
    shap_values = shap_explainer.shap_values(shap_eval)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
except Exception:
    shap_explainer = shap.Explainer(best_model_predict_proba_df, shap_background)
    shap_result = shap_explainer(shap_eval)
    shap_values = np.asarray(shap_result.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

shap_importance_df = pd.DataFrame({
    'feature': shap_eval.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0),
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

shap_importance_path = explain_dir / 'shap_importance.csv'
shap_importance_df.to_csv(shap_importance_path, index=False)

# SHAP global bar summary (paper-friendly, deterministic filename)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, shap_eval, plot_type='bar', max_display=20, show=False)
plt.tight_layout()
shap_bar_path = explain_dir / 'shap_summary_bar.png'
plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
plt.close()

# SHAP beeswarm summary (distributional global view)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, shap_eval, max_display=20, show=False)
plt.tight_layout()
shap_beeswarm_path = explain_dir / 'shap_summary_beeswarm.png'
plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches='tight')
plt.close()

lime_explainer = LimeTabularExplainer(
    training_data=shap_background.to_numpy(),
    feature_names=shap_background.columns.tolist(),
    class_names=['class_0', 'class_1'],
    mode='classification',
    discretize_continuous=True,
    random_state=RANDOM_SEED,
)

lime_row = shap_eval.iloc[0]
lime_exp = lime_explainer.explain_instance(
    lime_row.to_numpy(),
    best_model_predict_proba_df,
    num_features=min(12, shap_background.shape[1]),
)

lime_weights_df = pd.DataFrame(lime_exp.as_list(), columns=['feature', 'weight'])
lime_csv_path = explain_dir / 'lime_explanation_row0.csv'
lime_html_path = explain_dir / 'lime_explanation_row0.html'
lime_png_path = explain_dir / 'lime_explanation_row0.png'
lime_weights_df.to_csv(lime_csv_path, index=False)
lime_exp.save_to_file(str(lime_html_path))

lime_fig = lime_exp.as_pyplot_figure()
lime_fig.tight_layout()
lime_fig.savefig(lime_png_path, dpi=300, bbox_inches='tight')
plt.close(lime_fig)

print('Best model explanation artifacts:')
print(' - Feature set used by fitted model:', shap_background.shape[1])
print(' - SHAP importance CSV:', shap_importance_path)
print(' - SHAP bar plot:', shap_bar_path)
print(' - SHAP beeswarm plot:', shap_beeswarm_path)
print(' - LIME weights CSV:', lime_csv_path)
print(' - LIME HTML:', lime_html_path)
print(' - LIME PNG:', lime_png_path)

display(shap_importance_df.head(20))
display(lime_weights_df)


# ## Explanation
# 
# ### Why this is rigorous
# - Two-stage hyperparameter search with cross-validation (broad search then local refinement).
# - Threshold optimization inside each fold rather than fixing 0.5.
# - Final model selection based on validation performance with explicit emphasis on accuracy and recall.
# - Post-hoc probability calibration using multiple methods with ranking.
# 
# ### Optimization target
# The search objective is explicitly centered on **accuracy** and **recall**:
# 
# $$
# 	ext{score} = 0.60verline{	ext{Accuracy}} + 0.40verline{	ext{Recall}} + 0.05verline{	ext{F1}} - 0.08verline{	ext{LogLoss}} - 0.03igma_{	ext{Accuracy}} - 0.03igma_{	ext{Recall}}
# $$
# 
# This preserves your requested focus while penalizing unstable or poorly calibrated candidates.
# 
# ### GPU behavior
# - XGBoost: GPU-first with CUDA device; auto CPU fallback if GPU fails.
# - CatBoost: GPU-first (`task_type='GPU'`); auto CPU fallback if needed.
# - Random Forest: GPU-first with cuML if available, otherwise scikit-learn fallback with `n_jobs=-1`.
# 
# ### Collinearity visualization order (as requested)
# 1. Heatmap after manual variable dropping (annotated).
# 2. Heatmap after collinearity dropping (colors only, no numbers).

# In[14]:


# Missing data visualization: % missing per variable + missingness heatmap
if 'EDA_DIR' not in globals():
    EDA_DIR = ARTIFACT_DIR / 'eda'
    EDA_IMG_DIR = EDA_DIR / 'images'
    EDA_TABLE_DIR = EDA_DIR / 'tables'
for p in [EDA_DIR, EDA_IMG_DIR, EDA_TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

missing_pct = (X.isna().sum() / len(X) * 100).sort_values(ascending=False)
missing_pct_df = missing_pct.rename('missing_percent').reset_index().rename(columns={'index': 'feature'})
missing_pct_path = EDA_TABLE_DIR / 'missing_data_summary.csv'
missing_pct_df.to_csv(missing_pct_path, index=False)
print(f'Saved missing data summary table: {missing_pct_path}')

missing_pct_nonzero = missing_pct[missing_pct > 0]

if len(missing_pct_nonzero) > 0:
    plt.figure(figsize=(12, max(8, len(missing_pct_nonzero) * 0.25)))
    missing_pct_nonzero.sort_values().plot(kind='barh', color='#d32f2f')
    plt.xlabel('% Missing', fontsize=12)
    plt.title('Missing Data by Variable', fontsize=14, fontweight='bold')
    plt.tight_layout()
    missing_path = EDA_IMG_DIR / 'missing_data_pct.png'
    plt.savefig(missing_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved missing data chart: {missing_path}')
    print(f'Variables with missing data: {len(missing_pct_nonzero)}')
    print(missing_pct_nonzero.to_string())

    top_missing_cols = missing_pct_nonzero.head(min(30, len(missing_pct_nonzero))).index.tolist()
    missing_sample_n = min(3000, len(X))
    missing_sample = X[top_missing_cols].sample(n=missing_sample_n, random_state=RANDOM_SEED) if len(X) > missing_sample_n else X[top_missing_cols].copy()

    plt.figure(figsize=(14, max(6, len(top_missing_cols) * 0.3)))
    sns.heatmap(missing_sample.isna().T, cmap=['#f5f5f5', '#212121'], cbar=True)
    plt.title(f'Missingness Heatmap (Top {len(top_missing_cols)} Variables with Highest Missingness)', fontsize=14, fontweight='bold')
    plt.xlabel('Sampled Observations')
    plt.ylabel('Variable')
    plt.tight_layout()
    missing_heatmap_path = EDA_IMG_DIR / 'missingness_heatmap_top_variables.png'
    plt.savefig(missing_heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved missingness heatmap: {missing_heatmap_path}')
else:
    print('No missing data detected in raw input.')


# In[16]:


# Per-variable distributions: numeric features split by target class
if 'EDA_DIR' not in globals():
    EDA_DIR = ARTIFACT_DIR / 'eda'
    EDA_IMG_DIR = EDA_DIR / 'images'
    EDA_TABLE_DIR = EDA_DIR / 'tables'
for p in [EDA_DIR, EDA_IMG_DIR, EDA_TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

num_cols_dist = X.select_dtypes(include=[np.number]).columns.tolist()
print(f'Generating per-variable boxplots for {len(num_cols_dist)} numeric features...')

if len(num_cols_dist) > 0:
    n_cols_per_page = 3
    n_rows_per_page = 2
    n_per_page = n_cols_per_page * n_rows_per_page
    n_pages = (len(num_cols_dist) + n_per_page - 1) // n_per_page

    for page_idx in range(n_pages):
        start_idx = page_idx * n_per_page
        end_idx = min(start_idx + n_per_page, len(num_cols_dist))
        page_cols = num_cols_dist[start_idx:end_idx]

        fig, axes = plt.subplots(n_rows_per_page, n_cols_per_page, figsize=(15, 8))
        axes = np.array(axes).reshape(-1)

        for ax_idx, col in enumerate(page_cols):
            ax = axes[ax_idx]
            data_to_plot = [X[col][y == 0].dropna(), X[col][y == 1].dropna()]
            bp = ax.boxplot(data_to_plot, tick_labels=['No HTN', 'HTN'], patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#1976d2', '#c62828']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        for ax_idx in range(len(page_cols), len(axes)):
            axes[ax_idx].set_visible(False)

        plt.tight_layout()
        dist_path = EDA_IMG_DIR / f'numeric_distributions_by_target_page{page_idx+1}.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Saved numeric boxplot page {page_idx+1}/{n_pages}: {dist_path}')

    if 'target_relation_df' in globals() and not target_relation_df.empty:
        top_feature_list = target_relation_df['feature'].head(min(12, len(target_relation_df))).tolist()
    else:
        top_feature_list = num_cols_dist[:min(12, len(num_cols_dist))]

    violin_long_df = X[top_feature_list].copy()
    violin_long_df['target_label'] = y.map({0: 'No HTN', 1: 'HTN'}).fillna(y.astype(str))
    violin_long_df = violin_long_df.melt(id_vars='target_label', var_name='feature', value_name='value').dropna()

    if not violin_long_df.empty:
        violin_per_page = 6
        violin_pages = (len(top_feature_list) + violin_per_page - 1) // violin_per_page
        for page_idx in range(violin_pages):
            page_features = top_feature_list[page_idx * violin_per_page:(page_idx + 1) * violin_per_page]
            plot_df = violin_long_df[violin_long_df['feature'].isin(page_features)].copy()
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            axes = np.array(axes).reshape(-1)
            for ax_idx, feature_name in enumerate(page_features):
                ax = axes[ax_idx]
                sns.violinplot(
                    data=plot_df[plot_df['feature'] == feature_name],
                    x='target_label',
                    y='value',
                    palette=['#1976d2', '#c62828'],
                    cut=0,
                    inner='quartile',
                    ax=ax,
                )
                ax.set_title(feature_name, fontsize=10, fontweight='bold')
                ax.set_xlabel('')
                ax.grid(axis='y', alpha=0.3)
            for ax_idx in range(len(page_features), len(axes)):
                axes[ax_idx].set_visible(False)
            plt.tight_layout()
            violin_path = EDA_IMG_DIR / f'top_numeric_violin_by_target_page{page_idx+1}.png'
            plt.savefig(violin_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f'Saved numeric violin page {page_idx+1}/{violin_pages}: {violin_path}')

        density_per_page = 4
        density_pages = (len(top_feature_list) + density_per_page - 1) // density_per_page
        for page_idx in range(density_pages):
            page_features = top_feature_list[page_idx * density_per_page:(page_idx + 1) * density_per_page]
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes = np.array(axes).reshape(-1)
            for ax_idx, feature_name in enumerate(page_features):
                ax = axes[ax_idx]
                plot_data = pd.DataFrame({
                    'value': pd.to_numeric(X[feature_name], errors='coerce'),
                    'target_label': y.map({0: 'No HTN', 1: 'HTN'}).fillna(y.astype(str)),
                }).dropna()
                sns.histplot(
                    data=plot_data,
                    x='value',
                    hue='target_label',
                    stat='density',
                    common_norm=False,
                    bins=30,
                    kde=True,
                    palette=['#1976d2', '#c62828'],
                    alpha=0.35,
                    ax=ax,
                )
                ax.set_title(feature_name, fontsize=10, fontweight='bold')
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Density')
                ax.grid(axis='y', alpha=0.3)
            for ax_idx in range(len(page_features), len(axes)):
                axes[ax_idx].set_visible(False)
            plt.tight_layout()
            density_path = EDA_IMG_DIR / f'top_numeric_density_by_target_page{page_idx+1}.png'
            plt.savefig(density_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f'Saved numeric density page {page_idx+1}/{density_pages}: {density_path}')
else:
    print('No numeric columns found.')


# In[12]:


# Categorical distributions: count plots for categorical features vs target
if 'EDA_DIR' not in globals():
    EDA_DIR = ARTIFACT_DIR / 'eda'
    EDA_IMG_DIR = EDA_DIR / 'images'
    EDA_TABLE_DIR = EDA_DIR / 'tables'
for p in [EDA_DIR, EDA_IMG_DIR, EDA_TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

cat_cols_dist = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'Found {len(cat_cols_dist)} categorical features.')

if len(cat_cols_dist) > 0:
    # Create a grid of subplots (3 per page)
    n_per_page = 3
    n_pages = (len(cat_cols_dist) + n_per_page - 1) // n_per_page

    for page_idx in range(n_pages):
        start_idx = page_idx * n_per_page
        end_idx = min(start_idx + n_per_page, len(cat_cols_dist))
        page_cols = cat_cols_dist[start_idx:end_idx]

        fig, axes = plt.subplots(1, len(page_cols), figsize=(18, 5))
        if len(page_cols) == 1:
            axes = [axes]

        for ax_idx, col in enumerate(page_cols):
            ax = axes[ax_idx]

            # Build cross-tab: rows=categories, cols=target
            ct = pd.crosstab(X[col].fillna('Missing'), y, margins=False)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

            ct_pct.plot(kind='bar', ax=ax, color=['#1976d2', '#c62828'], alpha=0.8, width=0.7)
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_xlabel('Category', fontsize=10)
            ax.set_ylabel('% within Category', fontsize=10)
            ax.legend(['No HTN', 'HTN'], loc='upper right', fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        cat_path = EDA_IMG_DIR / f'categorical_distributions_page{page_idx+1}.png'
        plt.savefig(cat_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Saved categorical distributions page {page_idx+1}/{n_pages}: {cat_path}')
else:
    print('No categorical columns found.')

