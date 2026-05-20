# Generated from: EXP G sampling exp.ipynb
# Converted at: 2026-05-20T02:10:30.367Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Install required packages (idempotent, safe to re-run)
import sys
!{sys.executable} -m pip install --quiet lightgbm scikit-learn pandas numpy matplotlib joblib imblearn

# # EXP 3 G Sampling Experiment
# 
# Reliability check for `base`, `cw`, `smote`, `smote+cw`, `smotenc`, and `smote+nc` using the same EXP3 preprocessing bundle used by the app.
# Focus: detect physiologically/medically impossible synthetic patients.


import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)

# -- Utility helpers (mirrors Main_2015_Training_Core + EXP3-B) --------------

def _find_col(columns, aliases):
    lc = {c.lower(): c for c in columns}
    for a in aliases:
        if a.lower() in lc:
            return lc[a.lower()]
    for c in columns:
        if any(a.lower() in c.lower() for a in aliases):
            return c
    return None

def _to_num(s):
    return pd.to_numeric(s, errors='coerce').replace([9, 99, 888888, 999999], np.nan)

def _build_smoking_level(df_):
    sl = _find_col(df_.columns, ['smoking_level'])
    if sl:
        return _to_num(df_[sl]).clip(0, 3).astype(float), [sl]
    ss  = _find_col(df_.columns, ['smoke_status'])
    cs  = _find_col(df_.columns, ['current_smoking', 'currentsmoking'])
    es  = _find_col(df_.columns, ['ever_smk'])
    used, out = [], pd.Series(np.nan, index=df_.index, dtype=float)
    if ss:
        status = _to_num(df_[ss]); used.append(ss)
        out[status == 0] = 0; out[status == 2] = 1; out[status == 1] = 2
        if cs:
            cur = _to_num(df_[cs]); used.append(cs)
            out[(status == 1) & (cur == 3)] = 3
        return out, used
    if cs:
        cur = _to_num(df_[cs]); used.append(cs)
        out[cur == 0] = 0; out[cur.isin([1, 2])] = 2; out[cur == 3] = 3
        return out, used
    if es:
        ever = _to_num(df_[es]); used.append(es)
        out[ever == 0] = 0; out[ever > 0] = 1
        return out, used
    return None, []

def _build_alcohol_level(df_):
    al = _find_col(df_.columns, ['alcohol_level'])
    if al:
        return _to_num(df_[al]).clip(0, 3).astype(float), [al]
    as_ = _find_col(df_.columns, ['alcohol_status'])
    bg  = _find_col(df_.columns, ['binge_drink', 'binge_drinking'])
    ca  = _find_col(df_.columns, ['con_alcohol'])
    d30 = _find_col(df_.columns, ['drnk_30days'])
    ae  = _find_col(df_.columns, ['alcohol'])
    used, out = [], pd.Series(np.nan, index=df_.index, dtype=float)
    if as_:
        status = _to_num(df_[as_]); used.append(as_)
        out[status == 0] = 0; out[status == 2] = 1; out[status == 1] = 2
        if bg:
            binge = _to_num(df_[bg]); used.append(bg)
            out[(status == 1) & (binge == 1)] = 3
        return out, used
    out[:] = 0
    if ae:
        ever = _to_num(df_[ae]);   used.append(ae);   out[ever > 0] = 1
    if ca:
        cur  = _to_num(df_[ca]);   used.append(ca);   out[cur == 1] = np.maximum(out[cur == 1], 2)
    if d30:
        d    = _to_num(df_[d30]);  used.append(d30);  out[d == 1]   = np.maximum(out[d == 1], 2)
    if bg:
        b    = _to_num(df_[bg]);   used.append(bg);   out[b == 1]   = 3
    return (out, used) if used else (None, [])

def _build_bmi(df_):
    w = _find_col(df_.columns, ['weight'])
    h = _find_col(df_.columns, ['height'])
    if not w or not h:
        return None, []
    wt = pd.to_numeric(df_[w], errors='coerce')
    ht = pd.to_numeric(df_[h], errors='coerce')
    if pd.notna(ht.median()) and float(ht.median()) > 3.0:
        ht = ht / 100.0
    return (wt / ht**2).replace([np.inf, -np.inf], np.nan).astype(float), [w, h]

def _build_whr(df_):
    wc = _find_col(df_.columns, ['waist'])
    hc = _find_col(df_.columns, ['hip'])
    if not wc or not hc:
        return None, []
    waist = pd.to_numeric(df_[wc], errors='coerce')
    hip   = pd.to_numeric(df_[hc], errors='coerce').replace(0, np.nan)
    return (waist / hip).replace([np.inf, -np.inf], np.nan).astype(float), [wc, hc]

# -- Merge from Datasets2015 (mirrors Main_2015_Training_Core merge logic) -----
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

def _find_dataset(folder, hint=''):
    folder = Path(folder)
    cands  = sorted([
        p for p in folder.glob('*.csv')
        if 'data-set' in p.name.lower() and 'dictionary' not in p.name.lower()
    ])
    return next((p for p in cands if hint in p.name.lower()), cands[0] if cands else None)

KEY_PRIORITY = [
    ['enns_year', 'hhnum', 'member_code'],
    ['hhnum', 'member_code'],
    ['enns_year', 'hhnum'],
    ['hhnum'],
]

def _best_keys(left, right):
    for keys in KEY_PRIORITY:
        if all(k in left.columns and k in right.columns for k in keys):
            return keys
    return []

clin_path = _find_dataset(DS / 'Clinical',       'clinical')
diet_path = _find_dataset(DS / 'Dietary',         'dietary')
anth_path = _find_dataset(DS / 'Anthropometric',  'anthrop')

if clin_path is None:
    raise FileNotFoundError(f'Clinical CSV not found in {DS}/Clinical/')

clin_raw = pd.read_csv(clin_path, low_memory=False)
diet_raw = pd.read_csv(diet_path, low_memory=False) if diet_path else None
anth_raw = pd.read_csv(anth_path, low_memory=False) if anth_path else None

# Left-join dietary onto clinical
df = clin_raw.copy()
if diet_raw is not None:
    merge_keys = _best_keys(df, diet_raw)
    if not merge_keys:
        raise ValueError('No common merge keys between clinical and dietary.')
    diet_work = diet_raw.drop_duplicates(subset=merge_keys, keep='first')
    overlap   = [c for c in diet_work.columns if c in df.columns and c not in merge_keys]
    diet_work = diet_work.drop(columns=overlap, errors='ignore')
    df        = df.merge(diet_work, on=merge_keys, how='left')
    print(f'After dietary merge : {df.shape}  keys={merge_keys}')

# Left-join anthropometric onto result
if anth_raw is not None:
    merge_keys = _best_keys(df, anth_raw)
    if merge_keys:
        anth_work = anth_raw.drop_duplicates(subset=merge_keys, keep='first')
        overlap   = [c for c in anth_work.columns if c in df.columns and c not in merge_keys]
        anth_work = anth_work.rename(columns={c: f'{c}_anth' for c in overlap})
        df        = df.merge(anth_work, on=merge_keys, how='left')
        print(f'After anthro merge  : {df.shape}  keys={merge_keys}')

print(f'Using data root    : {ROOT}')
print(f'Final merged shape : {df.shape}')

# -- Target engineering (Hypertension from SBP/DBP, mirrors Main_2015_Training_Core) --
sbp_col = _find_col(df.columns, ['ave_sbp', 'sbp', 'systolic', 'sysbp'])
dbp_col = _find_col(df.columns, ['ave_dbp', 'dbp', 'diastolic', 'diabp'])

if sbp_col and dbp_col:
    sbp = pd.to_numeric(df[sbp_col], errors='coerce')
    dbp = pd.to_numeric(df[dbp_col], errors='coerce')
    df['Hypertension'] = ((sbp >= 140) | (dbp >= 90)).fillna(False).astype(int)
    TARGET_COL = 'Hypertension'
    print(f'Target created from {sbp_col}, {dbp_col}  (>=140/90 rule)')
else:
    TARGET_COL = _find_col(df.columns, ['hypertension', 'htn', 'target', 'label', 'outcome'])
    if TARGET_COL is None:
        raise ValueError(f'Cannot derive target. Columns: {list(df.columns[:30])}')
    print(f'Target column found : {TARGET_COL}')

df = df.dropna(subset=[TARGET_COL]).copy()
print('Class balance:', df[TARGET_COL].value_counts(normalize=True).round(3).to_dict())

# -- Feature engineering (mirrors EXP3-B) --------------------------------------
X_raw = df.drop(columns=[TARGET_COL]).copy()
y     = df[TARGET_COL].astype(int).copy()

smk, smk_src = _build_smoking_level(X_raw)
if smk is not None:
    X_raw['fe_smoking_level'] = smk

alc, alc_src = _build_alcohol_level(X_raw)
if alc is not None:
    X_raw['fe_alcohol_level'] = alc

bmi, bmi_src = _build_bmi(X_raw)
if bmi is not None:
    X_raw['bmi'] = bmi

whr, whr_src = _build_whr(X_raw)
if whr is not None:
    X_raw['whr'] = whr

# -- Drop non-predictive / raw source columns (mirrors EXP3-B manual_non_predictive) --
DROP_ALWAYS = [
    sbp_col, dbp_col,
    'height', 'weight', 'waist', 'hip',
    'enns_year', 'hhnum', 'member_code',
    'regcode', 'provcode', 'psc', 'csc', 'rhc', 'psurec', 'strrec',
    'wgts', 'finalwgt', 'finalwgt1', 'finalwgt4',
    'rep_natl', 'rep_prov', 'ms_psucode', 'wrkplace',
    'interview_status', 'intdate', 'enumcode',
    'alcohol', 'con_alcohol', 'drnk_30days', 'drnk_30d_num', 'alcohol_status',
    'binge_drinking', 'current_smoking', 'ever_smk', 'smoke_status', 'smoking_level', 'alcohol_level',
]
DROP_ALWAYS += list(set(bmi_src + whr_src + smk_src + alc_src))
lc_map = {c.lower(): c for c in X_raw.columns}
to_drop = sorted({lc_map[d.lower()] for d in DROP_ALWAYS if d and d.lower() in lc_map})
X_raw = X_raw.drop(columns=to_drop, errors='ignore')

# Deduplicate age/sex alias columns from anthro join
for _alias in ['age', 'sex']:
    _cands = [c for c in X_raw.columns if _alias in c.lower()]
    if len(_cands) > 1:
        _keep  = next((c for c in _cands if c.lower() == _alias), _cands[0])
        X_raw  = X_raw.drop(columns=[c for c in _cands if c != _keep], errors='ignore')

X = X_raw.copy()
print(f'Features: {X.shape[1]}  | Rows: {len(X)}  | Positives: {int(y.sum())}')

# ── EXP3-B style preprocessing: 3-way split + KNN impute + scale/OHE ─────────
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def _ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def e3_fit_imputers(X_tr, num_cols, cat_cols):
    knn = KNNImputer(n_neighbors=5)
    if num_cols:
        knn.fit(X_tr[num_cols])
    cat_imp = None
    if cat_cols:
        cat_imp = SimpleImputer(strategy='most_frequent')
        cat_imp.fit(X_tr[cat_cols])
    return knn, cat_imp

def e3_impute(knn, cat_imp, X_df, num_cols, cat_cols):
    frames = []
    if num_cols:
        frames.append(pd.DataFrame(knn.transform(X_df[num_cols]), columns=num_cols, index=X_df.index))
    if cat_cols and cat_imp is not None:
        frames.append(pd.DataFrame(cat_imp.transform(X_df[cat_cols]), columns=cat_cols, index=X_df.index))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=X_df.index)

def e3_fit_enc_scaler(X_imp, num_cols, cat_cols):
    scaler = StandardScaler()
    ohe_enc = None
    if num_cols:
        scaler.fit(X_imp[num_cols])
    if cat_cols:
        ohe_enc = _ohe()
        ohe_enc.fit(X_imp[cat_cols].astype(str))
    return scaler, ohe_enc

def e3_encode(scaler, ohe_enc, X_imp, num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(scaler.transform(X_imp[num_cols]))
    if cat_cols and ohe_enc is not None:
        parts.append(ohe_enc.transform(X_imp[cat_cols].astype(str)))
    return np.hstack(parts) if parts else np.empty((len(X_imp), 0))

# 3-way split: 60% train, 20% cal, 20% test (mirrors EXP3-B)
X_tr_raw, X_tmp_raw, y_train, y_tmp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_cal_raw, X_test_raw, y_cal, y_test = train_test_split(
    X_tmp_raw, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

# Detect numeric / categorical on training split

# Always treat these as categorical for SMOTENC, even if numeric
force_cat = ['fe_smoking_level', 'fe_alcohol_level']
# Try to find an ethnicity column (case-insensitive, partial match)
ethnicity_col = next((c for c in X_tr_raw.columns if 'ethnic' in c.lower()), None)
if ethnicity_col:
    force_cat.append(ethnicity_col)

e3_num_cols = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()
e3_cat_cols = [c for c in X_tr_raw.columns if c not in e3_num_cols or c in force_cat]
e3_num_cols = [c for c in e3_num_cols if c not in force_cat]

# Fit imputers and scalers on train only
knn_imp, cat_imp = e3_fit_imputers(X_tr_raw, e3_num_cols, e3_cat_cols)

X_tr_imp  = e3_impute(knn_imp, cat_imp, X_tr_raw,   e3_num_cols, e3_cat_cols)
X_cal_imp = e3_impute(knn_imp, cat_imp, X_cal_raw,  e3_num_cols, e3_cat_cols)
X_te_imp  = e3_impute(knn_imp, cat_imp, X_test_raw, e3_num_cols, e3_cat_cols)

scaler_e3, ohe_e3 = e3_fit_enc_scaler(X_tr_imp, e3_num_cols, e3_cat_cols)

# For SMOTENC: build ordinal-encoded train matrix (cat preserved as ordinal)
oe_sampling = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
if e3_cat_cols:
    Xc_ord = pd.DataFrame(
        oe_sampling.fit_transform(X_tr_imp[e3_cat_cols].astype(str)),
        columns=e3_cat_cols, index=X_tr_imp.index
    )
else:
    Xc_ord = pd.DataFrame(index=X_tr_imp.index)

X_train_sampling = pd.concat([X_tr_imp[e3_num_cols] if e3_num_cols else pd.DataFrame(index=X_tr_imp.index), Xc_ord], axis=1)
feature_cols_sampling = list(X_train_sampling.columns)
cat_idx = list(range(len(e3_num_cols), len(e3_num_cols) + len(e3_cat_cols)))

print(pd.DataFrame({
    'Split': ['Train', 'Cal', 'Test'],
    'N':     [len(y_train), len(y_cal), len(y_test)],
    'Pos%':  [f"{y_train.mean()*100:.1f}", f"{y_cal.mean()*100:.1f}", f"{y_test.mean()*100:.1f}"],
}).to_string(index=False))
print(f'\nNumeric: {len(e3_num_cols)} | Categorical: {len(e3_cat_cols)}')
print(f'Sampling matrix shape: {X_train_sampling.shape} | cat_idx for SMOTENC: {len(cat_idx)}')

def impossible_flags(df_sample: pd.DataFrame) -> pd.Series:
    f = pd.Series(False, index=df_sample.index)

    if 'age' in df_sample.columns:
        f |= (df_sample['age'] < 18) | (df_sample['age'] > 120)
    if 'sex' in df_sample.columns:
        f |= ~df_sample['sex'].round().isin([1, 2])
    if 'height' in df_sample.columns:
        f |= (df_sample['height'] <= 0) | (df_sample['height'] > 260)
    if 'weight' in df_sample.columns:
        f |= (df_sample['weight'] <= 0) | (df_sample['weight'] > 350)
    if 'waist' in df_sample.columns:
        f |= (df_sample['waist'] <= 0) | (df_sample['waist'] > 200)
    if 'hip' in df_sample.columns:
        f |= (df_sample['hip'] <= 0) | (df_sample['hip'] > 220)
    if 'BMI' in df_sample.columns:
        f |= (df_sample['BMI'] < 10) | (df_sample['BMI'] > 50)
    if 'bmi' in df_sample.columns:
        f |= (df_sample['bmi'] < 10) | (df_sample['bmi'] > 50)
    if 'whr' in df_sample.columns:
        f |= (df_sample['whr'] < 0.4) | (df_sample['whr'] > 2.0)

    nonneg_prefixes = ('Total_', 'fg', 'epwt_fg')
    for c in df_sample.columns:
        if c.startswith(nonneg_prefixes):
            f |= (pd.to_numeric(df_sample[c], errors='coerce') < 0)

    return f.fillna(True)

def resample_method(name: str, Xs: pd.DataFrame, ys: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if name in {'base', 'cw'}:
        return Xs.copy(), ys.copy()
    if name in {'smote', 'smote+cw'}:
        sampler = SMOTE(random_state=42, k_neighbors=5)
        Xr, yr = sampler.fit_resample(Xs, ys)
        return pd.DataFrame(Xr, columns=Xs.columns), pd.Series(yr)
    if name in {'smotenc', 'smote+nc'}:
        sampler = SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=5)
        Xr, yr = sampler.fit_resample(Xs, ys)
        return pd.DataFrame(Xr, columns=Xs.columns), pd.Series(yr)
    raise ValueError(name)

methods = ['base', 'cw', 'smote', 'smote+cw', 'smotenc', 'smote+nc']
summary_rows = []
examples = {}

n_orig = len(X_train_sampling)

for m in methods:
    Xr, yr = resample_method(m, X_train_sampling, y_train)
    flags_all = impossible_flags(Xr)

    n_all = int(len(Xr))
    n_bad_all = int(flags_all.sum())

    # For over-sampling methods, imblearn appends synthetic rows after originals.
    n_syn = max(n_all - n_orig, 0)
    if n_syn > 0:
        flags_syn = flags_all.iloc[n_orig:]
        n_bad_syn = int(flags_syn.sum())
        syn_bad_pct = 100.0 * n_bad_syn / n_syn
        if n_bad_syn > 0:
            examples[m] = Xr.iloc[n_orig:].loc[flags_syn].head(5).copy()
    else:
        n_bad_syn = 0
        syn_bad_pct = np.nan

    summary_rows.append({
        'method': m,
        'rows_total': n_all,
        'rows_synthetic': n_syn,
        'positive_rate': float(np.mean(yr)),
        'impossible_all_rows': n_bad_all,
        'impossible_all_pct': (100.0 * n_bad_all / max(n_all, 1)),
        'impossible_synth_rows': n_bad_syn,
        'impossible_synth_pct': syn_bad_pct,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(['impossible_synth_pct', 'impossible_all_pct', 'method'], na_position='last').reset_index(drop=True)
summary_df

from sklearn.metrics import roc_auc_score

def build_sampling_matrix(X_imp: pd.DataFrame) -> pd.DataFrame:
    if e3_cat_cols:
        Xc = pd.DataFrame(
            oe_sampling.transform(X_imp[e3_cat_cols].astype(str)),
            columns=e3_cat_cols,
            index=X_imp.index,
        )
    else:
        Xc = pd.DataFrame(index=X_imp.index)

    Xn = X_imp[e3_num_cols] if e3_num_cols else pd.DataFrame(index=X_imp.index)
    return pd.concat([Xn, Xc], axis=1)[feature_cols_sampling]

X_cal_sampling = build_sampling_matrix(X_cal_imp)
X_test_sampling = build_sampling_matrix(X_te_imp)

def make_lgbm(method_name: str) -> lgb.LGBMClassifier:
    params = dict(
        objective='binary',
        random_state=42,
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.0,
        verbosity=-1,
    )
    if method_name in {'cw', 'smote+cw'}:
        params['class_weight'] = 'balanced'

    # Try GPU first; fallback to CPU for environments without OpenCL/CUDA support.
    try:
        return lgb.LGBMClassifier(device='gpu', **params)
    except Exception:
        return lgb.LGBMClassifier(device='cpu', **params)

metric_rows = []
for m in methods:
    Xr, yr = resample_method(m, X_train_sampling, y_train)

    model = make_lgbm(m)
    try:
        model.fit(Xr, yr)
    except Exception:
        # If GPU fit fails at runtime, retry on CPU.
        model = lgb.LGBMClassifier(
            device='cpu',
            objective='binary',
            random_state=42,
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=0.0,
            verbosity=-1,
            class_weight='balanced' if m in {'cw', 'smote+cw'} else None,
        )
        model.fit(Xr, yr)

    proba = model.predict_proba(X_test_sampling)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metric_rows.append({
        'method': m,
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0),
        'f1': f1_score(y_test, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, proba),
    })

metrics_df = pd.DataFrame(metric_rows).sort_values('f1', ascending=False).reset_index(drop=True)
comparison_df = summary_df.merge(metrics_df, on='method', how='left')
comparison_df[['method', 'rows_synthetic', 'impossible_synth_pct', 'impossible_all_pct', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].sort_values('f1', ascending=False)

import matplotlib.pyplot as plt

plot_df = summary_df.copy()
plot_df['impossible_synth_pct_plot'] = plot_df['impossible_synth_pct'].fillna(0.0)

plt.figure(figsize=(9, 4))
plt.bar(plot_df['method'], plot_df['impossible_synth_pct_plot'])
plt.ylabel('Impossible synthetic patients (%)')
plt.xlabel('Sampling strategy')
plt.title('Synthetic-only Plausibility Check per Sampling Method')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()

# Inspect first impossible examples per method (if any).
for m in methods:
    print('\n===', m, '===')
    if m not in examples:
        print('No impossible rows detected.')
    else:
        display(examples[m])