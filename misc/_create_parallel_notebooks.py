"""
Create 3 parallel EXP3 training notebooks from the source notebook.
  EXP3_A_KNN_RF.ipynb          — KNN + RandomForest (RF uses cuML GPU)
  EXP3_B_XGB_AdaBoost.ipynb    — XGBoost (GPU) + AdaBoost
  EXP3_C_LogReg_CatBoost.ipynb — Logistic Regression + CatBoost (GPU)
"""
import json, copy, re

SRC = '/workspace/Thesis-part-2/Main_2015_GPU_RF_XGB_CAT_RIGOROUS_OPT_exp2.ipynb'
with open(SRC) as f:
    nb = json.load(f)
cells = nb['cells']

def src(cell):
    s = cell.get('source', '')
    return ''.join(s) if isinstance(s, list) else s

def clean(cell, new_source=None):
    c = copy.deepcopy(cell)
    if new_source is not None:
        c['source'] = new_source
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
    c.setdefault('metadata', {})
    return c

# ── cuML GPU RandomForest patch (Notebook A only) ─────────────────────────────
RF_OLD = \
"""    if model_name == 'randomforest':
        raw_mf = p.get('max_features', 'sqrt')
        if isinstance(raw_mf, float):
            raw_mf = float(np.clip(raw_mf, 0.01, 1.0))
        return RandomForestClassifier(
            n_estimators=int(epoch_budget),
            max_depth=p.get('max_depth', None),
            min_samples_split=max(2, int(round(p.get('min_samples_split', 2)))),
            min_samples_leaf=max(1, int(round(p.get('min_samples_leaf', 1)))),
            max_features=raw_mf,
            random_state=seed, n_jobs=-1,
            class_weight='balanced_subsample' if use_cw else None,
        )
    raise ValueError(f"Unknown model: {model_name}")"""

RF_CUML = \
"""    if model_name == 'randomforest':
        raw_mf = p.get('max_features', 'sqrt')
        if isinstance(raw_mf, float):
            raw_mf = float(np.clip(raw_mf, 0.01, 1.0))
        max_d  = p.get('max_depth', None)
        # Use cuML GPU RandomForest when available and no class-weight required
        _use_cuml_rf = bool(use_gpu and cuml_available and cuRFClassifier is not None and not use_cw)
        if _use_cuml_rf:
            # cuML RF: GPU-accelerated; max_depth must be a finite int (not None)
            cuml_max_d = int(max_d) if max_d is not None else 32
            cuml_mf    = raw_mf if isinstance(raw_mf, float) else 0.5
            return cuRFClassifier(
                n_estimators=int(epoch_budget),
                max_depth=max(1, cuml_max_d),
                min_samples_split=max(2, int(round(p.get('min_samples_split', 2)))),
                min_samples_leaf=max(1, int(round(p.get('min_samples_leaf', 1)))),
                max_features=cuml_mf,
                random_state=seed,
                n_streams=1,
            )
        # CPU sklearn RF (always used for class-weight sampling variants)
        return RandomForestClassifier(
            n_estimators=int(epoch_budget),
            max_depth=max_d,
            min_samples_split=max(2, int(round(p.get('min_samples_split', 2)))),
            min_samples_leaf=max(1, int(round(p.get('min_samples_leaf', 1)))),
            max_features=raw_mf,
            random_state=seed, n_jobs=-1,
            class_weight='balanced_subsample' if use_cw else None,
        )
    raise ValueError(f"Unknown model: {model_name}")"""

# ── Collinearity filter block (inserted after KNN imputation) ─────────────────
COLL_ANCHOR = "X_e3_te_imp  = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_te,  e3_num_cols, e3_cat_cols)"

COLL_BLOCK = """
# ── Collinearity filter (cutoff = COLLINEARITY_CUTOFF, mirrors original experiment) ──
# Protects age, sex, bmi, whr from ever being dropped.
_e3_protected = sorted({
    c for c in e3_num_cols
    if any(a in c.lower() for a in ['age', 'sex', 'bmi', 'whr'])
})
_e3_corr  = X_e3_tr_imp[e3_num_cols].corr().abs()
_e3_upper = _e3_corr.where(np.triu(np.ones(_e3_corr.shape), k=1).astype(bool))
_e3_drop  = [
    c for c in _e3_upper.columns
    if c not in _e3_protected and (_e3_upper[c] > COLLINEARITY_CUTOFF).any()
]
e3_num_cols  = [c for c in e3_num_cols if c not in _e3_drop]   # updated globally
_e3_all_keep = e3_num_cols + e3_cat_cols
X_e3_tr_imp  = X_e3_tr_imp[_e3_all_keep]
X_e3_cal_imp = X_e3_cal_imp[_e3_all_keep]
X_e3_te_imp  = X_e3_te_imp[_e3_all_keep]
print(f"Collinearity filter (cutoff={COLLINEARITY_CUTOFF}): "
      f"dropped {len(_e3_drop)} → kept {len(e3_num_cols)} numeric features")
if _e3_drop:
    print("  Dropped:", _e3_drop)
if _e3_protected:
    print("  Protected:", _e3_protected)
"""

# ── Notebook configurations ────────────────────────────────────────────────────
CONFIGS = [
    dict(
        fname   = 'EXP3_A_KNN_RF',
        title   = (
            '# EXP3-A · KNN + Random Forest (GPU)\n\n'
            'Trains **KNN** and **Random Forest** with 2-stage optimization '
            'across 6 sampling methods.  \n'
            'RandomForest uses **cuML GPU** acceleration when available '
            '(falls back to sklearn for class-weight variants).  \n'
            'Run in parallel with EXP3-B and EXP3-C.'
        ),
        models  = ['knn', 'randomforest'],
        dir_name= 'exp3_knn_rf',
        cuml_rf = True,
    ),
    dict(
        fname   = 'EXP3_B_XGB_AdaBoost',
        title   = (
            '# EXP3-B · XGBoost (GPU) + AdaBoost\n\n'
            'Trains **XGBoost** (CUDA GPU) and **AdaBoost** with 2-stage optimization '
            'across 6 sampling methods.  \n'
            'Run in parallel with EXP3-A and EXP3-C.'
        ),
        models  = ['xgboost', 'adaboost'],
        dir_name= 'exp3_xgb_ada',
        cuml_rf = False,
    ),
    dict(
        fname   = 'EXP3_C_LogReg_CatBoost',
        title   = (
            '# EXP3-C · Logistic Regression + CatBoost (GPU)\n\n'
            'Trains **Logistic Regression** and **CatBoost** (CUDA GPU) with 2-stage '
            'optimization across 6 sampling methods.  \n'
            'Run in parallel with EXP3-A and EXP3-B.'
        ),
        models  = ['logreg', 'catboost'],
        dir_name= 'exp3_logreg_cat',
        cuml_rf = False,
    ),
]

# ── Cell indices (0-based) in source notebook ─────────────────────────────────
IDX = dict(
    install     = 1,
    cuda        = 2,
    imports     = 3,
    config      = 4,
    load_fns    = 5,
    load_data   = 6,
    # EXP3 section
    exp3_setup  = 33,
    split       = 34,
    helpers     = 35,
    train       = 36,
    precal      = 37,
    cal         = 38,
    postcal     = 39,
    shap        = 40,
    lime        = 41,
)

meta = copy.deepcopy(nb.get('metadata', {}))

for cfg in CONFIGS:
    new_cells = []

    # ── 1. Notebook-specific markdown header ──────────────────────────────────
    hdr = copy.deepcopy(cells[0])
    hdr['source'] = cfg['title']
    hdr.setdefault('metadata', {})
    new_cells.append(hdr)

    # ── 2–7. Common preprocessing cells (install → data loading) ─────────────
    for i in [IDX['install'], IDX['cuda'], IDX['imports'],
              IDX['config'], IDX['load_fns'], IDX['load_data']]:
        new_cells.append(clean(cells[i]))

    # ── 8. EXP3 imports + setup (MODEL_NAMES and E3_DIR modified) ─────────────
    setup_src = src(cells[IDX['exp3_setup']])
    # Replace MODEL_NAMES list
    setup_src = re.sub(
        r"MODEL_NAMES\s*=\s*\[.*?\]",
        "MODEL_NAMES      = " + str(cfg['models']),
        setup_src,
    )
    # Replace E3_DIR path
    setup_src = re.sub(
        r"E3_DIR\s*=\s*PROJECT_ROOT\s*/\s*'[^']*'",
        f"E3_DIR = PROJECT_ROOT / '{cfg['dir_name']}'",
        setup_src,
    )
    new_cells.append(clean(cells[IDX['exp3_setup']], setup_src))

    # ── 9. 3-way split (identical) ────────────────────────────────────────────
    new_cells.append(clean(cells[IDX['split']]))

    # ── 10. Helper functions (Notebook A: patch cuML RF) ─────────────────────
    helpers_src = src(cells[IDX['helpers']])
    if cfg['cuml_rf']:
        if RF_OLD in helpers_src:
            helpers_src = helpers_src.replace(RF_OLD, RF_CUML)
            print(f"[{cfg['fname']}] cuML RF patch applied ✓")
        else:
            print(f"[{cfg['fname']}] WARNING: RF_OLD string not found in helpers — patch skipped")
    new_cells.append(clean(cells[IDX['helpers']], helpers_src))

    # ── 11. Training cell (add collinearity filter + MODEL_NAMES already set) ─
    train_src = src(cells[IDX['train']])
    if '_e3_protected' not in train_src:
        if COLL_ANCHOR in train_src:
            train_src = train_src.replace(COLL_ANCHOR, COLL_ANCHOR + '\n' + COLL_BLOCK)
            print(f"[{cfg['fname']}] Collinearity filter inserted ✓")
        else:
            print(f"[{cfg['fname']}] WARNING: COLL_ANCHOR not found in training cell — skipped")
    else:
        print(f"[{cfg['fname']}] Collinearity filter already present ✓")
    # Update header comment
    train_src = train_src.replace(
        '# ── EXP3 · KNN Imputation + Scale/OHE Fit ───────────────────────────────────',
        f'# ── {cfg["fname"]} · KNN Imputation + Collinearity Filter + Scale/OHE Fit ─────────────',
    ).replace(
        '# ── EXP3 · KNN Imputation + Collinearity Filter + Scale/OHE Fit ─────────────',
        f'# ── {cfg["fname"]} · KNN Imputation + Collinearity Filter + Scale/OHE Fit ─────────────',
    )
    new_cells.append(clean(cells[IDX['train']], train_src))

    # ── 12–16. Post-processing cells (pre-cal, cal, post-cal, SHAP, LIME) ─────
    for i in [IDX['precal'], IDX['cal'], IDX['postcal'], IDX['shap'], IDX['lime']]:
        new_cells.append(clean(cells[i]))

    # ── Write notebook ─────────────────────────────────────────────────────────
    new_nb = {
        'nbformat'      : 4,
        'nbformat_minor': 5,
        'metadata'      : meta,
        'cells'         : new_cells,
    }
    out = f'/workspace/Thesis-part-2/{cfg["fname"]}.ipynb'
    with open(out, 'w') as f:
        json.dump(new_nb, f, indent=1)
    print(f'Created: {out}  ({len(new_cells)} cells)')

print('\nAll done.')
