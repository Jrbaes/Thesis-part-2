"""Generate EXP3.2 notebooks from EXP3.1 notebooks with improved hyperparameters."""
import json, re
from pathlib import Path

ROOT = Path('/workspace/Thesis-part-2')

# ── Improved cell-11 source for each notebook ──────────────────────────────────

CELL11_A = r"""# ── EXP3_2_A · KNN + RF · Further Improved Hyperparameter Search ────────────
# Improvements over EXP3.1-A:
#   • Stage-1: 40 trials (↑ from 30), 4 folds (↑ from 3), epoch 120 (↑ from 100)
#   • Stage-2: 8 perturbations (↑ from 6), 6 folds (↑ from 5), epoch 400 (↑ from 300)
#   • Top-K: 8 S1 configs → 4 final models (↑ from 7→3)
#   • Final: 800 epochs (↑ from 600), threshold grid extended to 0.25
#   • KNN: n_neighbors tightened to 3–15 (EXP3.1 best were ≤15)
#   • RF:  max_features focused 0.15–0.45; min_samples_leaf 1–4; depth [None,10,15,20]

from sklearn.model_selection import ParameterSampler

e3_knn_imp, e3_cat_imp = e3_fit_imputers(X_e3_tr, e3_num_cols, e3_cat_cols)

X_e3_tr_imp  = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_tr,  e3_num_cols, e3_cat_cols)
X_e3_cal_imp = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_cal, e3_num_cols, e3_cat_cols)
X_e3_te_imp  = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_te,  e3_num_cols, e3_cat_cols)

# ── Collinearity filter (cutoff = COLLINEARITY_CUTOFF, mirrors original experiment) ──
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
e3_num_cols  = [c for c in e3_num_cols if c not in _e3_drop]
_e3_all_keep = e3_num_cols + e3_cat_cols
X_e3_tr_imp  = X_e3_tr_imp[_e3_all_keep]
X_e3_cal_imp = X_e3_cal_imp[_e3_all_keep]
X_e3_te_imp  = X_e3_te_imp[_e3_all_keep]
print(f"Collinearity filter (cutoff={COLLINEARITY_CUTOFF}): "
      f"dropped {len(_e3_drop)} → kept {len(e3_num_cols)} numeric features")
if _e3_drop:    print("  Dropped:", _e3_drop)
if _e3_protected: print("  Protected:", _e3_protected)

e3_scaler, e3_ohe, e3_feat_names = e3_fit_enc_scaler(X_e3_tr_imp, e3_num_cols, e3_cat_cols)
X_e3_cal_proc = e3_encode_scale(e3_scaler, e3_ohe, X_e3_cal_imp, e3_num_cols, e3_cat_cols)
X_e3_te_proc  = e3_encode_scale(e3_scaler, e3_ohe, X_e3_te_imp,  e3_num_cols, e3_cat_cols)
print(f"Cal : {X_e3_cal_proc.shape}  |  Test: {X_e3_te_proc.shape}  |  Features: {len(e3_feat_names)}")

# ── 2-Stage Optimization Config (FURTHER IMPROVED) ───────────────────────────
E3_S1_TRIALS    = 40    # ↑ from 30
E3_S1_EPOCHS    = 120   # ↑ from 100
E3_S1_FOLDS     = 4     # ↑ from 3
E3_TOP_K_S1     = 8     # ↑ from 7

E3_S2_REFINE    = 8     # ↑ from 6
E3_S2_EPOCHS    = 400   # ↑ from 300
E3_S2_FOLDS     = 6     # ↑ from 5
E3_TOP_K_S2     = 4     # ↑ from 3

E3_FINAL_EPOCHS = 800   # ↑ from 600
THRESHOLD_GRID  = np.round(np.arange(0.25, 0.60, 0.05), 2)  # extended lower bound

# ── Improved Parameter Search Spaces ─────────────────────────────────────────
# Informed by EXP3.1-A: best KNN n_neighbors ≤15; RF max_features ~0.2–0.4
E3_MODEL_SPACES = {
    'knn': {
        # Tighten to ≤15; distance weighting often helps
        'n_neighbors': randint(3, 16),
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan'],
    },
    'randomforest': {
        # Focus max_features on proven effective range 0.15–0.45
        'max_features':      uniform(0.15, 0.30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf':  randint(1, 5),
        # Finer depth grid around best region
        'max_depth':         [None, 10, 15, 20],
    },
}

_n_s2_cands = E3_TOP_K_S1 * (E3_S2_REFINE + 1)
print(f"\nEXP3.2-A · Further Improved 2-Stage Config")
print(f"Stage 1 : {E3_S1_TRIALS} trials × {E3_S1_FOLDS} folds, top-{E3_TOP_K_S1} kept")
print(f"Stage 2 : ~{_n_s2_cands} candidates × {E3_S2_FOLDS} folds, top-{E3_TOP_K_S2} for final")
print(f"Final   : {E3_FINAL_EPOCHS} epochs, threshold swept on Cal set (grid: {THRESHOLD_GRID.min():.2f}–{THRESHOLD_GRID.max():.2f})")
print(f"Total   : {len(SAMPLING_METHODS)} samplings × {len(MODEL_NAMES)} models = {len(SAMPLING_METHODS)*len(MODEL_NAMES)} combos")

# ── 2-Stage Optimization + Final Training Loop ────────────────────────────────
e3_stage1_results = {}
e3_stage2_results = {}
e3_best_configs   = {}
e3_results        = []
e3_models         = {}
e3_tr_data        = {}

for sampling in SAMPLING_METHODS:
    use_cw = sampling in CW_SAMPLINGS
    print(f"\n{'═'*68}")
    print(f"Sampling: {sampling:12s}  |  class-weight active: {use_cw}")
    print(f"{'═'*68}")

    for model_name in MODEL_NAMES:
        key = (sampling, model_name)

        trials = list(ParameterSampler(
            E3_MODEL_SPACES[model_name], n_iter=E3_S1_TRIALS, random_state=E3_SEED))
        s1_rows = []
        for i, params in enumerate(trials, start=1):
            try:
                cv_met = e3_evaluate_params_cv(
                    model_name, params, sampling,
                    X_e3_tr_imp, y_e3_tr,
                    E3_S1_EPOCHS, E3_S1_FOLDS, seed=E3_SEED)
                s1_rows.append({'trial': i, 'params': params, **cv_met})
            except Exception as exc:
                s1_rows.append({'trial': i, 'params': params,
                                'stage_score': -999.0, 'error': str(exc)})

        df_s1 = (pd.DataFrame(s1_rows)
                   .sort_values('stage_score', ascending=False)
                   .reset_index(drop=True))
        e3_stage1_results[key] = df_s1
        top_s1_params = df_s1.head(E3_TOP_K_S1)['params'].tolist()
        s1_best_score = float(df_s1.iloc[0].get('stage_score', -999))

        candidates = e3_refine_candidates(top_s1_params, n_refine=E3_S2_REFINE, seed=E3_SEED)
        s2_rows = []
        for i, params in enumerate(candidates, start=1):
            try:
                cv_met = e3_evaluate_params_cv(
                    model_name, params, sampling,
                    X_e3_tr_imp, y_e3_tr,
                    E3_S2_EPOCHS, E3_S2_FOLDS, seed=E3_SEED)
                s2_rows.append({'trial': i, 'params': params, **cv_met})
            except Exception as exc:
                s2_rows.append({'trial': i, 'params': params,
                                'stage_score': -999.0, 'error': str(exc)})

        df_s2 = (pd.DataFrame(s2_rows)
                   .sort_values('stage_score', ascending=False)
                   .reset_index(drop=True))
        e3_stage2_results[key] = df_s2
        best_params_list = df_s2.head(E3_TOP_K_S2)['params'].tolist()
        e3_best_configs[key]   = best_params_list
        s2_best_score = float(df_s2.iloc[0].get('stage_score', -999))

        print(f"  {model_name:14s}  S1={s1_best_score:.4f}  →  S2={s2_best_score:.4f}", end='  ')

        X_tr_s, y_tr_s = e3_get_sampled_train(
            X_e3_tr_imp, y_e3_tr, sampling,
            e3_scaler, e3_ohe, e3_num_cols, e3_cat_cols)
        y_tr_s = np.asarray(y_tr_s, dtype=int)

        final_model = None
        final_score = -np.inf
        final_thr   = 0.5

        for params in best_params_list:
            mdl = e3_build_model_for_search(
                model_name, params, use_cw, y_tr_s, E3_FINAL_EPOCHS, seed=E3_SEED)
            try:
                mdl.fit(X_tr_s, y_tr_s)
            except Exception as exc:
                msg = str(exc).lower()
                if 'gpu' in msg or 'cuda' in msg or 'device' in msg:
                    try:
                        if   model_name == 'catboost': mdl.set_params(task_type='CPU')
                        elif model_name == 'xgboost':  mdl.set_params(device='cpu')
                        mdl.fit(X_tr_s, y_tr_s)
                    except Exception:
                        continue
                else:
                    continue

            p_cal_v = np.clip(mdl.predict_proba(X_e3_cal_proc)[:, 1], 1e-9, 1 - 1e-9)
            local_s, local_thr = -np.inf, 0.5
            for thr in THRESHOLD_GRID:
                yp = (p_cal_v >= thr).astype(int)
                s  = 0.60 * accuracy_score(y_e3_cal, yp) + \
                     0.40 * recall_score(y_e3_cal, yp, zero_division=0)
                if s > local_s:
                    local_s, local_thr = s, float(thr)

            if local_s > final_score:
                final_score, final_model, final_thr = local_s, mdl, local_thr

        if final_model is None:
            print("FAILED — no valid model")
            continue

        y_proba = np.clip(final_model.predict_proba(X_e3_te_proc)[:, 1], 1e-9, 1 - 1e-9)
        met = e3_metrics(y_e3_te, y_proba, final_thr)

        e3_models[key]  = final_model
        e3_tr_data[key] = (X_tr_s, y_tr_s)
        e3_results.append({
            'sampling': sampling, 'model': model_name, 'threshold': final_thr,
            **{k: round(v, 4) for k, v in met.items()},
        })
        print(f"acc={met['accuracy']:.3f}  rec={met['recall']:.3f}  auc={met['auc']:.3f}")

print(f"\n{'═'*68}")
print(f"2-Stage Optimization complete.  Models stored: {len(e3_models)}")

_s1_rows = [{'sampling': s, 'model': m,
              'best_stage1_score': float(df.iloc[0].get('stage_score', np.nan)),
              'best_acc_cv':       float(df.iloc[0].get('accuracy_mean', np.nan)),
              'best_rec_cv':       float(df.iloc[0].get('recall_mean', np.nan))}
            for (s, m), df in e3_stage1_results.items()]
_s2_rows = [{'sampling': s, 'model': m,
              'best_stage2_score': float(df.iloc[0].get('stage_score', np.nan)),
              'best_acc_cv':       float(df.iloc[0].get('accuracy_mean', np.nan)),
              'best_rec_cv':       float(df.iloc[0].get('recall_mean', np.nan)),
              'best_thr_cv':       float(df.iloc[0].get('threshold_mean', np.nan))}
            for (s, m), df in e3_stage2_results.items()]

pd.DataFrame(_s1_rows).to_csv(E3_DIR / 'stage1_summary.csv', index=False)
pd.DataFrame(_s2_rows).to_csv(E3_DIR / 'stage2_summary.csv', index=False)

print("\n── Stage 2 CV Summary (best score per model × sampling) ──")
display(pd.DataFrame(_s2_rows).sort_values('best_stage2_score', ascending=False).reset_index(drop=True))
"""

CELL11_B = r"""# ── EXP3_2_B · XGBoost + AdaBoost · Further Improved Hyperparameter Search ──
# Improvements over EXP3.1-B:
#   • Stage-1: 50 trials (↑ from 40), 5 folds (↑ from 4), epoch 120 (↑ from 100)
#   • Stage-2: 8 perturbations (↑ from 6), 6 folds (↑ from 5), epoch 400 (↑ from 300)
#   • Top-K: 8 S1 configs → 4 final models (↑ from 7→3)
#   • Final: 800 epochs (↑ from 600), threshold grid extended to 0.25
#   • XGBoost: learning_rate 0.003–0.12; depth 3–6; reg_lambda 0.1–20
#   • AdaBoost: n_estimators 40–120; learning_rate 0.03–0.30; base_depth 1–3

from sklearn.model_selection import ParameterSampler

e3_knn_imp, e3_cat_imp = e3_fit_imputers(X_e3_tr, e3_num_cols, e3_cat_cols)

X_e3_tr_imp  = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_tr,  e3_num_cols, e3_cat_cols)
X_e3_cal_imp = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_cal, e3_num_cols, e3_cat_cols)
X_e3_te_imp  = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_te,  e3_num_cols, e3_cat_cols)

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
e3_num_cols  = [c for c in e3_num_cols if c not in _e3_drop]
_e3_all_keep = e3_num_cols + e3_cat_cols
X_e3_tr_imp  = X_e3_tr_imp[_e3_all_keep]
X_e3_cal_imp = X_e3_cal_imp[_e3_all_keep]
X_e3_te_imp  = X_e3_te_imp[_e3_all_keep]
print(f"Collinearity filter (cutoff={COLLINEARITY_CUTOFF}): "
      f"dropped {len(_e3_drop)} → kept {len(e3_num_cols)} numeric features")
if _e3_drop:    print("  Dropped:", _e3_drop)
if _e3_protected: print("  Protected:", _e3_protected)

e3_scaler, e3_ohe, e3_feat_names = e3_fit_enc_scaler(X_e3_tr_imp, e3_num_cols, e3_cat_cols)
X_e3_cal_proc = e3_encode_scale(e3_scaler, e3_ohe, X_e3_cal_imp, e3_num_cols, e3_cat_cols)
X_e3_te_proc  = e3_encode_scale(e3_scaler, e3_ohe, X_e3_te_imp,  e3_num_cols, e3_cat_cols)
print(f"Cal : {X_e3_cal_proc.shape}  |  Test: {X_e3_te_proc.shape}  |  Features: {len(e3_feat_names)}")

# ── 2-Stage Optimization Config (FURTHER IMPROVED) ───────────────────────────
E3_S1_TRIALS    = 50    # ↑ from 40
E3_S1_EPOCHS    = 120   # ↑ from 100
E3_S1_FOLDS     = 5     # ↑ from 4
E3_TOP_K_S1     = 8     # ↑ from 7

E3_S2_REFINE    = 8     # ↑ from 6
E3_S2_EPOCHS    = 400   # ↑ from 300
E3_S2_FOLDS     = 6     # ↑ from 5
E3_TOP_K_S2     = 4     # ↑ from 3

E3_FINAL_EPOCHS = 800   # ↑ from 600
# Per-model epoch caps (AdaBoost is CPU-only — cap trees hard)
E3_EPOCH_OVERRIDE = {
    'adaboost': {'s1': 60, 's2': 100, 'final': 150},   # ↑ from 50/80/100
}
THRESHOLD_GRID  = np.round(np.arange(0.25, 0.60, 0.05), 2)  # extended lower bound

# ── Improved Parameter Search Spaces ─────────────────────────────────────────
# XGBoost: EXP3.1 best had lr~0.01–0.10, depth 3–5, cw sampling won; tighten
# AdaBoost: shallow trees (depth 1–2) dominate; n_estimators 40–120
E3_MODEL_SPACES = {
    'xgboost': {
        # Tighter lower bound; best were 0.01–0.10
        'learning_rate':    loguniform(0.003, 0.12),
        # Depth 3–5 best; cap at 6
        'max_depth':        randint(3, 7),
        'subsample':        uniform(0.55, 0.40),
        'colsample_bytree': uniform(0.55, 0.40),
        'min_child_weight': randint(1, 8),
        # Best gamma near 0; keep narrow
        'gamma':            uniform(0.0, 0.30),
        # Broader reg_lambda for stronger L2 exploration
        'reg_lambda':       loguniform(0.1, 20.0),
    },
    'adaboost': {
        # Expand slightly from EXP3.1 (30–100) to cover more candidates
        'n_estimators':  randint(40, 121),
        # Tighter range around best values (0.04–0.30)
        'learning_rate': uniform(0.03, 0.27),
        # Depth 1–2 dominate; keep 3 as ceiling
        'base_depth':    randint(1, 4),
    },
}

_n_s2_cands = E3_TOP_K_S1 * (E3_S2_REFINE + 1)
print(f"\nEXP3.2-B · Further Improved 2-Stage Config")
print(f"Stage 1 : {E3_S1_TRIALS} trials × {E3_S1_FOLDS} folds, top-{E3_TOP_K_S1} kept")
print(f"Stage 2 : ~{_n_s2_cands} candidates × {E3_S2_FOLDS} folds, top-{E3_TOP_K_S2} for final")
print(f"Final   : {E3_FINAL_EPOCHS} epochs, threshold swept (grid: {THRESHOLD_GRID.min():.2f}–{THRESHOLD_GRID.max():.2f})")
print(f"Total   : {len(SAMPLING_METHODS)} samplings × {len(MODEL_NAMES)} models = {len(SAMPLING_METHODS)*len(MODEL_NAMES)} combos")

# ── 2-Stage Optimization + Final Training Loop ────────────────────────────────
e3_stage1_results = {}
e3_stage2_results = {}
e3_best_configs   = {}
e3_results        = []
e3_models         = {}
e3_tr_data        = {}

for sampling in SAMPLING_METHODS:
    use_cw = sampling in CW_SAMPLINGS
    print(f"\n{'═'*68}")
    print(f"Sampling: {sampling:12s}  |  class-weight active: {use_cw}")
    print(f"{'═'*68}")

    for model_name in MODEL_NAMES:
        key = (sampling, model_name)
        _ov = E3_EPOCH_OVERRIDE.get(model_name, {})
        _s1_ep    = _ov.get('s1',    E3_S1_EPOCHS)
        _s2_ep    = _ov.get('s2',    E3_S2_EPOCHS)
        _final_ep = _ov.get('final', E3_FINAL_EPOCHS)

        trials = list(ParameterSampler(
            E3_MODEL_SPACES[model_name], n_iter=E3_S1_TRIALS, random_state=E3_SEED))
        s1_rows = []
        for i, params in enumerate(trials, start=1):
            try:
                cv_met = e3_evaluate_params_cv(
                    model_name, params, sampling,
                    X_e3_tr_imp, y_e3_tr,
                    _s1_ep, E3_S1_FOLDS, seed=E3_SEED)
                s1_rows.append({'trial': i, 'params': params, **cv_met})
            except Exception as exc:
                s1_rows.append({'trial': i, 'params': params,
                                'stage_score': -999.0, 'error': str(exc)})

        df_s1 = (pd.DataFrame(s1_rows)
                   .sort_values('stage_score', ascending=False)
                   .reset_index(drop=True))
        e3_stage1_results[key] = df_s1
        top_s1_params = df_s1.head(E3_TOP_K_S1)['params'].tolist()
        s1_best_score = float(df_s1.iloc[0].get('stage_score', -999))

        candidates = e3_refine_candidates(top_s1_params, n_refine=E3_S2_REFINE, seed=E3_SEED)
        s2_rows = []
        for i, params in enumerate(candidates, start=1):
            try:
                cv_met = e3_evaluate_params_cv(
                    model_name, params, sampling,
                    X_e3_tr_imp, y_e3_tr,
                    _s2_ep, E3_S2_FOLDS, seed=E3_SEED)
                s2_rows.append({'trial': i, 'params': params, **cv_met})
            except Exception as exc:
                s2_rows.append({'trial': i, 'params': params,
                                'stage_score': -999.0, 'error': str(exc)})

        df_s2 = (pd.DataFrame(s2_rows)
                   .sort_values('stage_score', ascending=False)
                   .reset_index(drop=True))
        e3_stage2_results[key] = df_s2
        best_params_list = df_s2.head(E3_TOP_K_S2)['params'].tolist()
        e3_best_configs[key]   = best_params_list
        s2_best_score = float(df_s2.iloc[0].get('stage_score', -999))

        print(f"  {model_name:14s}  S1={s1_best_score:.4f}  →  S2={s2_best_score:.4f}", end='  ')

        X_tr_s, y_tr_s = e3_get_sampled_train(
            X_e3_tr_imp, y_e3_tr, sampling,
            e3_scaler, e3_ohe, e3_num_cols, e3_cat_cols)
        y_tr_s = np.asarray(y_tr_s, dtype=int)

        final_model = None
        final_score = -np.inf
        final_thr   = 0.5

        for params in best_params_list:
            mdl = e3_build_model_for_search(
                model_name, params, use_cw, y_tr_s, _final_ep, seed=E3_SEED)
            try:
                mdl.fit(X_tr_s, y_tr_s)
            except Exception as exc:
                msg = str(exc).lower()
                if 'gpu' in msg or 'cuda' in msg or 'device' in msg:
                    try:
                        if   model_name == 'catboost': mdl.set_params(task_type='CPU')
                        elif model_name == 'xgboost':  mdl.set_params(device='cpu')
                        mdl.fit(X_tr_s, y_tr_s)
                    except Exception:
                        continue
                else:
                    continue

            p_cal_v = np.clip(mdl.predict_proba(X_e3_cal_proc)[:, 1], 1e-9, 1 - 1e-9)
            local_s, local_thr = -np.inf, 0.5
            for thr in THRESHOLD_GRID:
                yp = (p_cal_v >= thr).astype(int)
                s  = 0.60 * accuracy_score(y_e3_cal, yp) + \
                     0.40 * recall_score(y_e3_cal, yp, zero_division=0)
                if s > local_s:
                    local_s, local_thr = s, float(thr)

            if local_s > final_score:
                final_score, final_model, final_thr = local_s, mdl, local_thr

        if final_model is None:
            print("FAILED — no valid model")
            continue

        y_proba = np.clip(final_model.predict_proba(X_e3_te_proc)[:, 1], 1e-9, 1 - 1e-9)
        met = e3_metrics(y_e3_te, y_proba, final_thr)

        e3_models[key]  = final_model
        e3_tr_data[key] = (X_tr_s, y_tr_s)
        e3_results.append({
            'sampling': sampling, 'model': model_name, 'threshold': final_thr,
            **{k: round(v, 4) for k, v in met.items()},
        })
        print(f"acc={met['accuracy']:.3f}  rec={met['recall']:.3f}  auc={met['auc']:.3f}")

print(f"\n{'═'*68}")
print(f"2-Stage Optimization complete.  Models stored: {len(e3_models)}")

_s1_rows = [{'sampling': s, 'model': m,
              'best_stage1_score': float(df.iloc[0].get('stage_score', np.nan)),
              'best_acc_cv':       float(df.iloc[0].get('accuracy_mean', np.nan)),
              'best_rec_cv':       float(df.iloc[0].get('recall_mean', np.nan))}
            for (s, m), df in e3_stage1_results.items()]
_s2_rows = [{'sampling': s, 'model': m,
              'best_stage2_score': float(df.iloc[0].get('stage_score', np.nan)),
              'best_acc_cv':       float(df.iloc[0].get('accuracy_mean', np.nan)),
              'best_rec_cv':       float(df.iloc[0].get('recall_mean', np.nan)),
              'best_thr_cv':       float(df.iloc[0].get('threshold_mean', np.nan))}
            for (s, m), df in e3_stage2_results.items()]

pd.DataFrame(_s1_rows).to_csv(E3_DIR / 'stage1_summary.csv', index=False)
pd.DataFrame(_s2_rows).to_csv(E3_DIR / 'stage2_summary.csv', index=False)

print("\n── Stage 2 CV Summary (best score per model × sampling) ──")
display(pd.DataFrame(_s2_rows).sort_values('best_stage2_score', ascending=False).reset_index(drop=True))
"""

CELL11_C = r"""# ── EXP3_2_C · LogReg + CatBoost · Further Improved Hyperparameter Search ───
# Improvements over EXP3.1-C:
#   • Stage-1: 40 trials (↑ from 30), 4 folds (↑ from 3), epoch 120 (↑ from 100)
#   • Stage-2: 8 perturbations (↑ from 6), 6 folds (↑ from 5), epoch 400 (↑ from 300)
#   • Top-K: 8 S1 configs → 4 final models (↑ from 7→3)
#   • Final: 800 epochs (↑ from 600), threshold grid extended to 0.25
#   • LogReg: C range tightened to 5e-4–20 (proven best ~1–10)
#   • CatBoost: lr 0.003–0.10; depth 4–7; l2_leaf_reg 1.0–8.0; random_strength 0.05–1.5

from sklearn.model_selection import ParameterSampler

e3_knn_imp, e3_cat_imp = e3_fit_imputers(X_e3_tr, e3_num_cols, e3_cat_cols)

X_e3_tr_imp  = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_tr,  e3_num_cols, e3_cat_cols)
X_e3_cal_imp = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_cal, e3_num_cols, e3_cat_cols)
X_e3_te_imp  = e3_impute(e3_knn_imp, e3_cat_imp, X_e3_te,  e3_num_cols, e3_cat_cols)

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
e3_num_cols  = [c for c in e3_num_cols if c not in _e3_drop]
_e3_all_keep = e3_num_cols + e3_cat_cols
X_e3_tr_imp  = X_e3_tr_imp[_e3_all_keep]
X_e3_cal_imp = X_e3_cal_imp[_e3_all_keep]
X_e3_te_imp  = X_e3_te_imp[_e3_all_keep]
print(f"Collinearity filter (cutoff={COLLINEARITY_CUTOFF}): "
      f"dropped {len(_e3_drop)} → kept {len(e3_num_cols)} numeric features")
if _e3_drop:    print("  Dropped:", _e3_drop)
if _e3_protected: print("  Protected:", _e3_protected)

e3_scaler, e3_ohe, e3_feat_names = e3_fit_enc_scaler(X_e3_tr_imp, e3_num_cols, e3_cat_cols)
X_e3_cal_proc = e3_encode_scale(e3_scaler, e3_ohe, X_e3_cal_imp, e3_num_cols, e3_cat_cols)
X_e3_te_proc  = e3_encode_scale(e3_scaler, e3_ohe, X_e3_te_imp,  e3_num_cols, e3_cat_cols)
print(f"Cal : {X_e3_cal_proc.shape}  |  Test: {X_e3_te_proc.shape}  |  Features: {len(e3_feat_names)}")

# ── 2-Stage Optimization Config (FURTHER IMPROVED) ───────────────────────────
E3_S1_TRIALS    = 40    # ↑ from 30
E3_S1_EPOCHS    = 120   # ↑ from 100
E3_S1_FOLDS     = 4     # ↑ from 3
E3_TOP_K_S1     = 8     # ↑ from 7

E3_S2_REFINE    = 8     # ↑ from 6
E3_S2_EPOCHS    = 400   # ↑ from 300
E3_S2_FOLDS     = 6     # ↑ from 5
E3_TOP_K_S2     = 4     # ↑ from 3

E3_FINAL_EPOCHS = 800   # ↑ from 600
THRESHOLD_GRID  = np.round(np.arange(0.25, 0.60, 0.05), 2)  # extended lower bound

# ── Improved Parameter Search Spaces ─────────────────────────────────────────
# LogReg: best C in moderate range ~1–10; narrow log-space further
# CatBoost: cw dominates; lr 0.05–0.12 best; depth 4–6 best
E3_MODEL_SPACES = {
    'logreg': {
        # Tighter: best C was ~1–10; narrow log-space from 5e-4 to 20
        'C':      loguniform(5e-4, 20.0),
        'solver': ['lbfgs', 'liblinear'],
    },
    'catboost': {
        # Even lower floor; best lr in 0.05–0.12 region
        'learning_rate':   loguniform(0.003, 0.10),
        # Tighten depth; best was 4–6; remove 8+
        'depth':           randint(4, 8),
        # l2_leaf_reg: start at 1.0, max 8.0 (tighter than 0.5–10.5)
        'l2_leaf_reg':     uniform(1.0, 7.0),
        # Tighter random_strength; best ~0.1–1.0
        'random_strength': uniform(0.05, 1.45),
    },
}

_n_s2_cands = E3_TOP_K_S1 * (E3_S2_REFINE + 1)
print(f"\nEXP3.2-C · Further Improved 2-Stage Config")
print(f"Stage 1 : {E3_S1_TRIALS} trials × {E3_S1_FOLDS} folds, top-{E3_TOP_K_S1} kept")
print(f"Stage 2 : ~{_n_s2_cands} candidates × {E3_S2_FOLDS} folds, top-{E3_TOP_K_S2} for final")
print(f"Final   : {E3_FINAL_EPOCHS} epochs, threshold swept (grid: {THRESHOLD_GRID.min():.2f}–{THRESHOLD_GRID.max():.2f})")
print(f"Total   : {len(SAMPLING_METHODS)} samplings × {len(MODEL_NAMES)} models = {len(SAMPLING_METHODS)*len(MODEL_NAMES)} combos")

# ── 2-Stage Optimization + Final Training Loop ────────────────────────────────
e3_stage1_results = {}
e3_stage2_results = {}
e3_best_configs   = {}
e3_results        = []
e3_models         = {}
e3_tr_data        = {}

for sampling in SAMPLING_METHODS:
    use_cw = sampling in CW_SAMPLINGS
    print(f"\n{'═'*68}")
    print(f"Sampling: {sampling:12s}  |  class-weight active: {use_cw}")
    print(f"{'═'*68}")

    for model_name in MODEL_NAMES:
        key = (sampling, model_name)

        trials = list(ParameterSampler(
            E3_MODEL_SPACES[model_name], n_iter=E3_S1_TRIALS, random_state=E3_SEED))
        s1_rows = []
        for i, params in enumerate(trials, start=1):
            try:
                cv_met = e3_evaluate_params_cv(
                    model_name, params, sampling,
                    X_e3_tr_imp, y_e3_tr,
                    E3_S1_EPOCHS, E3_S1_FOLDS, seed=E3_SEED)
                s1_rows.append({'trial': i, 'params': params, **cv_met})
            except Exception as exc:
                s1_rows.append({'trial': i, 'params': params,
                                'stage_score': -999.0, 'error': str(exc)})

        df_s1 = (pd.DataFrame(s1_rows)
                   .sort_values('stage_score', ascending=False)
                   .reset_index(drop=True))
        e3_stage1_results[key] = df_s1
        top_s1_params = df_s1.head(E3_TOP_K_S1)['params'].tolist()
        s1_best_score = float(df_s1.iloc[0].get('stage_score', -999))

        candidates = e3_refine_candidates(top_s1_params, n_refine=E3_S2_REFINE, seed=E3_SEED)
        s2_rows = []
        for i, params in enumerate(candidates, start=1):
            try:
                cv_met = e3_evaluate_params_cv(
                    model_name, params, sampling,
                    X_e3_tr_imp, y_e3_tr,
                    E3_S2_EPOCHS, E3_S2_FOLDS, seed=E3_SEED)
                s2_rows.append({'trial': i, 'params': params, **cv_met})
            except Exception as exc:
                s2_rows.append({'trial': i, 'params': params,
                                'stage_score': -999.0, 'error': str(exc)})

        df_s2 = (pd.DataFrame(s2_rows)
                   .sort_values('stage_score', ascending=False)
                   .reset_index(drop=True))
        e3_stage2_results[key] = df_s2
        best_params_list = df_s2.head(E3_TOP_K_S2)['params'].tolist()
        e3_best_configs[key]   = best_params_list
        s2_best_score = float(df_s2.iloc[0].get('stage_score', -999))

        print(f"  {model_name:14s}  S1={s1_best_score:.4f}  →  S2={s2_best_score:.4f}", end='  ')

        X_tr_s, y_tr_s = e3_get_sampled_train(
            X_e3_tr_imp, y_e3_tr, sampling,
            e3_scaler, e3_ohe, e3_num_cols, e3_cat_cols)
        y_tr_s = np.asarray(y_tr_s, dtype=int)

        final_model = None
        final_score = -np.inf
        final_thr   = 0.5

        for params in best_params_list:
            mdl = e3_build_model_for_search(
                model_name, params, use_cw, y_tr_s, E3_FINAL_EPOCHS, seed=E3_SEED)
            try:
                mdl.fit(X_tr_s, y_tr_s)
            except Exception as exc:
                msg = str(exc).lower()
                if 'gpu' in msg or 'cuda' in msg or 'device' in msg:
                    try:
                        if   model_name == 'catboost': mdl.set_params(task_type='CPU')
                        elif model_name == 'xgboost':  mdl.set_params(device='cpu')
                        mdl.fit(X_tr_s, y_tr_s)
                    except Exception:
                        continue
                else:
                    continue

            p_cal_v = np.clip(mdl.predict_proba(X_e3_cal_proc)[:, 1], 1e-9, 1 - 1e-9)
            local_s, local_thr = -np.inf, 0.5
            for thr in THRESHOLD_GRID:
                yp = (p_cal_v >= thr).astype(int)
                s  = 0.60 * accuracy_score(y_e3_cal, yp) + \
                     0.40 * recall_score(y_e3_cal, yp, zero_division=0)
                if s > local_s:
                    local_s, local_thr = s, float(thr)

            if local_s > final_score:
                final_score, final_model, final_thr = local_s, mdl, local_thr

        if final_model is None:
            print("FAILED — no valid model")
            continue

        y_proba = np.clip(final_model.predict_proba(X_e3_te_proc)[:, 1], 1e-9, 1 - 1e-9)
        met = e3_metrics(y_e3_te, y_proba, final_thr)

        e3_models[key]  = final_model
        e3_tr_data[key] = (X_tr_s, y_tr_s)
        e3_results.append({
            'sampling': sampling, 'model': model_name, 'threshold': final_thr,
            **{k: round(v, 4) for k, v in met.items()},
        })
        print(f"acc={met['accuracy']:.3f}  rec={met['recall']:.3f}  auc={met['auc']:.3f}")

print(f"\n{'═'*68}")
print(f"2-Stage Optimization complete.  Models stored: {len(e3_models)}")

_s1_rows = [{'sampling': s, 'model': m,
              'best_stage1_score': float(df.iloc[0].get('stage_score', np.nan)),
              'best_acc_cv':       float(df.iloc[0].get('accuracy_mean', np.nan)),
              'best_rec_cv':       float(df.iloc[0].get('recall_mean', np.nan))}
            for (s, m), df in e3_stage1_results.items()]
_s2_rows = [{'sampling': s, 'model': m,
              'best_stage2_score': float(df.iloc[0].get('stage_score', np.nan)),
              'best_acc_cv':       float(df.iloc[0].get('accuracy_mean', np.nan)),
              'best_rec_cv':       float(df.iloc[0].get('recall_mean', np.nan)),
              'best_thr_cv':       float(df.iloc[0].get('threshold_mean', np.nan))}
            for (s, m), df in e3_stage2_results.items()]

pd.DataFrame(_s1_rows).to_csv(E3_DIR / 'stage1_summary.csv', index=False)
pd.DataFrame(_s2_rows).to_csv(E3_DIR / 'stage2_summary.csv', index=False)

print("\n── Stage 2 CV Summary (best score per model × sampling) ──")
display(pd.DataFrame(_s2_rows).sort_values('best_stage2_score', ascending=False).reset_index(drop=True))
"""

# ── Configuration for each notebook ───────────────────────────────────────────
configs = [
    {
        'src':  ROOT / 'EXP3_1_A_KNN_RF.ipynb',
        'dst':  ROOT / 'EXP3_2_A_KNN_RF.ipynb',
        'title_new': '# EXP3.2-A · KNN + Random Forest — Further Improved Hyperparameters\n\nTrains **KNN** and **Random Forest** with further improved 2-stage optimization across 6 sampling methods.\nBuilds on EXP3.1-A improvements: tighter search spaces from EXP3.1 best-model analysis,\nmore trials, more CV folds, higher epoch budgets, and extended threshold grid.\nRun in parallel with EXP3.2-B and EXP3.2-C.',
        'title_old': 'EXP3.1-A',
        'dir_new': 'exp3_2_knn_rf',
        'dir_old': 'exp3_1_knn_rf',
        'setup_print_new': 'print("EXP3.2-A setup complete.")',
        'setup_print_old': 'print("EXP3.1-A setup complete.")',
        'cell11_new': CELL11_A,
    },
    {
        'src':  ROOT / 'EXP3_1_B_XGB_AdaBoost.ipynb',
        'dst':  ROOT / 'EXP3_2_B_XGB_AdaBoost.ipynb',
        'title_new': '# EXP3.2-B · XGBoost + AdaBoost — Further Improved Hyperparameters\n\nTrains **XGBoost** and **AdaBoost** with further improved 2-stage optimization across 6 sampling methods.\nBuilds on EXP3.1-B improvements: tighter search spaces from EXP3.1 best-model analysis,\nmore trials, more CV folds, higher epoch budgets, and extended threshold grid.\nRun in parallel with EXP3.2-A and EXP3.2-C.',
        'title_old': 'EXP3.1-B',
        'dir_new': 'exp3_2_xgb_ada',
        'dir_old': 'exp3_1_xgb_ada',
        'setup_print_new': 'print("EXP3.2-B setup complete.")',
        'setup_print_old': 'print("EXP3.1-B setup complete.")',
        'cell11_new': CELL11_B,
    },
    {
        'src':  ROOT / 'EXP3_1_C_LogReg_CatBoost.ipynb',
        'dst':  ROOT / 'EXP3_2_C_LogReg_CatBoost.ipynb',
        'title_new': '# EXP3.2-C · Logistic Regression + CatBoost — Further Improved Hyperparameters\n\nTrains **Logistic Regression** and **CatBoost** with further improved 2-stage optimization across 6 sampling methods.\nBuilds on EXP3.1-C improvements: tighter search spaces from EXP3.1 best-model analysis,\nmore trials, more CV folds, higher epoch budgets, and extended threshold grid.\nRun in parallel with EXP3.2-A and EXP3.2-B.',
        'title_old': 'EXP3.1-C',
        'dir_new': 'exp3_2_logreg_cat',
        'dir_old': 'exp3_1_logreg_cat',
        'setup_print_new': 'print("EXP3.2-C setup complete.")',
        'setup_print_old': 'print("EXP3.1-C setup complete.")',
        'cell11_new': CELL11_C,
    },
]


def cell_source(cell):
    """Return combined source string from a notebook cell."""
    src = cell.get('source', [])
    if isinstance(src, list):
        return ''.join(src)
    return src


def set_source(cell, text):
    """Set source of a notebook cell to list of lines."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        lines[-1] = lines[-1]  # keep as-is
    cell['source'] = lines


def find_cell11(cells):
    """Find the index of cell 11 (0-indexed = 10, i.e. the 11th cell).
    The cell immediately after the 3-way split cell that contains 'ParameterSampler'.
    """
    for i, c in enumerate(cells):
        if c.get('cell_type') == 'code':
            src = cell_source(c)
            if 'ParameterSampler' in src and ('E3_S1_TRIALS' in src or 'Improved' in src or 'stage_score' in src):
                return i
    return None


def find_setup_cell(cells):
    """Find cell that sets E3_DIR."""
    for i, c in enumerate(cells):
        if c.get('cell_type') == 'code':
            src = cell_source(c)
            if 'E3_DIR' in src and 'mkdir' in src:
                return i
    return None


def find_markdown_cell(cells):
    """Find first markdown cell (the title)."""
    for i, c in enumerate(cells):
        if c.get('cell_type') == 'markdown':
            return i
    return None


for cfg in configs:
    src_path = cfg['src']
    dst_path = cfg['dst']

    print(f"\nProcessing: {src_path.name} → {dst_path.name}")

    with open(src_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # 1. Fix markdown title
    md_idx = find_markdown_cell(cells)
    if md_idx is not None:
        old_src = cell_source(cells[md_idx])
        new_src = old_src.replace(cfg['title_old'], 'EXP3.2').replace(
            '3.1', '3.2').replace('EXP3.1', 'EXP3.2')
        # Also replace full title content
        new_src = cfg['title_new']
        set_source(cells[md_idx], new_src)
        print(f"  ✓ Updated markdown title (cell {md_idx+1})")
    else:
        print(f"  ✗ Markdown title cell not found")

    # 2. Fix E3_DIR in setup cell
    setup_idx = find_setup_cell(cells)
    if setup_idx is not None:
        old_src = cell_source(cells[setup_idx])
        new_src = old_src.replace(cfg['dir_old'], cfg['dir_new'])
        # Fix the print statement
        if 'EXP3.1' in new_src:
            new_src = new_src.replace('EXP3.1', 'EXP3.2')
        if cfg['setup_print_old'] in new_src:
            new_src = new_src.replace(cfg['setup_print_old'], cfg['setup_print_new'])
        else:
            # Just replace version substring
            new_src = new_src.replace('EXP3.1-A', 'EXP3.2-A').replace(
                'EXP3.1-B', 'EXP3.2-B').replace('EXP3.1-C', 'EXP3.2-C')
        set_source(cells[setup_idx], new_src)
        print(f"  ✓ Updated E3_DIR to '{cfg['dir_new']}' (cell {setup_idx+1})")
    else:
        print(f"  ✗ Setup cell (E3_DIR) not found")

    # 3. Replace cell 11 (hyperparameter search)
    c11_idx = find_cell11(cells)
    if c11_idx is not None:
        set_source(cells[c11_idx], cfg['cell11_new'])
        # Clear any existing outputs
        cells[c11_idx]['outputs'] = []
        cells[c11_idx]['execution_count'] = None
        print(f"  ✓ Replaced hyperparameter search cell (cell {c11_idx+1})")
    else:
        print(f"  ✗ Cell-11 (hyperparameter search) not found")

    # 4. Clear outputs from ALL cells (fresh notebook)
    for cell in cells:
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  ✓ Written: {dst_path}")

print("\nDone — 3 notebooks created.")
