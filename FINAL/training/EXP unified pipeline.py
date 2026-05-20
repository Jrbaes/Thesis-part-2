from __future__ import annotations

import argparse
import json
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, SMOTENC
from scipy.stats import chi2_contingency, loguniform, pointbiserialr, randint, uniform
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)


try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

try:
    from venn_abers import VennAbers

    HAS_VENN_ABERS = True
except Exception:
    VennAbers = None
    HAS_VENN_ABERS = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    import lime.lime_tabular

    HAS_LIME = True
except Exception:
    HAS_LIME = False

try:
    import torch

    HAS_TORCH = True
    TORCH_CUDA_AVAILABLE = bool(torch.cuda.is_available())
except Exception:
    HAS_TORCH = False
    TORCH_CUDA_AVAILABLE = False


SAMPLING_METHODS = ["base", "smote", "smotenc", "cw", "smotecw", "smotencw"]
CW_SAMPLINGS = {"cw", "smotecw", "smotencw"}
CAL_METHODS = ["base", "platt", "isotonic", "venn_abers"]
TARGET_CANDIDATES = ["hypertension", "htn", "target", "label", "outcome"]

MODEL_EXPERIMENT = {
    "knn": "A",
    "randomforest": "A",
    "xgboost": "B",
    "adaboost": "B",
    "logreg": "C",
    "catboost": "C",
    "lightgbm": "C",
    "naive_bayes": "D",
}

# Notebook-faithful per-model stage budgets.
MODEL_STAGE_DEFAULTS: dict[str, dict[str, int]] = {
    # EXP A
    "knn": {
        "s1_trials": 15,
        "s1_epochs": 80,
        "s1_folds": 2,
        "top_k_s1": 5,
        "s2_refine": 2,
        "s2_epochs": 200,
        "s2_folds": 3,
        "top_k_s2": 2,
        "final_epochs": 400,
    },
    "randomforest": {
        "s1_trials": 15,
        "s1_epochs": 80,
        "s1_folds": 2,
        "top_k_s1": 5,
        "s2_refine": 2,
        "s2_epochs": 200,
        "s2_folds": 3,
        "top_k_s2": 2,
        "final_epochs": 400,
    },
    # EXP B
    "xgboost": {
        "s1_trials": 25,
        "s1_epochs": 80,
        "s1_folds": 3,
        "top_k_s1": 5,
        "s2_refine": 4,
        "s2_epochs": 200,
        "s2_folds": 4,
        "top_k_s2": 2,
        "final_epochs": 400,
    },
    "adaboost": {
        "s1_trials": 25,
        "s1_epochs": 80,
        "s1_folds": 3,
        "top_k_s1": 5,
        "s2_refine": 4,
        "s2_epochs": 200,
        "s2_folds": 4,
        "top_k_s2": 2,
        "final_epochs": 400,
    },
    # EXP C
    "logreg": {
        "s1_trials": 15,
        "s1_epochs": 80,
        "s1_folds": 2,
        "top_k_s1": 5,
        "s2_refine": 2,
        "s2_epochs": 200,
        "s2_folds": 3,
        "top_k_s2": 2,
        "final_epochs": 400,
    },
    "catboost": {
        "s1_trials": 15,
        "s1_epochs": 80,
        "s1_folds": 2,
        "top_k_s1": 5,
        "s2_refine": 2,
        "s2_epochs": 200,
        "s2_folds": 3,
        "top_k_s2": 2,
        "final_epochs": 400,
    },
    "lightgbm": {
        "s1_trials": 15,
        "s1_epochs": 80,
        "s1_folds": 2,
        "top_k_s1": 5,
        "s2_refine": 2,
        "s2_epochs": 200,
        "s2_folds": 3,
        "top_k_s2": 2,
        "final_epochs": 400,
    },
    # EXP D
    "naive_bayes": {
        "s1_trials": 15,
        "s1_epochs": 80,
        "s1_folds": 2,
        "top_k_s1": 5,
        "s2_refine": 2,
        "s2_epochs": 200,
        "s2_folds": 3,
        "top_k_s2": 2,
        "final_epochs": 400,
    },
}

DEFAULT_DROP_COLUMNS = [
    "mos_lactation",
    "cu",
    "strrec_anthro",
    "psurec_anthro",
    "provcode_anthro",
    "mos_preg",
    "anthro_group",
]


@dataclass
class PreprocessArtifacts:
    X_train_imp: pd.DataFrame
    X_cal_imp: pd.DataFrame
    X_test_imp: pd.DataFrame
    X_train_proc: np.ndarray
    X_cal_proc: np.ndarray
    X_test_proc: np.ndarray
    y_train: np.ndarray
    y_cal: np.ndarray
    y_test: np.ndarray
    knn_imputer: KNNImputer
    cat_imputer: SimpleImputer | None
    scaler: StandardScaler
    ohe: OneHotEncoder | None
    num_cols_full: list[str]
    num_cols_reduced: list[str]
    cat_cols: list[str]
    feature_names: list[str]


def resolve_training_root(cli_root: str | None) -> Path:
    if cli_root:
        root = Path(cli_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Provided root does not exist: {root}")
        return root

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        Path.cwd(),
        Path.cwd() / "src" / "training",
        Path.cwd().parent / "training",
    ]

    for candidate in candidates:
        if (candidate / "Datasets2015").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve training root. Provide --root pointing to the folder containing Datasets2015."
    )


def _find_first_dataset_csv(folder: Path) -> Path:
    preferred = sorted(folder.glob("*data-set*.csv"))
    if preferred:
        return preferred[0]
    csvs = sorted([p for p in folder.glob("*.csv") if "dictionary" not in p.name.lower()])
    if csvs:
        return csvs[0]
    raise FileNotFoundError(f"No dataset CSV found in {folder}")


def merge_datasets(training_root: Path) -> Path:
    datasets_root = training_root / "Datasets2015"
    clinical_path = _find_first_dataset_csv(datasets_root / "Clinical")
    dietary_path = _find_first_dataset_csv(datasets_root / "Dietary")
    anthropometric_path = _find_first_dataset_csv(datasets_root / "Anthropometric")

    clinical = pd.read_csv(clinical_path, low_memory=False)
    dietary = pd.read_csv(dietary_path, low_memory=False)
    anthropometric = pd.read_csv(anthropometric_path, low_memory=False)

    requested_keys = ["hhnum", "member_code"]

    if "hhnum" not in clinical.columns:
        raise KeyError("Clinical dataset must contain 'hhnum'.")

    anthropometric_join_keys = [
        k for k in requested_keys if k in anthropometric.columns and k in clinical.columns
    ]
    if len(anthropometric_join_keys) < 2:
        raise KeyError("Anthropometric dataset must contain both 'hhnum' and 'member_code'.")

    dietary_join_keys = [k for k in requested_keys if k in dietary.columns and k in clinical.columns]
    if not dietary_join_keys:
        raise KeyError("Dietary dataset must contain at least 'hhnum' for joining.")

    dietary = dietary.drop_duplicates(subset=dietary_join_keys, keep="first")
    anthropometric = anthropometric.drop_duplicates(subset=anthropometric_join_keys, keep="first")

    dietary_overlap = [
        c for c in dietary.columns if c in clinical.columns and c not in dietary_join_keys
    ]
    if dietary_overlap:
        dietary = dietary.rename(columns={c: f"{c}_dietary" for c in dietary_overlap})

    merged = clinical.merge(dietary, on=dietary_join_keys, how="left")

    anthropometric_overlap = [
        c for c in anthropometric.columns if c in merged.columns and c not in anthropometric_join_keys
    ]
    if anthropometric_overlap:
        anthropometric = anthropometric.rename(columns={c: f"{c}_anthro" for c in anthropometric_overlap})

    merged = merged.merge(anthropometric, on=anthropometric_join_keys, how="left")

    output_path = training_root / "merged_clinical_leftjoin.csv"
    merged.to_csv(output_path, index=False)

    print("Merged dataset created")
    print(f"  Clinical source      : {clinical_path.name}")
    print(f"  Dietary source       : {dietary_path.name}")
    print(f"  Anthropometric source: {anthropometric_path.name}")
    print(f"  Output               : {output_path}")
    print(f"  Rows                 : {len(merged):,}")
    print(f"  Columns              : {merged.shape[1]:,}")

    return output_path


def infer_target_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for column in df.columns:
        col_lc = column.lower()
        if any(candidate.lower() in col_lc for candidate in candidates):
            return column
    return None


def infer_bp_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    sbp_aliases = ["ave_sbp"]
    dbp_aliases = ["ave_dbp"]
    lower_map = {c.lower(): c for c in df.columns}

    sbp_col = None
    for alias in sbp_aliases:
        if alias in lower_map:
            sbp_col = lower_map[alias]
            break
    if sbp_col is None:
        for column in df.columns:
            if any(alias in column.lower() for alias in sbp_aliases):
                sbp_col = column
                break

    dbp_col = None
    for alias in dbp_aliases:
        if alias in lower_map:
            dbp_col = lower_map[alias]
            break
    if dbp_col is None:
        for column in df.columns:
            if any(alias in column.lower() for alias in dbp_aliases):
                dbp_col = column
                break

    return sbp_col, dbp_col


def find_first_column_case_insensitive(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for column in columns:
        col_lc = column.lower()
        if any(candidate.lower() in col_lc for candidate in candidates):
            return column
    return None


def to_numeric_clean(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(~values.isin([9, 99, 888888, 999999]), np.nan)


def build_smoking_level_feature(df_in: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    used_cols: list[str] = []
    smoking_level_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["smoking_level"])
    smoke_status_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["smoke_status"])
    current_smoking_col = find_first_column_case_insensitive(
        df_in.columns.tolist(), ["current_smoking", "currentsmoking"]
    )
    ever_smoke_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["ever_smk"])

    if smoking_level_col is not None:
        smoking = to_numeric_clean(df_in[smoking_level_col]).clip(lower=0, upper=3)
        used_cols.append(smoking_level_col)
        return smoking.astype(float), sorted(set(used_cols))

    idx = df_in.index
    smoking = pd.Series(np.nan, index=idx, dtype=float)

    if smoke_status_col is not None:
        status = to_numeric_clean(df_in[smoke_status_col])
        used_cols.append(smoke_status_col)
        smoking.loc[status == 0] = 0
        smoking.loc[status == 2] = 1
        smoking.loc[status == 1] = 2
        if current_smoking_col is not None:
            current = to_numeric_clean(df_in[current_smoking_col])
            used_cols.append(current_smoking_col)
            smoking.loc[(status == 1) & (current == 3)] = 3
        return smoking.astype(float), sorted(set(used_cols))

    if current_smoking_col is not None:
        current = to_numeric_clean(df_in[current_smoking_col])
        used_cols.append(current_smoking_col)
        smoking.loc[current == 0] = 0
        smoking.loc[current.isin([1, 2])] = 2
        smoking.loc[current == 3] = 3
        if ever_smoke_col is not None:
            ever = to_numeric_clean(df_in[ever_smoke_col])
            used_cols.append(ever_smoke_col)
            smoking.loc[(current == 0) & (ever > 0)] = 1
        return smoking.astype(float), sorted(set(used_cols))

    if ever_smoke_col is not None:
        ever = to_numeric_clean(df_in[ever_smoke_col])
        used_cols.append(ever_smoke_col)
        smoking.loc[ever == 0] = 0
        smoking.loc[ever > 0] = 1
        return smoking.astype(float), sorted(set(used_cols))

    return None, []


def build_alcohol_level_feature(df_in: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    used_cols: list[str] = []
    alcohol_level_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["alcohol_level"])
    alcohol_status_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["alcohol_status"])
    alcohol_ever_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["alcohol"])
    current_alcohol_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["con_alcohol"])
    drink30_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["drnk_30days"])
    binge_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["binge_drink", "binge_drinking"])

    if alcohol_level_col is not None:
        alcohol = to_numeric_clean(df_in[alcohol_level_col]).clip(lower=0, upper=3)
        used_cols.append(alcohol_level_col)
        return alcohol.astype(float), sorted(set(used_cols))

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
        drink30 = to_numeric_clean(df_in[drink30_col])
        used_cols.append(drink30_col)
        alcohol.loc[drink30 == 1] = np.maximum(alcohol.loc[drink30 == 1], 2)
    if binge_col is not None:
        binge = to_numeric_clean(df_in[binge_col])
        used_cols.append(binge_col)
        alcohol.loc[binge == 1] = 3

    if used_cols:
        return alcohol.astype(float), sorted(set(used_cols))
    return None, []


def build_bmi_feature(df_in: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    weight_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["weight"])
    height_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["height"])
    if weight_col is None or height_col is None:
        return None, []

    weight = pd.to_numeric(df_in[weight_col], errors="coerce")
    height = pd.to_numeric(df_in[height_col], errors="coerce")
    height_m = height.copy()
    if pd.notna(height_m.median(skipna=True)) and float(height_m.median(skipna=True)) > 3.0:
        height_m = height_m / 100.0

    bmi = (weight / (height_m**2)).replace([np.inf, -np.inf], np.nan)
    return bmi.astype(float), [weight_col, height_col]


def build_whr_feature(df_in: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    waist_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["waist"])
    hip_col = find_first_column_case_insensitive(df_in.columns.tolist(), ["hip"])
    if waist_col is None or hip_col is None:
        return None, []

    waist = pd.to_numeric(df_in[waist_col], errors="coerce")
    hip = pd.to_numeric(df_in[hip_col], errors="coerce").replace(0, np.nan)
    whr = (waist / hip).replace([np.inf, -np.inf], np.nan)
    return whr.astype(float), [waist_col, hip_col]


def load_and_prepare_features(data_path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    df = pd.read_csv(data_path)
    target_col = infer_target_column(df, TARGET_CANDIDATES)
    target_defined_from_bp = False

    if target_col is None:
        sbp_col, dbp_col = infer_bp_columns(df)
        if sbp_col is not None and dbp_col is not None:
            sbp = pd.to_numeric(df[sbp_col], errors="coerce")
            dbp = pd.to_numeric(df[dbp_col], errors="coerce")
            df["Hypertension"] = (((sbp >= 140) | (dbp >= 90)).fillna(False)).astype(int)
            target_col = "Hypertension"
            target_defined_from_bp = True
        else:
            raise ValueError(
                "Could not infer target and could not derive Hypertension from SBP/DBP (140/90 OR rule)."
            )

    df = df.dropna(subset=[target_col]).copy()
    y_raw = df[target_col]
    if y_raw.nunique() != 2:
        raise ValueError(f"Target must be binary. Found {y_raw.nunique()} classes.")

    if y_raw.dtype == "O":
        y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), index=y_raw.index, name=target_col)
    else:
        y = pd.Series(y_raw.astype(int), index=y_raw.index, name=target_col)

    X = df.drop(columns=[target_col]).copy()

    smoking_feature, smoking_sources = build_smoking_level_feature(X)
    if smoking_feature is not None:
        X["fe_smoking_level"] = smoking_feature

    alcohol_feature, alcohol_sources = build_alcohol_level_feature(X)
    if alcohol_feature is not None:
        X["fe_alcohol_level"] = alcohol_feature

    bmi_feature, bmi_sources = build_bmi_feature(X)
    if bmi_feature is not None:
        X["bmi"] = bmi_feature

    whr_feature, whr_sources = build_whr_feature(X)
    if whr_feature is not None:
        X["whr"] = whr_feature

    behavior_raw_candidates = [
        "current_smoking",
        "currentsmoking",
        "ever_smk",
        "smoke_status",
        "smoking_level",
        "alcohol",
        "con_alcohol",
        "drnk_30days",
        "drnk_30d_num",
        "alcohol_status",
        "binge_drink",
        "binge_drinking",
        "alcohol_level",
    ]
    x_lc = {c.lower(): c for c in X.columns}
    behavior_drop = sorted({x_lc[c.lower()] for c in behavior_raw_candidates if c.lower() in x_lc})
    if behavior_drop:
        X = X.drop(columns=behavior_drop, errors="ignore")

    anthropometric_source_drop = sorted(
        {
            c
            for c in set((bmi_sources or []) + (whr_sources or []))
            if c in X.columns and c.lower() not in {"bmi", "whr"}
        }
    )
    if anthropometric_source_drop:
        X = X.drop(columns=anthropometric_source_drop, errors="ignore")

    non_removable_base_aliases = ["age", "sex"]
    manual_non_predictive = [
        "regcode",
        "provcode",
        "provhuc",
        "psc",
        "csc",
        "rhc",
        "psurec",
        "strrec",
        "wgts",
        "fwgt",
        "finalwgt",
        "finalwgt1",
        "finalwgt4",
        "fwgth_natl_var",
        "fwgth_prov",
        "fwgth_natl2_var",
        "fwgti_natl_var",
        "fwgti_prov",
        "fwgti_natl2_var",
        "fwgti_prov2",
        "rep_natl",
        "rep_prov",
        "ms_psucode",
        "enns_year",
        "wrkplace",
        "interview_status",
        "intdate",
        "enumcode",
        "hhnum",
        "member_code",
        "ave_sbp",
        "ave_dbp",
        "sbp",
        "dbp",
        "systolic",
        "diastolic",
        "sysbp",
        "diabp",
        "blood_pressure",
        "height",
        "weight",
        "waist",
        "hip",
    ]

    x_lc = {c.lower(): c for c in X.columns}
    manual_drop = sorted({x_lc[c.lower()] for c in manual_non_predictive if c.lower() in x_lc})

    protected_base_cols: list[str] = []
    for column in X.columns:
        col_lc = column.lower()
        if any(alias in col_lc for alias in non_removable_base_aliases):
            protected_base_cols.append(column)
    protected_base_cols = sorted(set(protected_base_cols))

    if manual_drop:
        manual_drop = [c for c in manual_drop if c not in protected_base_cols]
        X = X.drop(columns=manual_drop, errors="ignore")

    age_columns = sorted([c for c in X.columns if "age" in c.lower()])
    if len(age_columns) > 1:
        canonical_age = (
            next((c for c in age_columns if c.lower() == "age"), None)
            or next((c for c in age_columns if c.lower() == "agemos"), None)
            or age_columns[0]
        )
        age_drop_duplicates = [c for c in age_columns if c != canonical_age]
        X = X.drop(columns=age_drop_duplicates, errors="ignore")

    sex_columns = sorted([c for c in X.columns if "sex" in c.lower()])
    if len(sex_columns) > 1:
        canonical_sex = next((c for c in sex_columns if c.lower() == "sex"), None) or sex_columns[0]
        sex_drop_duplicates = [c for c in sex_columns if c != canonical_sex]
        X = X.drop(columns=sex_drop_duplicates, errors="ignore")

    for candidate in DEFAULT_DROP_COLUMNS:
        if candidate in X.columns:
            X = X.drop(columns=[candidate], errors="ignore")

    metadata = {
        "target_col": target_col,
        "target_defined_from_bp": target_defined_from_bp,
        "manual_drop_count": len(manual_drop),
        "behavior_drop_count": len(behavior_drop),
        "anthropometric_source_drop_count": len(anthropometric_source_drop),
        "protected_base_cols": protected_base_cols,
        "smoking_sources": smoking_sources,
        "alcohol_sources": alcohol_sources,
    }

    return X, y, metadata


def build_available_models(requested: list[str] | None) -> list[str]:
    base = ["knn", "randomforest", "adaboost", "logreg", "naive_bayes"]
    if HAS_XGBOOST:
        base.append("xgboost")
    if HAS_CATBOOST:
        base.append("catboost")
    if HAS_LIGHTGBM:
        base.append("lightgbm")

    if requested:
        requested_set = set(requested)
        base = [m for m in base if m in requested_set]

    if not base:
        raise RuntimeError("No trainable models available with the current environment and --models filter.")

    return base


def build_model_spaces(models: list[str]) -> dict[str, dict[str, Any]]:
    spaces: dict[str, dict[str, Any]] = {}

    if "knn" in models:
        spaces["knn"] = {
            "n_neighbors": randint(3, 51),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "chebyshev"],
        }

    if "randomforest" in models:
        spaces["randomforest"] = {
            "max_features": uniform(0.1, 0.9),
            "min_samples_split": randint(2, 30),
            "min_samples_leaf": randint(1, 15),
            "max_depth": [None, 10, 20, 30, 50],
        }

    if "xgboost" in models:
        spaces["xgboost"] = {
            "learning_rate": loguniform(0.005, 0.40),
            "max_depth": randint(3, 12),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": randint(1, 10),
            "gamma": uniform(0.0, 1.0),
            "reg_lambda": loguniform(1e-2, 100),
            "reg_alpha": loguniform(1e-3, 10),
        }

    if "adaboost" in models:
        spaces["adaboost"] = {
            "n_estimators": randint(20, 80),
            "learning_rate": loguniform(0.01, 1.0),
            "base_depth": randint(1, 5),
        }

    if "logreg" in models:
        spaces["logreg"] = {
            "C": loguniform(1e-5, 1e3),
            "solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            "penalty": ["l2", "l1", "elasticnet"],
            "max_iter": randint(500, 5000),
        }

    if "catboost" in models:
        spaces["catboost"] = {
            "learning_rate": loguniform(0.001, 0.5),
            "depth": randint(4, 12),
            "l2_leaf_reg": loguniform(1e-2, 100),
            "random_strength": uniform(0.0, 2.5),
            "bagging_temperature": uniform(0.0, 2.0),
        }

    if "lightgbm" in models:
        spaces["lightgbm"] = {
            "learning_rate": loguniform(0.005, 0.40),
            "max_depth": randint(3, 12),
            "num_leaves": randint(20, 200),
            "min_child_samples": randint(5, 50),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "reg_lambda": loguniform(1e-3, 100),
            "reg_alpha": loguniform(1e-3, 10),
        }

    if "naive_bayes" in models:
        spaces["naive_bayes"] = {
            "var_smoothing": loguniform(1e-13, 5e-7),
            "positive_prior": [None, 0.20, 0.24, 0.28, 0.32, 0.36],
            "probability_floor": [0.0, 0.0005, 0.001, 0.0025, 0.005],
            "cw_pos_scale": uniform(0.7, 1.5),
            "smote_sampling_strategy": uniform(0.60, 0.35),
            "smote_k_neighbors": randint(2, 8),
        }

    return spaces


def _ohe_factory() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_imputers(X_train: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> tuple[KNNImputer, SimpleImputer | None]:
    knn = KNNImputer(n_neighbors=5)
    if num_cols:
        knn.fit(X_train[num_cols])
    else:
        knn.fit(pd.DataFrame(index=X_train.index))

    cat_imputer = None
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputer.fit(X_train[cat_cols])

    return knn, cat_imputer


def impute_frame(
    knn_imputer: KNNImputer,
    cat_imputer: SimpleImputer | None,
    X: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if num_cols:
        num_data = pd.DataFrame(
            knn_imputer.transform(X[num_cols]),
            columns=num_cols,
            index=X.index,
        )
        frames.append(num_data)
    if cat_cols and cat_imputer is not None:
        cat_data = pd.DataFrame(
            cat_imputer.transform(X[cat_cols]),
            columns=cat_cols,
            index=X.index,
        )
        frames.append(cat_data)

    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=X.index)


def fit_encoder_scaler(
    X_train_imp: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[StandardScaler, OneHotEncoder | None, list[str]]:
    scaler = StandardScaler()
    ohe = None

    feature_names = list(num_cols)

    if num_cols:
        scaler.fit(X_train_imp[num_cols])

    if cat_cols:
        ohe = _ohe_factory()
        ohe.fit(X_train_imp[cat_cols].astype(str))
        feature_names += ohe.get_feature_names_out(cat_cols).tolist()

    return scaler, ohe, feature_names


def encode_scale(
    scaler: StandardScaler,
    ohe: OneHotEncoder | None,
    X_imp: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> np.ndarray:
    parts: list[np.ndarray] = []

    if num_cols:
        parts.append(scaler.transform(X_imp[num_cols]))

    if cat_cols and ohe is not None:
        parts.append(ohe.transform(X_imp[cat_cols].astype(str)))

    return np.hstack(parts) if parts else np.empty((len(X_imp), 0), dtype=float)


def get_sampled_train(
    X_train_imp: pd.DataFrame,
    y_train: np.ndarray,
    sampling: str,
    scaler: StandardScaler,
    ohe: OneHotEncoder | None,
    num_cols: list[str],
    cat_cols: list[str],
    seed: int,
    sampling_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y_train, dtype=int)
    class_counts = np.bincount(y_arr)

    # SMOTE/SMOTENC requires at least 2 minority samples.
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        return encode_scale(scaler, ohe, X_train_imp, num_cols, cat_cols), y_arr.copy()

    minority_count = int(class_counts.min()) if len(class_counts) > 1 else 0
    k_neighbors = max(1, min(5, minority_count - 1)) if minority_count > 1 else 1
    sampling_strategy: float | str = "auto"

    if sampling_params is not None:
        if "smote_k_neighbors" in sampling_params:
            requested_k = max(1, int(round(float(sampling_params["smote_k_neighbors"]))))
            k_neighbors = max(1, min(k_neighbors, requested_k))

        if "smote_sampling_strategy" in sampling_params:
            sampling_strategy = float(np.clip(float(sampling_params["smote_sampling_strategy"]), 0.05, 1.0))

    if sampling in {"base", "cw"}:
        return encode_scale(scaler, ohe, X_train_imp, num_cols, cat_cols), y_arr.copy()

    if sampling in {"smote", "smotecw"}:
        X_proc = encode_scale(scaler, ohe, X_train_imp, num_cols, cat_cols)
        sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
        return sampler.fit_resample(X_proc, y_arr)

    if not cat_cols:
        X_proc = encode_scale(scaler, ohe, X_train_imp, num_cols, cat_cols)
        sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
        return sampler.fit_resample(X_proc, y_arr)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat_ord = encoder.fit_transform(X_train_imp[cat_cols].astype(str))
    X_num = X_train_imp[num_cols].values if num_cols else np.empty((len(X_train_imp), 0))
    X_joined = np.hstack([X_num, X_cat_ord])

    categorical_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    sampler = SMOTENC(
        categorical_features=categorical_indices,
        random_state=seed,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_joined, y_arr)

    df_num = (
        pd.DataFrame(X_resampled[:, : len(num_cols)], columns=num_cols)
        if num_cols
        else pd.DataFrame(index=range(len(X_resampled)))
    )

    X_cat_ord_res = np.round(X_resampled[:, len(num_cols) :]).astype(float)
    for j, categories in enumerate(encoder.categories_):
        X_cat_ord_res[:, j] = np.clip(X_cat_ord_res[:, j], 0, len(categories) - 1)
    X_cat_back = encoder.inverse_transform(X_cat_ord_res)
    df_cat = pd.DataFrame(X_cat_back, columns=cat_cols)

    X_rebuilt = pd.concat([df_num, df_cat], axis=1)
    X_proc = encode_scale(scaler, ohe, X_rebuilt, num_cols, cat_cols)
    return X_proc, np.asarray(y_resampled, dtype=int)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob), 1e-9, 1 - 1e-9)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0

    for idx in range(n_bins):
        mask = bin_ids == idx
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())

    return float(ece)


def metric_pack(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "logloss": float(log_loss(y_true, y_prob)),
        "ece": float(compute_ece(y_true, y_prob)),
    }


def _fit_model_with_nb_weights(model: Any, model_name: str, use_class_weight: bool, X: np.ndarray, y: np.ndarray) -> None:
    if model_name == "naive_bayes" and use_class_weight:
        pos_scale = float(getattr(model, "_cw_pos_scale", 1.0))
        y_arr = np.asarray(y, dtype=int)
        n_pos = max(1, int(y_arr.sum()))
        n_neg = max(1, int(len(y_arr) - n_pos))
        sw = np.where(y_arr == 1, (n_neg / n_pos) * pos_scale, 1.0)
        model.fit(X, y_arr, sample_weight=sw)
    else:
        model.fit(X, y)


def _apply_nb_probability_floor(model: Any, p: np.ndarray) -> np.ndarray:
    p_floor = float(getattr(model, "_probability_floor", 0.0))
    if p_floor > 0.0:
        return np.clip(p, p_floor, 1.0 - p_floor)
    return p


def fit_calibrator(method: str, p_cal: np.ndarray, y_cal: np.ndarray) -> Any:
    if method == "base":
        return None

    p_cal = np.clip(np.asarray(p_cal), 1e-9, 1 - 1e-9)
    y_cal = np.asarray(y_cal, dtype=int)

    if method == "platt":
        calibrator = LogisticRegression(max_iter=3000, random_state=42)
        calibrator.fit(p_cal.reshape(-1, 1), y_cal)
        return calibrator

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p_cal, y_cal)
        return calibrator

    if method == "venn_abers":
        if not HAS_VENN_ABERS:
            raise RuntimeError("venn-abers is not installed")
        calibrator = VennAbers()
        calibrator.fit(np.column_stack([1.0 - p_cal, p_cal]), y_cal)
        return calibrator

    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator: Any, p_eval: np.ndarray) -> np.ndarray:
    p_eval = np.clip(np.asarray(p_eval), 1e-9, 1 - 1e-9)

    if method == "base" or calibrator is None:
        return p_eval

    if method == "platt":
        return np.clip(calibrator.predict_proba(p_eval.reshape(-1, 1))[:, 1], 1e-9, 1 - 1e-9)

    if method == "isotonic":
        return np.clip(calibrator.predict(p_eval), 1e-9, 1 - 1e-9)

    if method == "venn_abers":
        pred = calibrator.predict_proba(np.column_stack([1.0 - p_eval, p_eval]))
        if isinstance(pred, tuple) and len(pred) == 2:
            calibrated_pair, interval_pair = pred
            calibrated_pair = np.asarray(calibrated_pair)
            if calibrated_pair.ndim == 2 and calibrated_pair.shape[1] >= 2:
                return np.clip(calibrated_pair[:, 1], 1e-9, 1 - 1e-9)

            interval_pair = np.asarray(interval_pair)
            if interval_pair.ndim == 2 and interval_pair.shape[1] >= 2:
                return np.clip(interval_pair[:, 1], 1e-9, 1 - 1e-9)
            return np.clip(interval_pair.reshape(-1), 1e-9, 1 - 1e-9)

        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] >= 2:
            return np.clip(pred[:, 1], 1e-9, 1 - 1e-9)
        return np.clip(pred.reshape(-1), 1e-9, 1 - 1e-9)

    raise ValueError(f"Unknown calibration method: {method}")


def refine_candidates(base_params_list: list[dict[str, Any]], n_refine: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed)
    candidates: list[dict[str, Any]] = []

    for params in base_params_list:
        candidates.append(deepcopy(params))
        for _ in range(n_refine):
            refined: dict[str, Any] = {}
            for key, value in params.items():
                if isinstance(value, (int, np.integer)):
                    refined[key] = max(1, int(round(value * rng.uniform(0.7, 1.3))))
                elif isinstance(value, (float, np.floating)):
                    refined[key] = max(1e-7, float(value * rng.uniform(0.7, 1.3)))
                else:
                    refined[key] = value
            candidates.append(refined)

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for params in candidates:
        key = json.dumps(params, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(params)

    return unique


def build_model_for_search(
    model_name: str,
    params: dict[str, Any],
    use_class_weight: bool,
    y_sample: np.ndarray,
    epoch_budget: int,
    seed: int,
    use_gpu_when_available: bool,
) -> Any:
    y_sample = np.asarray(y_sample, dtype=int)
    n_pos = int(y_sample.sum())
    n_neg = int(len(y_sample) - n_pos)
    pos_weight = float(n_neg) / max(n_pos, 1)
    p = deepcopy(params)

    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=max(1, int(p.get("n_neighbors", 7))),
            weights=p.get("weights", "uniform"),
            metric=p.get("metric", "euclidean"),
            n_jobs=1,
        )

    if model_name == "randomforest":
        max_features = p.get("max_features", "sqrt")
        if isinstance(max_features, float):
            max_features = float(np.clip(max_features, 0.01, 1.0))

        return RandomForestClassifier(
            n_estimators=int(max(20, epoch_budget)),
            max_depth=p.get("max_depth", None),
            min_samples_split=max(2, int(round(p.get("min_samples_split", 2)))),
            min_samples_leaf=max(1, int(round(p.get("min_samples_leaf", 1)))),
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample" if use_class_weight else None,
        )

    if model_name == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not available")

        use_gpu = bool(use_gpu_when_available and TORCH_CUDA_AVAILABLE)
        return XGBClassifier(
            n_estimators=int(max(30, epoch_budget)),
            learning_rate=max(1e-4, float(p.get("learning_rate", 0.1))),
            max_depth=max(1, int(round(p.get("max_depth", 5)))),
            subsample=float(np.clip(p.get("subsample", 0.8), 0.1, 1.0)),
            colsample_bytree=float(np.clip(p.get("colsample_bytree", 0.8), 0.1, 1.0)),
            min_child_weight=max(1, int(round(p.get("min_child_weight", 1)))),
            gamma=max(0.0, float(p.get("gamma", 0.0))),
            reg_lambda=max(1e-4, float(p.get("reg_lambda", 1.0))),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            verbosity=0,
            scale_pos_weight=pos_weight if use_class_weight else 1.0,
        )

    if model_name == "adaboost":
        base_depth = max(1, int(round(p.get("base_depth", 1))))
        base_estimator = DecisionTreeClassifier(
            max_depth=base_depth,
            random_state=seed,
            class_weight="balanced" if use_class_weight else None,
        )
        kwargs = {
            "n_estimators": max(10, int(round(p.get("n_estimators", epoch_budget)))),
            "learning_rate": max(1e-4, float(p.get("learning_rate", 0.1))),
            "random_state": seed,
        }
        try:
            return AdaBoostClassifier(estimator=base_estimator, **kwargs)
        except TypeError:
            return AdaBoostClassifier(base_estimator=base_estimator, **kwargs)

    if model_name == "logreg":
        return LogisticRegression(
            C=float(p.get("C", 1.0)),
            solver=p.get("solver", "lbfgs"),
            penalty=p.get("penalty", "l2"),
            max_iter=1000,
            random_state=seed,
            class_weight="balanced" if use_class_weight else None,
        )

    if model_name == "catboost":
        if not HAS_CATBOOST:
            raise RuntimeError("catboost is not available")

        use_gpu = bool(use_gpu_when_available and TORCH_CUDA_AVAILABLE)
        kwargs = {
            "iterations": int(max(30, epoch_budget)),
            "learning_rate": max(1e-4, float(p.get("learning_rate", 0.1))),
            "depth": max(1, int(round(p.get("depth", 6)))),
            "l2_leaf_reg": max(1e-4, float(p.get("l2_leaf_reg", 3.0))),
            "random_strength": max(1e-4, float(p.get("random_strength", 1.0))),
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": seed,
            "verbose": 0,
        }
        if use_class_weight:
            kwargs["auto_class_weights"] = "Balanced"
        if use_gpu:
            kwargs["task_type"] = "GPU"
            kwargs["devices"] = "0"
        return CatBoostClassifier(**kwargs)

    if model_name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm is not available")

        use_gpu = bool(use_gpu_when_available and TORCH_CUDA_AVAILABLE)
        return LGBMClassifier(
            n_estimators=int(max(30, epoch_budget)),
            learning_rate=max(1e-4, float(p.get("learning_rate", 0.1))),
            max_depth=int(round(p.get("max_depth", -1))),
            num_leaves=max(2, int(round(p.get("num_leaves", 31)))),
            min_child_samples=max(1, int(round(p.get("min_child_samples", 20)))),
            subsample=float(np.clip(p.get("subsample", 0.8), 0.1, 1.0)),
            colsample_bytree=float(np.clip(p.get("colsample_bytree", 0.8), 0.1, 1.0)),
            reg_lambda=max(0.0, float(p.get("reg_lambda", 1.0))),
            reg_alpha=max(0.0, float(p.get("reg_alpha", 0.0))),
            scale_pos_weight=pos_weight if use_class_weight else 1.0,
            device_type="gpu" if use_gpu else "cpu",
            random_state=seed,
            verbose=-1,
        )

    if model_name == "naive_bayes":
        return GaussianNB(var_smoothing=max(1e-15, float(p.get("var_smoothing", 1e-9))))

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_params_cv(
    model_name: str,
    params: dict[str, Any],
    sampling: str,
    X_train_imp: pd.DataFrame,
    y_train: np.ndarray,
    epoch_budget: int,
    n_splits: int,
    seed: int,
    num_cols: list[str],
    cat_cols: list[str],
    threshold_grid: np.ndarray,
    use_gpu_when_available: bool,
) -> dict[str, float]:
    use_class_weight = sampling in CW_SAMPLINGS
    y_arr = np.asarray(y_train, dtype=int)
    class_counts = np.bincount(y_arr)

    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        raise ValueError("Need at least 2 samples per class for stratified CV.")

    effective_splits = min(int(n_splits), int(class_counts.min()))
    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

    fold_rows: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_imp, y_arr), start=1):
        X_fold_train = X_train_imp.iloc[train_idx]
        X_fold_val = X_train_imp.iloc[val_idx]
        y_fold_train = y_arr[train_idx]
        y_fold_val = y_arr[val_idx]

        fold_scaler, fold_ohe, _ = fit_encoder_scaler(X_fold_train, num_cols, cat_cols)
        X_fold_train_sampled, y_fold_train_sampled = get_sampled_train(
            X_fold_train,
            y_fold_train,
            sampling,
            fold_scaler,
            fold_ohe,
            num_cols,
            cat_cols,
            seed,
            sampling_params=params,
        )
        X_fold_val_proc = encode_scale(fold_scaler, fold_ohe, X_fold_val, num_cols, cat_cols)

        model = build_model_for_search(
            model_name,
            params,
            use_class_weight,
            y_fold_train_sampled,
            epoch_budget,
            seed,
            use_gpu_when_available,
        )

        try:
            model.fit(X_fold_train_sampled, y_fold_train_sampled)
        except Exception as exc:
            msg = str(exc).lower()
            if ("gpu" in msg or "cuda" in msg or "device" in msg) and model_name in {
                "xgboost",
                "catboost",
                "lightgbm",
            }:
                if model_name == "xgboost":
                    model.set_params(device="cpu")
                elif model_name == "catboost":
                    model.set_params(task_type="CPU")
                elif model_name == "lightgbm":
                    model.set_params(device_type="cpu")
                model.fit(X_fold_train_sampled, y_fold_train_sampled)
            else:
                raise

        prob_val = np.clip(model.predict_proba(X_fold_val_proc)[:, 1], 1e-9, 1 - 1e-9)
        prob_val = _apply_nb_probability_floor(model, prob_val)

        best_obj = -np.inf
        best_metrics: dict[str, float] | None = None
        best_thr = 0.5

        for thr in threshold_grid:
            preds = (prob_val >= thr).astype(int)
            acc = accuracy_score(y_fold_val, preds)
            rec = recall_score(y_fold_val, preds, zero_division=0)
            obj = 0.60 * acc + 0.40 * rec
            if obj > best_obj:
                best_obj = obj
                best_metrics = {
                    "accuracy": float(acc),
                    "recall": float(rec),
                    "precision": float(precision_score(y_fold_val, preds, zero_division=0)),
                    "f1": float(f1_score(y_fold_val, preds, zero_division=0)),
                    "auc": float(roc_auc_score(y_fold_val, prob_val)) if len(np.unique(y_fold_val)) > 1 else 0.5,
                    "logloss": float(log_loss(y_fold_val, prob_val)),
                }
                best_thr = float(thr)

        assert best_metrics is not None
        best_metrics["fold"] = float(fold_idx)
        best_metrics["best_threshold"] = best_thr
        fold_rows.append(best_metrics)

    summary_df = pd.DataFrame(fold_rows)

    summary = {
        "accuracy_mean": float(summary_df["accuracy"].mean()),
        "accuracy_std": float(summary_df["accuracy"].std(ddof=0)),
        "recall_mean": float(summary_df["recall"].mean()),
        "recall_std": float(summary_df["recall"].std(ddof=0)),
        "precision_mean": float(summary_df["precision"].mean()),
        "f1_mean": float(summary_df["f1"].mean()),
        "f1_std": float(summary_df["f1"].std(ddof=0)),
        "auc_mean": float(summary_df["auc"].mean()),
        "logloss_mean": float(summary_df["logloss"].mean()),
        "logloss_std": float(summary_df["logloss"].std(ddof=0)),
        "threshold_mean": float(summary_df["best_threshold"].mean()),
    }

    summary["stage_score"] = (
        0.60 * summary["accuracy_mean"]
        + 0.40 * summary["recall_mean"]
        + 0.05 * summary["f1_mean"]
        - 0.08 * summary["logloss_mean"]
        - 0.03 * summary["accuracy_std"]
        - 0.03 * summary["recall_std"]
    )

    return summary


def run_eda(
    X: pd.DataFrame,
    y: pd.Series,
    target_col: str,
    X_train_raw: pd.DataFrame,
    X_train_imp_before_filter: pd.DataFrame,
    num_cols_after_filter: list[str],
    output_dir: Path,
    collinearity_cutoff: float,
) -> None:
    eda_dir = output_dir / "eda_plots"
    eda_dir.mkdir(parents=True, exist_ok=True)

    eda_df = X.copy()
    eda_df["__target__"] = y.values
    num_cols_raw = eda_df.drop(columns=["__target__"]).select_dtypes(include=np.number).columns.tolist()
    cat_cols_raw = [c for c in eda_df.columns if c not in num_cols_raw and c != "__target__"]

    overview_rows = []
    for col in eda_df.columns:
        if col == "__target__":
            continue
        s = eda_df[col]
        overview_rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "missing_n": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100.0),
                "n_unique": int(s.nunique(dropna=False)),
            }
        )

    overview_df = pd.DataFrame(overview_rows).sort_values("missing_pct", ascending=False)
    overview_df.to_csv(output_dir / "eda_feature_overview.csv", index=False)

    class_counts = y.value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["No Hypertension (0)", "Hypertension (1)"]
    bars = axes[0].bar(labels, class_counts.values, color=["#4878CF", "#D65F5F"], edgecolor="white")
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{count:,}\n({count / len(y) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_title("Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")

    axes[1].pie(
        class_counts.values,
        labels=labels,
        colors=["#4878CF", "#D65F5F"],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Class Proportion", fontweight="bold")

    fig.suptitle(f"Target Variable: {target_col}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(eda_dir / "fig01_target_distribution.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    missing_df = overview_df[overview_df["missing_n"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, min(12, len(missing_df) * 0.35 + 2))))
    if len(missing_df) > 0:
        axes[0].barh(missing_df["feature"], missing_df["missing_pct"], color="#E07B54", edgecolor="white")
        axes[0].axvline(5, color="#555", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[0].axvline(20, color="#C22", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[0].set_title("Missing Data per Feature", fontweight="bold")
        axes[0].set_xlabel("Missing (%)")
        axes[0].invert_yaxis()

        miss_matrix = eda_df.drop(columns=["__target__"]).isnull()
        show_cols = missing_df["feature"].tolist()[:30]
        sampled = miss_matrix[show_cols].astype(int)
        sample_n = min(500, len(sampled))
        sampled = sampled.sample(sample_n, random_state=42)
        sns.heatmap(
            sampled.T,
            cmap=["#EEF2FF", "#D65F5F"],
            cbar=False,
            xticklabels=False,
            yticklabels=True,
            linewidths=0,
            ax=axes[1],
        )
        axes[1].set_title(f"Missingness Pattern (sample={sample_n})", fontweight="bold")
        axes[1].set_xlabel("Observations")
    else:
        axes[0].text(0.5, 0.5, "No missing values", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()
        axes[1].text(0.5, 0.5, "No missingness map", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(eda_dir / "fig02_missing_data.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    all_num_before = X_train_raw.select_dtypes(include=np.number).columns.tolist()
    knn_vis = KNNImputer(n_neighbors=5)
    X_before_imp = pd.DataFrame(
        knn_vis.fit_transform(X_train_raw[all_num_before]),
        columns=all_num_before,
        index=X_train_raw.index,
    )
    corr_before = X_before_imp.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(corr_before) * 0.35 + 1), max(8, len(corr_before) * 0.35)))
    mask = np.triu(np.ones_like(corr_before, dtype=bool), k=0)
    sns.heatmap(
        corr_before,
        mask=mask,
        annot=(len(corr_before) <= 20),
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor="#eee",
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        ax=ax,
        square=True,
    )
    ax.set_title("Correlation Matrix Before Culling", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(eda_dir / "fig03a_corr_before_culling.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    corr_after = X_train_imp_before_filter[num_cols_after_filter].corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr_after) * 0.4 + 1), max(6, len(corr_after) * 0.4)))
    mask_after = np.triu(np.ones_like(corr_after, dtype=bool), k=0)
    sns.heatmap(
        corr_after,
        mask=mask_after,
        annot=(len(corr_after) <= 25),
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor="#eee",
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        ax=ax,
        square=True,
    )
    ax.set_title(
        f"Correlation Matrix After Culling (cutoff={collinearity_cutoff})",
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(eda_dir / "fig03b_corr_after_culling.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    assoc_rows: list[dict[str, Any]] = []
    y_vals = eda_df["__target__"].values

    for col in num_cols_raw:
        vals = pd.to_numeric(eda_df[col], errors="coerce").values
        valid = ~(np.isnan(vals) | np.isnan(y_vals.astype(float)))
        if valid.sum() < 30:
            continue
        corr, p_val = pointbiserialr(vals[valid], y_vals[valid])
        assoc_rows.append(
            {
                "feature": col,
                "type": "numeric",
                "metric": "point_biserial_r",
                "statistic": float(corr),
                "abs_stat": float(abs(corr)),
                "p_value": float(p_val),
            }
        )

    def cramers_v(x: pd.Series, y_local: pd.Series) -> float:
        table = pd.crosstab(x, y_local)
        if table.empty:
            return 0.0
        chi2, _, _, _ = chi2_contingency(table)
        n = table.values.sum()
        k = min(table.shape) - 1
        if n <= 0 or k <= 0:
            return 0.0
        return float(np.sqrt(chi2 / n) / np.sqrt(k))

    for col in cat_cols_raw:
        vals = eda_df[col].dropna()
        if len(vals) < 30:
            continue
        aligned_target = eda_df.loc[vals.index, "__target__"]
        v = cramers_v(vals, aligned_target)
        assoc_rows.append(
            {
                "feature": col,
                "type": "categorical",
                "metric": "cramers_v",
                "statistic": float(v),
                "abs_stat": float(abs(v)),
                "p_value": np.nan,
            }
        )

    assoc_df = pd.DataFrame(assoc_rows).sort_values("abs_stat", ascending=False).reset_index(drop=True)
    assoc_df.to_csv(output_dir / "eda_feature_association.csv", index=False)

    top_n = min(40, len(assoc_df))
    if top_n > 0:
        plot_df = assoc_df.head(top_n).iloc[::-1]
        colors = ["#D65F5F" if t == "numeric" else "#8DA0CB" for t in plot_df["type"]]

        fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1.5))
        ax.barh(plot_df["feature"], plot_df["abs_stat"], color=colors, edgecolor="white")
        ax.axvline(0.1, color="#999", linestyle="--", linewidth=0.8)
        ax.axvline(0.3, color="#555", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Association Strength")
        ax.set_title("Feature-Target Association Ranking", fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        plt.savefig(eda_dir / "fig05_feature_association_ranking.png", bbox_inches="tight", dpi=200)
        plt.close(fig)


def run_shap(
    best_model: Any,
    best_model_name: str,
    best_sampling: str,
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    shap_test_n: int,
    shap_bg_n: int,
) -> int:
    if not HAS_SHAP:
        print("SHAP is unavailable. Skipping SHAP stage.")
        return 0

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_n = min(shap_test_n, len(X_test_proc))
    bg_n = min(shap_bg_n, len(X_train_proc))

    if test_n < 1 or bg_n < 1:
        print("SHAP skipped: empty background or test subset after parameter limits.")
        return 0

    X_test_shap = X_test_proc[:test_n]
    X_bg = X_train_proc[:bg_n]

    tree_types = {"catboost", "xgboost", "randomforest", "adaboost", "lightgbm"}

    if best_model_name in tree_types:
        explainer = shap.TreeExplainer(best_model)
    elif best_model_name == "logreg":
        explainer = shap.LinearExplainer(best_model, X_bg)
    else:
        kernel_bg = X_bg[: min(60, len(X_bg))]
        explainer = shap.KernelExplainer(lambda x: best_model.predict_proba(x)[:, 1], kernel_bg)

    shap_values_raw = explainer.shap_values(X_test_shap)

    if isinstance(shap_values_raw, list):
        shap_values = np.array(shap_values_raw[1])
        if hasattr(explainer.expected_value, "__len__"):
            expected_value = float(explainer.expected_value[1])
        else:
            expected_value = float(explainer.expected_value)
    else:
        shap_values = np.array(shap_values_raw)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        if hasattr(explainer.expected_value, "__len__"):
            expected_value = float(explainer.expected_value[0])
        else:
            expected_value = float(explainer.expected_value)

    if shap_values.ndim != 2:
        shap_values = np.atleast_2d(shap_values)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test_shap,
        feature_names=np.array(feature_names),
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(
        f"SHAP Global - Feature Importance\n{best_model_name} | {best_sampling}",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test_shap,
        feature_names=np.array(feature_names),
        show=False,
        max_display=20,
    )
    plt.title(
        f"SHAP Global - Beeswarm\n{best_model_name} | {best_sampling}",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_global_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    positive_indices = np.where(y_test[:test_n] == 1)[0]
    local_idx = int(positive_indices[0]) if len(positive_indices) else 0

    local_explanation = shap.Explanation(
        values=shap_values[local_idx],
        base_values=expected_value,
        data=X_test_shap[local_idx],
        feature_names=list(feature_names),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(local_explanation, show=False, max_display=15)
    plt.title(
        f"SHAP Local - Waterfall (sample #{local_idx})\n{best_model_name} | {best_sampling}",
        fontweight="bold",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_local_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP plots saved to: {plots_dir}")
    return local_idx


def run_lime(
    best_model: Any,
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    feature_names: list[str],
    local_idx: int,
    output_dir: Path,
    lime_features: int,
    seed: int,
) -> None:
    if not HAS_LIME:
        print("LIME is unavailable. Skipping LIME stage.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if len(X_test_proc) < 1:
        print("LIME skipped: empty test set.")
        return

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_proc,
        feature_names=feature_names,
        class_names=["No HTN", "HTN"],
        mode="classification",
        random_state=seed,
        discretize_continuous=True,
    )

    lime_exp = explainer.explain_instance(
        data_row=X_test_proc[local_idx],
        predict_fn=best_model.predict_proba,
        num_features=min(lime_features, len(feature_names)),
        labels=(1,),
    )

    fig = lime_exp.as_pyplot_figure(label=1)
    fig.set_size_inches(10, 6)
    plt.title(
        f"LIME Local Explanation (sample #{local_idx})",
        fontweight="bold",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "lime_local.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lime_pairs = lime_exp.as_list(label=1)
    lime_table = pd.DataFrame(lime_pairs, columns=["Feature Condition", "LIME Weight"])
    lime_table["abs_weight"] = lime_table["LIME Weight"].abs()
    lime_table = lime_table.sort_values("abs_weight", ascending=False).reset_index(drop=True)
    lime_table.to_csv(output_dir / "lime_local_weights.csv", index=False)

    print(f"LIME outputs saved to: {plots_dir}")


def _effective_stage_value(model_name: str, key: str, cli_override: int | None) -> int:
    profile = MODEL_STAGE_DEFAULTS.get(model_name)
    if profile is None:
        raise KeyError(f"Missing stage defaults for model: {model_name}")
    return int(profile[key] if cli_override is None else cli_override)


def run_pipeline(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    training_root = resolve_training_root(args.root)
    output_dir = training_root / args.out_dir
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 90)
    print("EXP3 Unified Pipeline")
    print("-" * 90)
    print(f"Training root: {training_root}")
    print(f"Output dir   : {output_dir}")
    print(f"CUDA available (torch): {TORCH_CUDA_AVAILABLE}")

    if not args.skip_merge:
        merge_datasets(training_root)

    data_path = training_root / "merged_clinical_leftjoin.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Required dataset not found: {data_path}")

    X, y, base_meta = load_and_prepare_features(data_path)

    X_train_raw, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=args.seed,
        stratify=y,
    )
    X_cal_raw, X_test_raw, y_cal, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        random_state=args.seed,
        stratify=y_tmp,
    )

    y_train_arr = np.asarray(y_train, dtype=int)
    y_cal_arr = np.asarray(y_cal, dtype=int)
    y_test_arr = np.asarray(y_test, dtype=int)

    num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train_raw.columns if c not in num_cols]

    print("Split summary")
    print(
        pd.DataFrame(
            {
                "Split": ["Train", "Cal", "Test"],
                "N": [len(y_train_arr), len(y_cal_arr), len(y_test_arr)],
                "Pos %": [
                    f"{y_train_arr.mean() * 100:.1f}",
                    f"{y_cal_arr.mean() * 100:.1f}",
                    f"{y_test_arr.mean() * 100:.1f}",
                ],
            }
        ).to_string(index=False)
    )

    knn_imputer, cat_imputer = fit_imputers(X_train_raw, num_cols, cat_cols)

    X_train_imp = impute_frame(knn_imputer, cat_imputer, X_train_raw, num_cols, cat_cols)
    X_cal_imp = impute_frame(knn_imputer, cat_imputer, X_cal_raw, num_cols, cat_cols)
    X_test_imp = impute_frame(knn_imputer, cat_imputer, X_test_raw, num_cols, cat_cols)

    protected_numeric = sorted(
        {
            c
            for c in num_cols
            if any(alias in c.lower() for alias in ["age", "sex", "bmi", "whr"])
        }
    )

    corr_matrix = X_train_imp[num_cols].corr().abs() if num_cols else pd.DataFrame()
    if not corr_matrix.empty:
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_columns = [
            c
            for c in upper.columns
            if c not in protected_numeric and (upper[c] > args.collinearity_cutoff).any()
        ]
    else:
        drop_columns = []

    num_cols_reduced = [c for c in num_cols if c not in drop_columns]
    keep_columns = num_cols_reduced + cat_cols

    X_train_imp = X_train_imp[keep_columns]
    X_cal_imp = X_cal_imp[keep_columns]
    X_test_imp = X_test_imp[keep_columns]

    scaler, ohe, feature_names = fit_encoder_scaler(X_train_imp, num_cols_reduced, cat_cols)
    X_cal_proc = encode_scale(scaler, ohe, X_cal_imp, num_cols_reduced, cat_cols)
    X_test_proc = encode_scale(scaler, ohe, X_test_imp, num_cols_reduced, cat_cols)

    print(f"Numeric features before cull: {len(num_cols)}")
    print(f"Numeric features after cull : {len(num_cols_reduced)}")
    print(f"Dropped by collinearity     : {len(drop_columns)}")
    if drop_columns:
        print(f"Dropped columns             : {drop_columns}")
    print(f"Model feature count         : {len(feature_names)}")

    preprocess_artifacts = PreprocessArtifacts(
        X_train_imp=X_train_imp,
        X_cal_imp=X_cal_imp,
        X_test_imp=X_test_imp,
        X_train_proc=encode_scale(scaler, ohe, X_train_imp, num_cols_reduced, cat_cols),
        X_cal_proc=X_cal_proc,
        X_test_proc=X_test_proc,
        y_train=y_train_arr,
        y_cal=y_cal_arr,
        y_test=y_test_arr,
        knn_imputer=knn_imputer,
        cat_imputer=cat_imputer,
        scaler=scaler,
        ohe=ohe,
        num_cols_full=num_cols,
        num_cols_reduced=num_cols_reduced,
        cat_cols=cat_cols,
        feature_names=feature_names,
    )

    if not args.skip_eda:
        run_eda(
            X=X,
            y=y,
            target_col=str(base_meta["target_col"]),
            X_train_raw=X_train_raw,
            X_train_imp_before_filter=impute_frame(knn_imputer, cat_imputer, X_train_raw, num_cols, cat_cols),
            num_cols_after_filter=num_cols_reduced,
            output_dir=output_dir,
            collinearity_cutoff=args.collinearity_cutoff,
        )

    available_models = build_available_models(args.models)
    model_spaces = build_model_spaces(available_models)
    threshold_grid = np.round(
        np.arange(args.threshold_min, args.threshold_max, args.threshold_step),
        2,
    )
    if len(threshold_grid) < 1:
        raise ValueError("Threshold grid is empty. Check --threshold-min/--threshold-max/--threshold-step.")

    print("Models to train:", available_models)
    print("Sampling methods:", args.sampling_methods)

    stage1_results: dict[tuple[str, str], pd.DataFrame] = {}
    stage2_results: dict[tuple[str, str], pd.DataFrame] = {}
    best_config_per_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    trained_models: dict[tuple[str, str], Any] = {}
    train_data_by_key: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    model_thresholds: dict[tuple[str, str], float] = {}
    pre_rows: list[dict[str, Any]] = []

    for sampling in args.sampling_methods:
        use_cw = sampling in CW_SAMPLINGS
        print("=" * 80)
        print(f"Sampling: {sampling} | class-weight active: {use_cw}")
        print("=" * 80)

        for model_name in available_models:
            key = (sampling, model_name)
            print(f"  Training {model_name} ...")

            s1_trials = _effective_stage_value(model_name, "s1_trials", args.s1_trials)
            s1_epochs = _effective_stage_value(model_name, "s1_epochs", args.s1_epochs)
            s1_folds = _effective_stage_value(model_name, "s1_folds", args.s1_folds)
            top_k_s1_eff = _effective_stage_value(model_name, "top_k_s1", args.top_k_s1)
            s2_refine = _effective_stage_value(model_name, "s2_refine", args.s2_refine)
            s2_epochs = _effective_stage_value(model_name, "s2_epochs", args.s2_epochs)
            s2_folds = _effective_stage_value(model_name, "s2_folds", args.s2_folds)
            top_k_s2_eff = _effective_stage_value(model_name, "top_k_s2", args.top_k_s2)
            final_epochs = _effective_stage_value(model_name, "final_epochs", args.final_epochs)

            stage1_trials = list(
                ParameterSampler(
                    model_spaces[model_name],
                    n_iter=s1_trials,
                    random_state=args.seed,
                )
            )

            stage1_rows: list[dict[str, Any]] = []
            for trial_idx, params in enumerate(stage1_trials, start=1):
                try:
                    cv_summary = evaluate_params_cv(
                        model_name=model_name,
                        params=params,
                        sampling=sampling,
                        X_train_imp=preprocess_artifacts.X_train_imp,
                        y_train=preprocess_artifacts.y_train,
                        epoch_budget=s1_epochs,
                        n_splits=s1_folds,
                        seed=args.seed,
                        num_cols=preprocess_artifacts.num_cols_reduced,
                        cat_cols=preprocess_artifacts.cat_cols,
                        threshold_grid=threshold_grid,
                        use_gpu_when_available=args.use_gpu_when_available,
                    )
                    stage1_rows.append({"trial": trial_idx, "params": params, **cv_summary})
                except Exception as exc:
                    stage1_rows.append(
                        {
                            "trial": trial_idx,
                            "params": params,
                            "stage_score": -999.0,
                            "error": str(exc),
                        }
                    )

            stage1_df = pd.DataFrame(stage1_rows).sort_values("stage_score", ascending=False).reset_index(drop=True)
            stage1_results[key] = stage1_df

            if stage1_df.empty:
                print("    Stage 1 failed: no candidates")
                continue

            stage1_valid = stage1_df if "error" not in stage1_df.columns else stage1_df[stage1_df["error"].isna()]
            if stage1_valid.empty:
                print("    Stage 1 failed: all trials errored")
                continue

            top_k_s1_eff = max(1, min(top_k_s1_eff, len(stage1_valid)))
            top_stage1_params = stage1_valid.head(top_k_s1_eff)["params"].tolist()
            stage1_best_score = float(stage1_valid.iloc[0].get("stage_score", -999.0))

            stage2_candidates = refine_candidates(
                top_stage1_params,
                n_refine=s2_refine,
                seed=args.seed,
            )

            stage2_rows: list[dict[str, Any]] = []
            for trial_idx, params in enumerate(stage2_candidates, start=1):
                try:
                    cv_summary = evaluate_params_cv(
                        model_name=model_name,
                        params=params,
                        sampling=sampling,
                        X_train_imp=preprocess_artifacts.X_train_imp,
                        y_train=preprocess_artifacts.y_train,
                        epoch_budget=s2_epochs,
                        n_splits=s2_folds,
                        seed=args.seed,
                        num_cols=preprocess_artifacts.num_cols_reduced,
                        cat_cols=preprocess_artifacts.cat_cols,
                        threshold_grid=threshold_grid,
                        use_gpu_when_available=args.use_gpu_when_available,
                    )
                    stage2_rows.append({"trial": trial_idx, "params": params, **cv_summary})
                except Exception as exc:
                    stage2_rows.append(
                        {
                            "trial": trial_idx,
                            "params": params,
                            "stage_score": -999.0,
                            "error": str(exc),
                        }
                    )

            stage2_df = pd.DataFrame(stage2_rows).sort_values("stage_score", ascending=False).reset_index(drop=True)
            stage2_results[key] = stage2_df

            if stage2_df.empty:
                print("    Stage 2 failed: no candidates")
                continue

            stage2_valid = stage2_df if "error" not in stage2_df.columns else stage2_df[stage2_df["error"].isna()]
            if stage2_valid.empty:
                print("    Stage 2 failed: all trials errored")
                continue

            top_k_s2_eff = max(1, min(top_k_s2_eff, len(stage2_valid)))
            best_params_list = stage2_valid.head(top_k_s2_eff)["params"].tolist()
            best_config_per_key[key] = best_params_list

            stage2_best_score = float(stage2_valid.iloc[0].get("stage_score", -999.0))
            print(f"    Stage1 best={stage1_best_score:.4f} | Stage2 best={stage2_best_score:.4f}")

            best_model = None
            best_score = -np.inf
            best_threshold = 0.5
            best_tr_data: tuple[np.ndarray, np.ndarray] | None = None

            for params in best_params_list:
                X_train_sampled, y_train_sampled = get_sampled_train(
                    preprocess_artifacts.X_train_imp,
                    preprocess_artifacts.y_train,
                    sampling,
                    preprocess_artifacts.scaler,
                    preprocess_artifacts.ohe,
                    preprocess_artifacts.num_cols_reduced,
                    preprocess_artifacts.cat_cols,
                    args.seed,
                    sampling_params=params,
                )
                y_train_sampled = np.asarray(y_train_sampled, dtype=int)

                model = build_model_for_search(
                    model_name=model_name,
                    params=params,
                    use_class_weight=use_cw,
                    y_sample=y_train_sampled,
                    epoch_budget=final_epochs,
                    seed=args.seed,
                    use_gpu_when_available=args.use_gpu_when_available,
                )

                try:
                    _fit_model_with_nb_weights(
                        model,
                        model_name,
                        use_cw,
                        X_train_sampled,
                        y_train_sampled,
                    )
                except Exception as exc:
                    msg = str(exc).lower()
                    if ("gpu" in msg or "cuda" in msg or "device" in msg) and model_name in {
                        "xgboost",
                        "catboost",
                        "lightgbm",
                    }:
                        if model_name == "xgboost":
                            model.set_params(device="cpu")
                        elif model_name == "catboost":
                            model.set_params(task_type="CPU")
                        elif model_name == "lightgbm":
                            model.set_params(device_type="cpu")
                        _fit_model_with_nb_weights(
                            model,
                            model_name,
                            use_cw,
                            X_train_sampled,
                            y_train_sampled,
                        )
                    else:
                        continue

                p_cal = np.clip(model.predict_proba(preprocess_artifacts.X_cal_proc)[:, 1], 1e-9, 1 - 1e-9)
                p_cal = _apply_nb_probability_floor(model, p_cal)

                local_best_score = -np.inf
                local_best_threshold = 0.5
                for threshold in threshold_grid:
                    preds = (p_cal >= threshold).astype(int)
                    objective = 0.60 * accuracy_score(preprocess_artifacts.y_cal, preds) + 0.40 * recall_score(
                        preprocess_artifacts.y_cal,
                        preds,
                        zero_division=0,
                    )
                    if objective > local_best_score:
                        local_best_score = objective
                        local_best_threshold = float(threshold)

                if local_best_score > best_score:
                    best_score = local_best_score
                    best_model = model
                    best_threshold = local_best_threshold
                    best_tr_data = (X_train_sampled, y_train_sampled)

            if best_model is None:
                print("    Final training failed: no valid model")
                continue

            p_test = np.clip(best_model.predict_proba(preprocess_artifacts.X_test_proc)[:, 1], 1e-9, 1 - 1e-9)
            p_test = _apply_nb_probability_floor(best_model, p_test)
            metrics = metric_pack(preprocess_artifacts.y_test, p_test, best_threshold)

            trained_models[key] = best_model
            train_data_by_key[key] = best_tr_data if best_tr_data is not None else (preprocess_artifacts.X_train_proc, preprocess_artifacts.y_train)
            model_thresholds[key] = best_threshold

            pre_rows.append(
                {
                    "experiment": MODEL_EXPERIMENT.get(model_name, "U"),
                    "sampling": sampling,
                    "model": model_name,
                    "threshold": best_threshold,
                    **{k: round(v, 4) for k, v in metrics.items()},
                }
            )

            print(
                "    Test pre-cal metrics: "
                f"acc={metrics['accuracy']:.3f} rec={metrics['recall']:.3f} auc={metrics['auc']:.3f}"
            )

    stage1_summary_rows: list[dict[str, Any]] = []
    for (sampling, model_name), frame in stage1_results.items():
        if frame.empty:
            continue
        top = frame.iloc[0]
        stage1_summary_rows.append(
            {
                "sampling": sampling,
                "model": model_name,
                "best_stage1_score": float(top.get("stage_score", np.nan)),
                "best_acc_cv": float(top.get("accuracy_mean", np.nan)),
                "best_rec_cv": float(top.get("recall_mean", np.nan)),
            }
        )

    stage2_summary_rows: list[dict[str, Any]] = []
    for (sampling, model_name), frame in stage2_results.items():
        if frame.empty:
            continue
        top = frame.iloc[0]
        stage2_summary_rows.append(
            {
                "sampling": sampling,
                "model": model_name,
                "best_stage2_score": float(top.get("stage_score", np.nan)),
                "best_acc_cv": float(top.get("accuracy_mean", np.nan)),
                "best_rec_cv": float(top.get("recall_mean", np.nan)),
                "best_thr_cv": float(top.get("threshold_mean", np.nan)),
            }
        )

    pd.DataFrame(stage1_summary_rows).to_csv(output_dir / "stage1_summary.csv", index=False)
    pd.DataFrame(stage2_summary_rows).to_csv(output_dir / "stage2_summary.csv", index=False)

    if not pre_rows:
        raise RuntimeError("No models were trained successfully.")

    pre_df = pd.DataFrame(pre_rows)
    pre_df["combined"] = (pre_df["accuracy"] + pre_df["recall"]) / 2.0
    pre_df = pre_df.sort_values(["combined", "auc"], ascending=False).reset_index(drop=True)
    pre_df.index += 1
    pre_df.to_csv(output_dir / "pre_calibration_results.csv", index=True, index_label="rank")

    print("Saved pre_calibration_results.csv")

    calibrator_store: dict[tuple[str, str, str], Any] = {}
    post_rows: list[dict[str, Any]] = []

    effective_cal_methods = CAL_METHODS.copy()
    if not HAS_VENN_ABERS:
        effective_cal_methods = [m for m in effective_cal_methods if m != "venn_abers"]

    for (sampling, model_name), model in trained_models.items():
        p_cal = np.clip(model.predict_proba(preprocess_artifacts.X_cal_proc)[:, 1], 1e-9, 1 - 1e-9)
        p_test = np.clip(model.predict_proba(preprocess_artifacts.X_test_proc)[:, 1], 1e-9, 1 - 1e-9)
        p_cal = _apply_nb_probability_floor(model, p_cal)
        p_test = _apply_nb_probability_floor(model, p_test)

        for cal_method in effective_cal_methods:
            try:
                calibrator = fit_calibrator(cal_method, p_cal, preprocess_artifacts.y_cal)
                p_test_cal = apply_calibrator(cal_method, calibrator, p_test)
                metrics = metric_pack(preprocess_artifacts.y_test, p_test_cal, threshold=0.5)

                row = {
                    "experiment": MODEL_EXPERIMENT.get(model_name, "U"),
                    "sampling": sampling,
                    "model": model_name,
                    "calibration": cal_method,
                    **{k: round(v, 4) for k, v in metrics.items()},
                }
                post_rows.append(row)
                calibrator_store[(sampling, model_name, cal_method)] = calibrator
            except Exception as exc:
                post_rows.append(
                    {
                        "experiment": MODEL_EXPERIMENT.get(model_name, "U"),
                        "sampling": sampling,
                        "model": model_name,
                        "calibration": cal_method,
                        "error": str(exc),
                    }
                )

    post_df = pd.DataFrame([row for row in post_rows if "error" not in row])
    if post_df.empty:
        raise RuntimeError("No calibrated results were produced.")

    post_df["combined"] = (
        post_df["accuracy"]
        + post_df["recall"]
        + (1.0 - post_df["ece"])
        + (1.0 - post_df["logloss"])
    ) / 4.0
    post_df = post_df.sort_values(["combined", "auc"], ascending=False).reset_index(drop=True)
    post_df.index += 1
    post_df.to_csv(output_dir / "post_calibration_results.csv", index=True, index_label="rank")

    print("Saved post_calibration_results.csv")

    best_row = post_df.iloc[0]
    best_sampling = str(best_row["sampling"])
    best_model_name = str(best_row["model"])
    best_cal_method = str(best_row["calibration"])
    best_key = (best_sampling, best_model_name)

    best_model = trained_models[best_key]
    best_calibrator = calibrator_store.get((best_sampling, best_model_name, best_cal_method))
    best_threshold = float(model_thresholds.get(best_key, 0.5))

    pre_entry = pre_df[(pre_df["sampling"] == best_sampling) & (pre_df["model"] == best_model_name)]
    if not pre_entry.empty:
        best_threshold = float(pre_entry.iloc[0]["threshold"])

    bundle = {
        "model": best_model,
        "knn_imputer": preprocess_artifacts.knn_imputer,
        "cat_imputer": preprocess_artifacts.cat_imputer,
        "scaler": preprocess_artifacts.scaler,
        "ohe": preprocess_artifacts.ohe,
        "num_cols_full": list(preprocess_artifacts.num_cols_full),
        "num_cols_reduced": list(preprocess_artifacts.num_cols_reduced),
        "cat_cols": list(preprocess_artifacts.cat_cols),
        "feat_names": list(preprocess_artifacts.feature_names),
        "input_feature_names": list(X.columns),
        "threshold": best_threshold,
        "sampling": best_sampling,
        "model_name": best_model_name,
        "calibration": best_cal_method,
        "metrics": {
            "accuracy": float(best_row["accuracy"]),
            "recall": float(best_row["recall"]),
            "precision": float(best_row["precision"]),
            "f1": float(best_row["f1"]),
            "auc": float(best_row["auc"]),
            "logloss": float(best_row["logloss"]),
            "ece": float(best_row["ece"]),
        },
    }

    bundle_path = models_dir / "best_model_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    if best_calibrator is not None:
        calibrator_path = models_dir / "best_calibrator.joblib"
        joblib.dump(best_calibrator, calibrator_path, compress=3)

    with open(output_dir / "best_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": MODEL_EXPERIMENT.get(best_model_name, "U"),
                "sampling": best_sampling,
                "model": best_model_name,
                "calibration": best_cal_method,
                "threshold": best_threshold,
                "metrics": bundle["metrics"],
                "bundle_path": str(bundle_path),
            },
            f,
            indent=2,
        )

    print("Best model selected")
    print(f"  Model       : {best_model_name}")
    print(f"  Sampling    : {best_sampling}")
    print(f"  Calibration : {best_cal_method}")
    print(f"  Bundle      : {bundle_path}")

    local_idx = 0
    if not args.skip_explainability:
        try:
            local_idx = run_shap(
                best_model=best_model,
                best_model_name=best_model_name,
                best_sampling=best_sampling,
                X_train_proc=train_data_by_key[best_key][0],
                X_test_proc=preprocess_artifacts.X_test_proc,
                y_test=preprocess_artifacts.y_test,
                feature_names=preprocess_artifacts.feature_names,
                output_dir=output_dir,
                shap_test_n=args.shap_test_n,
                shap_bg_n=args.shap_bg_n,
            )
        except Exception as exc:
            print(f"SHAP stage failed: {exc}")

        try:
            run_lime(
                best_model=best_model,
                X_train_proc=train_data_by_key[best_key][0],
                X_test_proc=preprocess_artifacts.X_test_proc,
                feature_names=preprocess_artifacts.feature_names,
                local_idx=local_idx,
                output_dir=output_dir,
                lime_features=args.lime_features,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"LIME stage failed: {exc}")

    print("-" * 90)
    print("Pipeline complete")
    print(f"Pre-cal results : {output_dir / 'pre_calibration_results.csv'}")
    print(f"Post-cal results: {output_dir / 'post_calibration_results.csv'}")
    print(f"Best bundle     : {bundle_path}")
    print("-" * 90)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified EXP3 pipeline: merge, preprocess, EDA, train, calibrate, select best, SHAP/LIME."
    )
    parser.add_argument("--root", type=str, default=None, help="Training root containing Datasets2015.")
    parser.add_argument("--out-dir", type=str, default="exp3_unified", help="Output subdirectory name under training root.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--skip-merge", action="store_true", help="Skip dataset merge stage.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA stage.")
    parser.add_argument("--skip-explainability", action="store_true", help="Skip SHAP and LIME stages.")

    parser.add_argument(
        "--sampling-methods",
        nargs="*",
        default=SAMPLING_METHODS,
        choices=SAMPLING_METHODS,
        help="Sampling methods to run.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=["knn", "randomforest", "xgboost", "adaboost", "logreg", "catboost", "lightgbm", "naive_bayes"],
        help="Subset of models to run.",
    )

    parser.add_argument("--s1-trials", type=int, default=None, help="Override Stage-1 trials for all models.")
    parser.add_argument("--s1-folds", type=int, default=None, help="Override Stage-1 folds for all models.")
    parser.add_argument("--s1-epochs", type=int, default=None, help="Override Stage-1 epochs for all models.")

    parser.add_argument("--s2-refine", type=int, default=None, help="Override Stage-2 refinement count for all models.")
    parser.add_argument("--s2-folds", type=int, default=None, help="Override Stage-2 folds for all models.")
    parser.add_argument("--s2-epochs", type=int, default=None, help="Override Stage-2 epochs for all models.")

    parser.add_argument("--top-k-s1", type=int, default=None, help="Override Stage-1 top-k carryover for all models.")
    parser.add_argument("--top-k-s2", type=int, default=None, help="Override Stage-2 top-k finalists for all models.")
    parser.add_argument("--final-epochs", type=int, default=None, help="Override final-fit epochs for all models.")

    parser.add_argument("--collinearity-cutoff", type=float, default=0.70)

    parser.add_argument("--threshold-min", type=float, default=0.35)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.05)

    parser.add_argument("--shap-test-n", type=int, default=300)
    parser.add_argument("--shap-bg-n", type=int, default=100)
    parser.add_argument("--lime-features", type=int, default=15)

    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--use-gpu-when-available",
        dest="use_gpu_when_available",
        action="store_true",
        help="Enable GPU usage when CUDA is available.",
    )
    gpu_group.add_argument(
        "--no-gpu",
        dest="use_gpu_when_available",
        action="store_false",
        help="Force CPU mode even if CUDA is available.",
    )
    parser.set_defaults(use_gpu_when_available=True)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    required_positive_fields = ["shap_test_n", "shap_bg_n", "lime_features"]
    for field in required_positive_fields:
        value = int(getattr(args, field))
        if value < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be >= 1.")

    optional_positive_fields = [
        "s1_trials",
        "s1_folds",
        "s1_epochs",
        "s2_folds",
        "s2_epochs",
        "top_k_s1",
        "top_k_s2",
        "final_epochs",
    ]
    for field in optional_positive_fields:
        value = getattr(args, field)
        if value is not None and int(value) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be >= 1 when provided.")

    if args.s2_refine is not None and int(args.s2_refine) < 0:
        raise ValueError("--s2-refine must be >= 0 when provided.")

    if int(args.seed) < 0:
        raise ValueError("--seed must be >= 0.")

    if args.s1_folds is not None and args.s1_folds < 2:
        raise ValueError("--s1-folds must be >= 2 when provided.")
    if args.s2_folds is not None and args.s2_folds < 2:
        raise ValueError("--s2-folds must be >= 2 when provided.")

    if not (0.0 <= float(args.threshold_min) <= 1.0):
        raise ValueError("--threshold-min must be within [0, 1].")
    if not (0.0 <= float(args.threshold_max) <= 1.0):
        raise ValueError("--threshold-max must be within [0, 1].")
    if float(args.threshold_min) > float(args.threshold_max):
        raise ValueError("--threshold-min must be <= --threshold-max.")
    if float(args.threshold_step) <= 0:
        raise ValueError("--threshold-step must be > 0.")

    if not (0.0 < float(args.collinearity_cutoff) < 1.0):
        raise ValueError("--collinearity-cutoff must be between 0 and 1 (exclusive).")

    if args.models is not None and len(args.models) == 0:
        raise ValueError("--models was provided but empty.")
    if args.sampling_methods is not None and len(args.sampling_methods) == 0:
        raise ValueError("--sampling-methods was provided but empty.")

    # top-k vs trial-count consistency is applied per-model during run-time
    # after notebook-faithful defaults and optional CLI overrides are resolved.


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args)

    run_pipeline(args)


if __name__ == "__main__":
    main()
