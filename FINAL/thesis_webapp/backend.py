from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from venn_abers import VennAbers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "training"


def _first_existing(paths: list[Path], fallback: Path) -> Path:
    return next((path for path in paths if path.exists()), fallback)


# ── Auto-select best EXP3 bundle from updated runs (A/B/C/D) ─────────────────
_EXP3_CANDIDATES = [
    ("exp3_knn_rf", TRAINING_ROOT / "exp3_knn_rf"),
    ("exp3_xgb_ada", TRAINING_ROOT / "exp3_xgb_ada"),
    ("exp3_logreg_cat", TRAINING_ROOT / "exp3_logreg_cat"),
    ("exp3_naive_bayes", TRAINING_ROOT / "exp3_naive_bayes"),
    ("exp3_knn_rf", PROJECT_ROOT / "exp3_knn_rf"),
    ("exp3_xgb_ada", PROJECT_ROOT / "exp3_xgb_ada"),
    ("exp3_logreg_cat", PROJECT_ROOT / "exp3_logreg_cat"),
    ("exp3_naive_bayes", PROJECT_ROOT / "exp3_naive_bayes"),
]


def _select_best_exp3_bundle() -> Path | None:
    best_path: Path | None = None
    best_combined = -np.inf

    for _, exp_dir in _EXP3_CANDIDATES:
        bundle_path = exp_dir / "models" / "best_model_bundle.joblib"
        post_cal_path = exp_dir / "post_calibration_results.csv"

        if not bundle_path.exists() or not post_cal_path.exists():
            continue

        try:
            top_row = pd.read_csv(post_cal_path, nrows=1)
            if top_row.empty or "combined" not in top_row.columns:
                continue
            combined = float(top_row.iloc[0]["combined"])
        except Exception:
            continue

        if combined > best_combined:
            best_combined = combined
            best_path = bundle_path

    return best_path


EXP3_BUNDLE_PATH = _select_best_exp3_bundle()

# Fall back to old GPU-exp2 artifacts when EXP3 bundles are unavailable.
_OLD_MODEL_CANDIDATES = [
    PROJECT_ROOT / "gpu_rf_xgb_cat_exp2_artifacts" / "models" / "calibrated_top_models" / "top3_catboost_isotonic.joblib",
    PROJECT_ROOT.parent / "gpu_rf_xgb_cat_exp2_artifacts" / "models" / "calibrated_top_models" / "top3_catboost_isotonic.joblib",
]
_OLD_PREPROCESSOR_CANDIDATES = [
    PROJECT_ROOT / "gpu_rf_xgb_cat_exp2_artifacts" / "preprocessor.joblib",
    PROJECT_ROOT.parent / "gpu_rf_xgb_cat_exp2_artifacts" / "preprocessor.joblib",
]

_OLD_MODEL_PATH = _first_existing(_OLD_MODEL_CANDIDATES, _OLD_MODEL_CANDIDATES[0])
_OLD_PREPROCESSOR_PATH = _first_existing(_OLD_PREPROCESSOR_CANDIDATES, _OLD_PREPROCESSOR_CANDIDATES[0])

# Pinned to EXP3-B (XGBoost cw base) — best recall (76.5%) across updated runs.
_PINNED_BUNDLE_CANDIDATES = [
    TRAINING_ROOT / "exp3_xgb_ada" / "models" / "best_model_bundle.joblib",
    PROJECT_ROOT / "exp3_xgb_ada" / "models" / "best_model_bundle.joblib",
]
_PINNED_BUNDLE = _first_existing(_PINNED_BUNDLE_CANDIDATES, _PINNED_BUNDLE_CANDIDATES[0])

DEFAULT_MODEL_PATH = _PINNED_BUNDLE if _PINNED_BUNDLE.exists() else (EXP3_BUNDLE_PATH if EXP3_BUNDLE_PATH is not None and EXP3_BUNDLE_PATH.exists() else _OLD_MODEL_PATH)
DEFAULT_PREPROCESSOR_PATH = _PINNED_BUNDLE if _PINNED_BUNDLE.exists() else (EXP3_BUNDLE_PATH if EXP3_BUNDLE_PATH is not None and EXP3_BUNDLE_PATH.exists() else _OLD_PREPROCESSOR_PATH)

_calibrator_candidates = [
    PROJECT_ROOT / "main_2015_balanced_gpu_artifacts" / "models" / "venn_abers_calibrator.joblib",
    WORKSPACE_ROOT / "main_2015_balanced_gpu_artifacts" / "models" / "venn_abers_calibrator.joblib",
]

# EXP3 bundles already encode their selected calibration strategy in the bundle
# metadata, so avoid applying an unrelated external calibrator by default.
if DEFAULT_MODEL_PATH == _OLD_MODEL_PATH:
    DEFAULT_CALIBRATOR_PATH = next((path for path in _calibrator_candidates if path.exists()), _calibrator_candidates[0])
else:
    DEFAULT_CALIBRATOR_PATH = PROJECT_ROOT / "__no_external_calibrator__.joblib"

ONE_HOT_GROUPS = {
    "alcohol_level": ["0.0", "1.0", "2.0", "3.0", "nan"], "smoking_level": ["0.0", "1.0", "2.0", "3.0", "nan"],
}

# EXP3-C uses numeric fe_smoking_level / fe_alcohol_level instead of OHE columns.
ENGINEERED_TARGET_FEATURES = {
    "alcohol_level", "smoking_level",
    "fe_alcohol_level", "fe_smoking_level",
    "BMI", "bmi", "whr",
}

NUMERIC_DEFAULTS = {
    "ethnicity": 1.0, "waist": 0.0, "hip": 0.0, "BMI": 0.0, "bmi": 0.0, "whr": 0.0,
    "fe_smoking_level": 0.0, "fe_alcohol_level": 0.0,
    "Total_Food_epwt": 0.0, "Total_Ener": 0.0, "Total_Prot": 0.0, "Total_Calc": 0.0,
    "Total_Iron": 0.0, "Total_VitA": 0.0, "Total_VitC": 0.0,
    "Total_Thia": 0.0, "Total_Ribo": 0.0, "Total_Nia": 0.0, "Total_CHO": 0.0, "Total_Fat": 0.0,
}

RANGE_HINTS = {
    "ethnicity": (0.0, 10.0, 1.0), "waist": (40.0, 180.0, 0.5), "hip": (40.0, 180.0, 0.5),
    "BMI": (10.0, 60.0, 0.1), "bmi": (10.0, 60.0, 0.1), "whr": (0.5, 2.0, 0.01),
    "fe_smoking_level": (0.0, 3.0, 1.0), "fe_alcohol_level": (0.0, 3.0, 1.0),
    "Total_Food_epwt": (0.0, 1760.0, 5.0), "Total_Ener": (0.0, 3890.0, 10.0), "Total_Prot": (0.0, 150.0, 1.0),
    "Total_Calc": (0.0, 1270.0, 10.0), "Total_Iron": (0.0, 30.0, 0.5), "Total_VitA": (0.0, 4810.0, 10.0),
    "Total_VitC": (0.0, 190.0, 1.0), "Total_Thia": (0.0, 10.0, 0.05), "Total_Ribo": (0.0, 10.0, 0.05),
    "Total_Nia": (0.0, 50.0, 0.5), "Total_CHO": (0.0, 710.0, 5.0), "Total_Fat": (0.0, 120.0, 1.0),
    # Dietary food-group weights (edible-portion grams) – maxima from dataset p99 × 1.1
    "epwt_fg1": (0.0, 870.0, 1.0), "epwt_fg2": (0.0, 790.0, 1.0), "epwt_fg3": (0.0, 480.0, 1.0),
    "epwt_fg4": (0.0, 340.0, 1.0), "epwt_fg5": (0.0, 480.0, 1.0), "epwt_fg6": (0.0, 690.0, 1.0),
    "epwt_fg7": (0.0, 200.0, 1.0), "epwt_fg8": (0.0, 440.0, 1.0), "epwt_fg9": (0.0, 260.0, 1.0),
    "epwt_fg10": (0.0, 400.0, 1.0), "epwt_fg11": (0.0, 630.0, 1.0), "epwt_fg12": (0.0, 550.0, 1.0),
    "epwt_fg13": (0.0, 600.0, 1.0), "epwt_fg14": (0.0, 530.0, 1.0), "epwt_fg15": (0.0, 360.0, 1.0),
    "epwt_fg16": (0.0, 510.0, 1.0), "epwt_fg17": (0.0, 380.0, 1.0), "epwt_fg18": (0.0, 180.0, 1.0),
    "epwt_fg19": (0.0, 270.0, 1.0), "epwt_fg20": (0.0, 240.0, 1.0), "epwt_fg21": (0.0, 320.0, 1.0),
    "epwt_fg23": (0.0, 60.0, 0.5), "epwt_fg24": (0.0, 590.0, 1.0), "epwt_fg25": (0.0, 590.0, 1.0),
    "epwt_fg26": (0.0, 60.0, 0.5), "epwt_fg27": (0.0, 550.0, 1.0),
}


@dataclass(frozen=True)
class PredictionResult:
    raw_probability: float
    calibrated_probability: float
    lower_bound: float
    upper_bound: float
    uncertainty_width: float


def _compute_bootstrap_interval(
    base_model: Any,
    input_frame: pd.DataFrame,
    n_samples: int = 60,
    noise_std: float = 0.25,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate a 10th–90th percentile prediction interval via input perturbation.

    Adds Gaussian noise (±noise_std in model-feature / standardised space) to
    the single input row, obtains n_samples predictions in one batched call,
    and returns (p10, p90).  This reflects how sensitive the risk score is to
    typical measurement uncertainty in the input features.
    """
    rng = np.random.default_rng(seed)
    base = input_frame.values  # (1, n_features)
    noise = rng.normal(0.0, noise_std, size=(n_samples, base.shape[1]))
    perturbed = pd.DataFrame(base + noise, columns=input_frame.columns)
    try:
        probs = np.asarray(base_model.predict_proba(perturbed))[:, 1]
        return float(np.percentile(probs, 10)), float(np.percentile(probs, 90))
    except Exception:
        p = float(np.asarray(base_model.predict_proba(input_frame))[0, 1])
        return p, p


class Exp3BundlePreprocessor:
    """Sklearn-compatible transformer wrapping the EXP3-C multi-step preprocessing bundle.

    Pipeline: KNN impute (full num cols) → collinearity select (reduced num cols)
              → StandardScaler + OneHotEncoder → hstack output.
    """

    def __init__(self, bundle: dict) -> None:
        self._b = bundle
        self.feature_names_in_ = np.array(bundle["input_feature_names"])

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        b = self._b
        num_full = b["num_cols_full"]
        num_red  = b["num_cols_reduced"]
        cat      = b["cat_cols"]

        # --- impute numerics (full set) ---
        if num_full:
            num_imp = pd.DataFrame(
                b["knn_imputer"].transform(X[num_full]),
                columns=num_full, index=X.index,
            )
        else:
            num_imp = pd.DataFrame(index=X.index)

        # --- impute categoricals ---
        if cat and b.get("cat_imputer") is not None:
            cat_imp = pd.DataFrame(
                b["cat_imputer"].transform(X[cat]),
                columns=cat, index=X.index,
            )
        else:
            cat_imp = pd.DataFrame(index=X.index)

        # --- scale surviving numeric cols + OHE categoricals ---
        parts: list[np.ndarray] = []
        if num_red:
            parts.append(b["scaler"].transform(num_imp[num_red]))
        if cat and b.get("ohe") is not None:
            parts.append(b["ohe"].transform(cat_imp[cat].astype(str)))

        return np.hstack(parts) if parts else np.empty((len(X), 0), dtype=float)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(self._b["feat_names"])


def _is_exp3_bundle(obj: Any) -> bool:
    return isinstance(obj, dict) and "knn_imputer" in obj


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    obj = joblib.load(Path(model_path))
    if _is_exp3_bundle(obj):
        # Wrap as the dict format predict_with_venn_abers expects.
        return {
            "base_model":         obj["model"],
            "feature_names":      obj["feat_names"],
            "calibration_method": obj.get("calibration", "base"),
            "calibrator":         None,
            "threshold":          obj.get("threshold", 0.5),
        }
    return obj


def load_calibrator(calibrator_path: Path | str = DEFAULT_CALIBRATOR_PATH):
    return joblib.load(path) if (path := Path(calibrator_path)).exists() else None


def load_preprocessor(preprocessor_path: Path | str = DEFAULT_PREPROCESSOR_PATH):
    path = Path(preprocessor_path)
    if not path.exists():
        return None
    obj = joblib.load(path)
    if _is_exp3_bundle(obj):
        return Exp3BundlePreprocessor(obj)
    return obj


def load_feature_names(model: Any) -> list[str]:
    if isinstance(model, dict):
        feature_names = model.get("feature_names")
        if feature_names:
            return [str(name) for name in feature_names]
        if "base_model" in model:
            return load_feature_names(model["base_model"])
    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return [str(name) for name in step.feature_names_in_]
    raise ValueError("The model does not expose feature names.")


def unwrap_model(model: Any) -> Any:
    return model["base_model"] if isinstance(model, dict) and "base_model" in model else model


def load_input_feature_names(model: Any, preprocessor: Any | None = None) -> list[str]:
    return [str(name) for name in preprocessor.feature_names_in_] if preprocessor is not None and hasattr(preprocessor, "feature_names_in_") else load_feature_names(model)


def load_model_feature_names(model: Any, preprocessor: Any | None = None) -> list[str]:
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            pass
    return load_feature_names(model)


def make_input_frame(feature_names: list[str], values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{name: values.get(name, 0.0) for name in feature_names}], columns=feature_names)


def prepare_model_input(input_frame: pd.DataFrame, preprocessor: Any | None = None) -> pd.DataFrame:
    if preprocessor is None:
        return input_frame

    transformed = preprocessor.transform(input_frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)
    if transformed.ndim == 1:
        transformed = transformed.reshape(1, -1)

    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            columns = [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            columns = [f"feature_{index}" for index in range(transformed.shape[1])]
    else:
        columns = [f"feature_{index}" for index in range(transformed.shape[1])]

    return pd.DataFrame(transformed, columns=columns)


def predict_with_venn_abers(
    model: Any,
    input_frame: pd.DataFrame,
    calibrator: Any | None = None,
) -> PredictionResult:
    raw_probability = float(np.asarray(unwrap_model(model).predict_proba(input_frame))[0, 1])

    # Prefer explicit external Venn-Abers calibrator when supplied.
    # This keeps uncertainty intervals available even if the wrapped model
    # uses a point calibrator (e.g., isotonic/platt) internally.
    if calibrator is not None:
        probability_pair = np.array([[1.0 - float(np.clip(raw_probability, 1e-9, 1 - 1e-9)), float(np.clip(raw_probability, 1e-9, 1 - 1e-9))]], dtype=float)
        calibrated_pair, p0_p1 = calibrator.predict_proba(p_test=probability_pair)

        calibrated_probability = float(calibrated_pair[0, 1])
        lower_bound = float(np.clip(min(p0_p1[0, 0], p0_p1[0, 1], calibrated_probability), 0.0, 1.0))
        upper_bound = float(np.clip(max(p0_p1[0, 0], p0_p1[0, 1], calibrated_probability), 0.0, 1.0))

        return PredictionResult(
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            uncertainty_width=max(0.0, upper_bound - lower_bound),
        )

    if isinstance(model, dict):
        method = str(model.get("calibration_method", "base"))
        wrapped_calibrator = model.get("calibrator")

        if method == "base" or wrapped_calibrator is None:
            calibrated_probability = raw_probability
            lower_bound, upper_bound = _compute_bootstrap_interval(
                unwrap_model(model), input_frame
            )
        elif method in {"isotonic", "platt", "sigmoid"}:
            calibrated_probability = float(np.clip(wrapped_calibrator.predict(np.array([raw_probability]))[0], 0.0, 1.0))
            lower_bound = calibrated_probability
            upper_bound = calibrated_probability
        elif method == "venn_abers":
            probability_pair = np.array([[1.0 - float(np.clip(raw_probability, 1e-9, 1 - 1e-9)), float(np.clip(raw_probability, 1e-9, 1 - 1e-9))]], dtype=float)
            calibrated_pair, p0_p1 = wrapped_calibrator.predict_proba(p_test=probability_pair)
            calibrated_probability = float(calibrated_pair[0, 1])
            lower_bound = float(np.clip(min(p0_p1[0, 0], p0_p1[0, 1], calibrated_probability), 0.0, 1.0))
            upper_bound = float(np.clip(max(p0_p1[0, 0], p0_p1[0, 1], calibrated_probability), 0.0, 1.0))
        else:
            calibrated_probability = raw_probability
            lower_bound = raw_probability
            upper_bound = raw_probability

        return PredictionResult(
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            uncertainty_width=max(0.0, upper_bound - lower_bound),
        )

    return PredictionResult(
        raw_probability=raw_probability,
        calibrated_probability=raw_probability,
        lower_bound=raw_probability,
        upper_bound=raw_probability,
        uncertainty_width=0.0,
    )


def predict_with_venn_abers_safe(
    model: Any,
    input_frame: "pd.DataFrame",
    calibrator: Any | None = None,
    timeout: int = 60,
) -> "PredictionResult":
    """Run predict_with_venn_abers in an isolated subprocess.

    Prevents XGBoost's OpenMP thread pool from being initialised in the
    Streamlit main process, which would cause os._exit() on Windows when
    Streamlit's own threads try to run concurrently.

    Falls back to in-process prediction if the payload cannot be pickled.
    """
    import pathlib
    import pickle
    import subprocess
    import sys
    import tempfile
    import uuid

    _dir = pathlib.Path(__file__).parent
    worker = _dir / "_predict_worker.py"

    payload = {"model": model, "input_frame": input_frame, "calibrator": calibrator}
    try:
        _bytes = pickle.dumps(payload)
        del _bytes
    except Exception:
        return predict_with_venn_abers(model, input_frame, calibrator)

    uid = uuid.uuid4().hex
    tmp_dir = pathlib.Path(tempfile.gettempdir())
    tmp_in = tmp_dir / f"_pred_in_{uid}.pkl"
    tmp_out = tmp_dir / f"_pred_out_{uid}.pkl"

    try:
        with open(tmp_in, "wb") as fh:
            pickle.dump(payload, fh)

        proc = subprocess.Popen(
            [sys.executable, str(worker), str(tmp_in), str(tmp_out)],
            cwd=str(_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return predict_with_venn_abers(model, input_frame, calibrator)

        if not tmp_out.exists():
            return predict_with_venn_abers(model, input_frame, calibrator)

        with open(tmp_out, "rb") as fh:
            result = pickle.load(fh)

        return result

    finally:
        for f in (tmp_in, tmp_out):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


def feature_default(feature_name: str) -> float:
    return float(NUMERIC_DEFAULTS.get(feature_name, 0.0))


def feature_range(feature_name: str) -> tuple[float, float, float]:
    if feature_name in RANGE_HINTS:
        return RANGE_HINTS[feature_name]
    if feature_name.startswith("epwt_fg") or feature_name.startswith("fg"):
        return (0.0, 500.0, 1.0)
    if feature_name.startswith("Total_"):
        return (0.0, 5000.0, 1.0)
    if feature_name.endswith("_nan"):
        return (0.0, 1.0, 1.0)
    return (0.0, 100.0, 1.0)


def group_feature_names(feature_names: list[str]) -> dict[str, list[str]]:
    grouped = {
        "Core clinical": [],
        "Anthropometrics": [],
        "Dietary pattern": [],
        "Lifestyle encodings": [],
        "Other model inputs": [],
    }

    for feature_name in feature_names:
        if feature_name.startswith("alcohol_level_") or feature_name.startswith("smoking_level_"):
            grouped["Lifestyle encodings"].append(feature_name)
        elif feature_name in {"fe_smoking_level", "fe_alcohol_level"}:
            # EXP3-C engineered behavioral features — computed, not shown directly.
            grouped["Lifestyle encodings"].append(feature_name)
        elif feature_name in {"pa_met", "fbs", "chol", "tri", "hdl", "ldl", "ethnicity",
                               "hemoglobin", "uic", "vita"}:
            grouped["Core clinical"].append(feature_name)
        elif feature_name in {"waist", "hip", "BMI", "bmi", "whr"}:
            # bmi and whr are computed from raw inputs, not shown directly.
            grouped["Anthropometrics"].append(feature_name)
        elif feature_name.startswith("epwt_fg") or feature_name.startswith("fg") or feature_name.startswith("Total_"):
            grouped["Dietary pattern"].append(feature_name)
        else:
            grouped["Other model inputs"].append(feature_name)

    return {group: names for group, names in grouped.items() if names}


def build_input_values_from_widgets(feature_names: list[str], widget_values: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}

    alcohol_level = _compute_alcohol_level(widget_values)
    smoking_level = _compute_smoking_level(widget_values)
    bmi_value = _compute_bmi(widget_values)
    whr_value = _compute_whr(widget_values)

    _populate_one_hot_group(values, feature_names, "alcohol_level", alcohol_level)
    _populate_one_hot_group(values, feature_names, "smoking_level", smoking_level)

    if "alcohol_level" in feature_names:
        values["alcohol_level"] = float(alcohol_level if alcohol_level is not None else feature_default("alcohol_level"))
    if "smoking_level" in feature_names:
        values["smoking_level"] = float(smoking_level if smoking_level is not None else feature_default("smoking_level"))

    # EXP3-C numeric engineered behavioral features.
    if "fe_smoking_level" in feature_names:
        values["fe_smoking_level"] = float(smoking_level if smoking_level is not None else 0.0)
    if "fe_alcohol_level" in feature_names:
        values["fe_alcohol_level"] = float(alcohol_level if alcohol_level is not None else 0.0)

    if bmi_value is not None:
        if "BMI" in feature_names:
            values["BMI"] = float(bmi_value)
        if "bmi" in feature_names:
            values["bmi"] = float(bmi_value)

    if whr_value is not None and "whr" in feature_names:
        values["whr"] = float(whr_value)

    for feature_name in feature_names:
        if feature_name in values:
            continue
        if feature_name.startswith("alcohol_level_") or feature_name.startswith("smoking_level_"):
            values[feature_name] = 0.0
            continue
        if feature_name in ENGINEERED_TARGET_FEATURES:
            values[feature_name] = float(feature_default(feature_name))
            continue
        values[feature_name] = float(widget_values.get(feature_name, feature_default(feature_name)))

    return values


def _to_numeric_clean_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(numeric) or numeric in {9, 99, 888888, 999999} else numeric


def _compute_smoking_level(raw_values: dict[str, Any]) -> float | None:
    direct = _to_numeric_clean_scalar(raw_values.get("smoking_level"))
    if direct is not None:
        return float(np.clip(direct, 0.0, 3.0))

    smoke_status = _to_numeric_clean_scalar(raw_values.get("smoke_status"))
    current_smoking = _to_numeric_clean_scalar(raw_values.get("current_smoking"))
    ever_smk = _to_numeric_clean_scalar(raw_values.get("ever_smk"))

    if smoke_status is not None:
        level: float | None = None
        if smoke_status == 0:
            level = 0.0
        elif smoke_status == 2:
            level = 1.0
        elif smoke_status == 1:
            level = 2.0

        if smoke_status == 1 and current_smoking == 3:
            level = 3.0
        return None if level is None else float(np.clip(level, 0.0, 3.0))

    if current_smoking is not None:
        level: float | None = None
        if current_smoking == 0:
            level = 0.0
        elif current_smoking in {1, 2}:
            level = 2.0
        elif current_smoking == 3:
            level = 3.0

        if current_smoking == 0 and ever_smk is not None and ever_smk > 0:
            level = 1.0
        return None if level is None else float(np.clip(level, 0.0, 3.0))

    if ever_smk is not None:
        return 0.0 if ever_smk == 0 else 1.0

    return None


def _compute_alcohol_level(raw_values: dict[str, Any]) -> float | None:
    direct = _to_numeric_clean_scalar(raw_values.get("alcohol_level"))
    if direct is not None:
        return float(np.clip(direct, 0.0, 3.0))

    alcohol_status = _to_numeric_clean_scalar(raw_values.get("alcohol_status"))
    alcohol_ever = _to_numeric_clean_scalar(raw_values.get("alcohol"))
    con_alcohol = _to_numeric_clean_scalar(raw_values.get("con_alcohol"))
    drnk_30days = _to_numeric_clean_scalar(raw_values.get("drnk_30days"))
    binge_drink = _to_numeric_clean_scalar(raw_values.get("binge_drink"))

    if alcohol_status is not None:
        level: float | None = None
        if alcohol_status == 0:
            level = 0.0
        elif alcohol_status == 2:
            level = 1.0
        elif alcohol_status == 1:
            level = 2.0

        if alcohol_status == 1 and binge_drink == 1:
            level = 3.0
        return None if level is None else float(np.clip(level, 0.0, 3.0))

    level = 0.0
    used = False

    if alcohol_ever is not None:
        used = True
        if alcohol_ever > 0:
            level = max(level, 1.0)

    if con_alcohol is not None:
        used = True
        if con_alcohol == 1:
            level = max(level, 2.0)

    if drnk_30days is not None:
        used = True
        if drnk_30days == 1:
            level = max(level, 2.0)

    if binge_drink is not None:
        used = True
        if binge_drink == 1:
            level = 3.0

    if used:
        return float(np.clip(level, 0.0, 3.0))
    return None


def _compute_bmi(raw_values: dict[str, Any]) -> float | None:
    weight = _to_numeric_clean_scalar(raw_values.get("weight"))
    height = _to_numeric_clean_scalar(raw_values.get("height"))
    if weight is None or height is None or height <= 0:
        return None

    height_m = height / 100.0 if height > 3.0 else height
    if height_m <= 0:
        return None

    bmi = weight / (height_m**2)
    return float(bmi) if np.isfinite(bmi) else None


def _compute_whr(raw_values: dict[str, Any]) -> float | None:
    waist = _to_numeric_clean_scalar(raw_values.get("waist"))
    hip = _to_numeric_clean_scalar(raw_values.get("hip"))
    if waist is None or hip is None or hip == 0:
        return None

    whr = waist / hip
    return float(whr) if np.isfinite(whr) else None


def _populate_one_hot_group(values: dict[str, float], feature_names: list[str], group_name: str, level: float | None):
    group_prefix = f"{group_name}_"
    model_suffixes = [name[len(group_prefix) :] for name in feature_names if name.startswith(group_prefix)]
    suffixes = model_suffixes if model_suffixes else ONE_HOT_GROUPS[group_name]
    selected_suffix = "nan" if level is None else f"{float(level):.1f}"

    if selected_suffix not in suffixes and "nan" in suffixes:
        selected_suffix = "nan"

    for suffix in suffixes:
        values[f"{group_name}_{suffix}"] = 1.0 if suffix == selected_suffix else 0.0