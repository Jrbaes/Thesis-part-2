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


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "gpu_rf_xgb_cat_exp2_artifacts"
    / "models"
    / "calibrated_top_models"
    / "top3_catboost_isotonic.joblib"
)

_calibrator_candidates = [
    PROJECT_ROOT / "main_2015_balanced_gpu_artifacts" / "models" / "venn_abers_calibrator.joblib",
    WORKSPACE_ROOT / "main_2015_balanced_gpu_artifacts" / "models" / "venn_abers_calibrator.joblib",
]
DEFAULT_CALIBRATOR_PATH = next((path for path in _calibrator_candidates if path.exists()), _calibrator_candidates[0])

DEFAULT_PREPROCESSOR_PATH = PROJECT_ROOT / "gpu_rf_xgb_cat_exp2_artifacts" / "preprocessor.joblib"

ONE_HOT_GROUPS = {
    "alcohol_level": ["0.0", "1.0", "2.0", "3.0", "nan"],
    "smoking_level": ["0.0", "1.0", "2.0", "3.0", "nan"],
}

ENGINEERED_TARGET_FEATURES = {"alcohol_level", "smoking_level", "BMI", "bmi", "whr"}

NUMERIC_DEFAULTS = {
    "pa_met": 4.0,
    "fbs": 90.0,
    "chol": 180.0,
    "tri": 140.0,
    "hdl": 52.0,
    "ldl": 110.0,
    "ethnicity": 1.0,
    "waist": 0.0,
    "hip": 0.0,
    "BMI": 0.0,
    "Total_Food_epwt": 12.0,
    "Total_Ener": 1800.0,
    "Total_Prot": 70.0,
    "Total_Calc": 0.0,
    "Total_Iron": 0.0,
    "Total_VitA": 0.0,
    "Total_VitC": 0.0,
    "Total_Thia": 0.0,
    "Total_Ribo": 0.0,
    "Total_Nia": 0.0,
    "Total_CHO": 240.0,
    "Total_Fat": 65.0,
}

RANGE_HINTS = {
    "pa_met": (0.0, 25.0, 0.5),
    "fbs": (50.0, 350.0, 1.0),
    "chol": (100.0, 400.0, 1.0),
    "tri": (40.0, 600.0, 1.0),
    "hdl": (10.0, 120.0, 1.0),
    "ldl": (10.0, 260.0, 1.0),
    "ethnicity": (0.0, 10.0, 1.0),
    "waist": (40.0, 180.0, 0.5),
    "hip": (40.0, 180.0, 0.5),
    "BMI": (10.0, 60.0, 0.1),
    "Total_Food_epwt": (0.0, 1760.0, 5.0),
    "Total_Ener": (0.0, 3890.0, 10.0),
    "Total_Prot": (0.0, 150.0, 1.0),
    "Total_Calc": (0.0, 1270.0, 10.0),
    "Total_Iron": (0.0, 30.0, 0.5),
    "Total_VitA": (0.0, 4810.0, 10.0),
    "Total_VitC": (0.0, 190.0, 1.0),
    "Total_Thia": (0.0, 10.0, 0.05),
    "Total_Ribo": (0.0, 10.0, 0.05),
    "Total_Nia": (0.0, 50.0, 0.5),
    "Total_CHO": (0.0, 710.0, 5.0),
    "Total_Fat": (0.0, 120.0, 1.0),
    # Dietary food-group weights (edible-portion grams) – maxima from dataset p99 × 1.1
    "epwt_fg1": (0.0, 870.0, 1.0),
    "epwt_fg2": (0.0, 790.0, 1.0),
    "epwt_fg3": (0.0, 480.0, 1.0),
    "epwt_fg4": (0.0, 340.0, 1.0),
    "epwt_fg5": (0.0, 480.0, 1.0),
    "epwt_fg6": (0.0, 690.0, 1.0),
    "epwt_fg7": (0.0, 200.0, 1.0),
    "epwt_fg8": (0.0, 440.0, 1.0),
    "epwt_fg9": (0.0, 260.0, 1.0),
    "epwt_fg10": (0.0, 400.0, 1.0),
    "epwt_fg11": (0.0, 630.0, 1.0),
    "epwt_fg12": (0.0, 550.0, 1.0),
    "epwt_fg13": (0.0, 600.0, 1.0),
    "epwt_fg14": (0.0, 530.0, 1.0),
    "epwt_fg15": (0.0, 360.0, 1.0),
    "epwt_fg16": (0.0, 510.0, 1.0),
    "epwt_fg17": (0.0, 380.0, 1.0),
    "epwt_fg18": (0.0, 180.0, 1.0),
    "epwt_fg19": (0.0, 270.0, 1.0),
    "epwt_fg20": (0.0, 240.0, 1.0),
    "epwt_fg21": (0.0, 320.0, 1.0),
    "epwt_fg23": (0.0, 60.0, 0.5),
    "epwt_fg24": (0.0, 590.0, 1.0),
    "epwt_fg25": (0.0, 590.0, 1.0),
    "epwt_fg26": (0.0, 60.0, 0.5),
    "epwt_fg27": (0.0, 550.0, 1.0),
}


@dataclass(frozen=True)
class PredictionResult:
    raw_probability: float
    calibrated_probability: float
    lower_bound: float
    upper_bound: float
    uncertainty_width: float
    risk_label: str


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    return joblib.load(Path(model_path))


def load_calibrator(calibrator_path: Path | str = DEFAULT_CALIBRATOR_PATH):
    path = Path(calibrator_path)
    if not path.exists():
        return None
    return joblib.load(path)


def load_preprocessor(preprocessor_path: Path | str = DEFAULT_PREPROCESSOR_PATH):
    path = Path(preprocessor_path)
    if not path.exists():
        return None
    return joblib.load(path)


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
    if isinstance(model, dict) and "base_model" in model:
        return model["base_model"]
    return model


def load_input_feature_names(model: Any, preprocessor: Any | None = None) -> list[str]:
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return [str(name) for name in preprocessor.feature_names_in_]
    return load_feature_names(model)


def load_model_feature_names(model: Any, preprocessor: Any | None = None) -> list[str]:
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            pass
    return load_feature_names(model)


def make_input_frame(feature_names: list[str], values: dict[str, Any]) -> pd.DataFrame:
    row = {name: values.get(name, 0.0) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


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


def score_to_label(score: float) -> str:
    if score < 0.33:
        return "Lower"
    if score < 0.67:
        return "Moderate"
    return "Higher"


def predict_probability(model: Any, input_frame: pd.DataFrame) -> float:
    base_model = unwrap_model(model)
    probabilities = base_model.predict_proba(input_frame)
    return float(np.asarray(probabilities)[0, 1])


def make_probability_pair(probability: float) -> np.ndarray:
    probability = float(np.clip(probability, 1e-9, 1 - 1e-9))
    return np.array([[1.0 - probability, probability]], dtype=float)


def fit_venn_abers_calibrator(
    model: Any,
    calibration_frame: pd.DataFrame,
    calibration_labels: pd.Series | np.ndarray,
    precision: int | None = 4,
):
    calibration_probabilities = np.asarray(model.predict_proba(calibration_frame))[:, 1]
    calibration_pairs = np.column_stack([1.0 - calibration_probabilities, calibration_probabilities])
    calibrator = VennAbers()
    calibrator.fit(p_cal=calibration_pairs, y_cal=np.asarray(calibration_labels).astype(int), precision=precision)
    return calibrator


def predict_with_venn_abers(
    model: Any,
    input_frame: pd.DataFrame,
    calibrator: Any | None = None,
) -> PredictionResult:
    raw_probability = predict_probability(model, input_frame)

    # Prefer explicit external Venn-Abers calibrator when supplied.
    # This keeps uncertainty intervals available even if the wrapped model
    # uses a point calibrator (e.g., isotonic/platt) internally.
    if calibrator is not None:
        probability_pair = make_probability_pair(raw_probability)
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
            risk_label=score_to_label(calibrated_probability),
        )

    if isinstance(model, dict):
        method = str(model.get("calibration_method", "base"))
        wrapped_calibrator = model.get("calibrator")

        if method == "base" or wrapped_calibrator is None:
            calibrated_probability = raw_probability
            lower_bound = raw_probability
            upper_bound = raw_probability
        elif method in {"isotonic", "platt", "sigmoid"}:
            calibrated_probability = float(np.clip(wrapped_calibrator.predict(np.array([raw_probability]))[0], 0.0, 1.0))
            lower_bound = calibrated_probability
            upper_bound = calibrated_probability
        elif method == "venn_abers":
            probability_pair = make_probability_pair(raw_probability)
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
            risk_label=score_to_label(calibrated_probability),
        )

    return PredictionResult(
        raw_probability=raw_probability,
        calibrated_probability=raw_probability,
        lower_bound=raw_probability,
        upper_bound=raw_probability,
        uncertainty_width=0.0,
        risk_label=score_to_label(raw_probability),
    )


def feature_default(feature_name: str) -> float:
    return float(NUMERIC_DEFAULTS.get(feature_name, 0.0))


def feature_range(feature_name: str) -> tuple[float, float, float]:
    if feature_name in RANGE_HINTS:
        return RANGE_HINTS[feature_name]
    if feature_name.startswith("epwt_fg") or feature_name.startswith("fg"):
        return (0.0, 500.0, 1.0)  # safe fallback for unmapped food groups
    if feature_name.startswith("Total_"):
        return (0.0, 5000.0, 1.0)
    if feature_name.endswith("_nan"):
        return (0.0, 1.0, 1.0)
    return (0.0, 100.0, 1.0)


def is_binary_encoded_feature(feature_name: str) -> bool:
    return feature_name.startswith("alcohol_level_") or feature_name.startswith("smoking_level_")


def group_feature_names(feature_names: list[str]) -> dict[str, list[str]]:
    grouped = {
        "Core clinical": [],
        "Anthropometrics": [],
        "Dietary pattern": [],
        "Lifestyle encodings": [],
        "Other model inputs": [],
    }

    for feature_name in feature_names:
        if is_binary_encoded_feature(feature_name):
            grouped["Lifestyle encodings"].append(feature_name)
        elif feature_name in {"pa_met", "fbs", "chol", "tri", "hdl", "ldl", "ethnicity"}:
            grouped["Core clinical"].append(feature_name)
        elif feature_name in {"waist", "hip", "BMI"}:
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
        if _is_engineered_one_hot_feature(feature_name):
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

    if np.isnan(numeric):
        return None
    if numeric in {9, 99, 888888, 999999}:
        return None
    return numeric


def _clip_level(level: float | None) -> float | None:
    if level is None:
        return None
    return float(np.clip(level, 0.0, 3.0))


def _compute_smoking_level(raw_values: dict[str, Any]) -> float | None:
    direct = _to_numeric_clean_scalar(raw_values.get("smoking_level"))
    if direct is not None:
        return _clip_level(direct)

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
        return _clip_level(level)

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
        return _clip_level(level)

    if ever_smk is not None:
        return 0.0 if ever_smk == 0 else 1.0

    return None


def _compute_alcohol_level(raw_values: dict[str, Any]) -> float | None:
    direct = _to_numeric_clean_scalar(raw_values.get("alcohol_level"))
    if direct is not None:
        return _clip_level(direct)

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
        return _clip_level(level)

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
        return _clip_level(level)
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
    if not np.isfinite(bmi):
        return None
    return float(bmi)


def _compute_whr(raw_values: dict[str, Any]) -> float | None:
    waist = _to_numeric_clean_scalar(raw_values.get("waist"))
    hip = _to_numeric_clean_scalar(raw_values.get("hip"))
    if waist is None or hip is None or hip == 0:
        return None

    whr = waist / hip
    if not np.isfinite(whr):
        return None
    return float(whr)


def _format_level_suffix(level: float | None) -> str:
    if level is None:
        return "nan"
    return f"{float(level):.1f}"


def _is_engineered_one_hot_feature(feature_name: str) -> bool:
    return feature_name.startswith("alcohol_level_") or feature_name.startswith("smoking_level_")


def _populate_one_hot_group(values: dict[str, float], feature_names: list[str], group_name: str, level: float | None):
    group_prefix = f"{group_name}_"
    model_suffixes = [name[len(group_prefix) :] for name in feature_names if name.startswith(group_prefix)]
    suffixes = model_suffixes if model_suffixes else ONE_HOT_GROUPS[group_name]
    selected_suffix = _format_level_suffix(level)

    if selected_suffix not in suffixes and "nan" in suffixes:
        selected_suffix = "nan"

    for suffix in suffixes:
        values[f"{group_name}_{suffix}"] = 1.0 if suffix == selected_suffix else 0.0