from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend import feature_default, feature_range


def _repeat_base_frame(base_input_frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    return pd.concat([base_input_frame.iloc[[0]].copy()] * rows, ignore_index=True)


def _prediction_fn_for_subset_explainability(
    model: Any,
    explain_feature_names: list[str],
    base_input_frame: pd.DataFrame,
):
    def _predict(nd_array: np.ndarray) -> np.ndarray:
        subset_frame = pd.DataFrame(nd_array, columns=explain_feature_names).replace([np.inf, -np.inf], np.nan)
        full_frame = _repeat_base_frame(base_input_frame, len(subset_frame))

        for feature_name in explain_feature_names:
            full_frame[feature_name] = pd.to_numeric(subset_frame[feature_name], errors="coerce").fillna(float(base_input_frame.iloc[0].get(feature_name, 0.0)))
        return np.asarray(model.predict_proba(full_frame))[:, 1]

    return _predict


def _build_subset_background_samples(
    feature_names: list[str],
    base_input_frame: pd.DataFrame,
    rows: int = 80,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    records: list[dict[str, float]] = []
    base_row = base_input_frame.iloc[0]

    for _ in range(rows):
        row: dict[str, float] = {}
        for feature_name in feature_names:
            base_value = float(base_row.get(feature_name, 0.0))

            if "__" in feature_name:
                spread = max(abs(base_value) * 0.08, 0.03)
                row[feature_name] = float(rng.normal(loc=base_value, scale=spread))
                continue

            minimum, maximum, _ = feature_range(feature_name)
            default_value = feature_default(feature_name)

            if maximum <= minimum:
                row[feature_name] = float(default_value)
                continue

            spread = (maximum - minimum) * 0.08
            sampled = float(rng.normal(loc=default_value, scale=max(spread, 1e-6)))
            row[feature_name] = float(np.clip(sampled, minimum, maximum))

        records.append(row)

    return pd.DataFrame(records, columns=feature_names)


def try_compute_shap(
    model: Any,
    feature_names: list[str],
    input_frame: pd.DataFrame,
    base_input_frame: pd.DataFrame | None = None,
):
    try:
        import shap  # type: ignore
    except Exception:
        return None, None, "SHAP package is not installed. Install with: pip install shap"

    try:
        if base_input_frame is None:
            base_input_frame = input_frame

        background = _build_subset_background_samples(feature_names, base_input_frame, rows=40)
        predict_fn = _prediction_fn_for_subset_explainability(model, feature_names, base_input_frame)
        explainer = shap.KernelExplainer(predict_fn, background.values)

        local_shap_values = explainer.shap_values(input_frame.values, nsamples=100)
        local_values = np.asarray(local_shap_values).reshape(-1)
        local_df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": local_values,
                "abs_shap": np.abs(local_values),
            }
        ).sort_values("abs_shap", ascending=False)

        global_shap_values = explainer.shap_values(background.values, nsamples=40)
        global_array = np.asarray(global_shap_values)
        if global_array.ndim == 1:
            global_array = global_array.reshape(1, -1)
        global_importance = np.mean(np.abs(global_array), axis=0)
        global_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": global_importance,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        return local_df, global_df, None
    except Exception as exc:
        return None, None, f"SHAP computation failed: {exc}"


def try_compute_lime(
    model: Any,
    feature_names: list[str],
    input_frame: pd.DataFrame,
    base_input_frame: pd.DataFrame | None = None,
):
    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    except Exception:
        return None, "LIME package is not installed. Install with: pip install lime"

    if base_input_frame is None:
        base_input_frame = input_frame

    background = _build_subset_background_samples(feature_names, base_input_frame, rows=120).replace([np.inf, -np.inf], np.nan)
    for name in feature_names:
        background[name] = pd.to_numeric(background[name], errors="coerce").fillna(float(base_input_frame.iloc[0].get(name, 0.0)))

    def _lime_predict(nd_array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(nd_array, columns=feature_names).replace([np.inf, -np.inf], np.nan)
        full_frame = _repeat_base_frame(base_input_frame, len(frame))
        for name in feature_names:
            full_frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(float(base_input_frame.iloc[0].get(name, 0.0)))
        return np.asarray(model.predict_proba(full_frame))

    lime_attempts = [
        {"discretize_continuous": True, "num_features": min(12, len(feature_names)), "num_samples": 3000},
        {"discretize_continuous": False, "num_features": min(10, len(feature_names)), "num_samples": 2000},
    ]

    last_error: Exception | None = None
    for attempt in lime_attempts:
        try:
            explainer = LimeTabularExplainer(
                training_data=background.values,
                feature_names=feature_names,
                class_names=["No HTN", "HTN"],
                mode="classification",
                discretize_continuous=attempt["discretize_continuous"],
                random_state=42,
            )

            explanation = explainer.explain_instance(
                data_row=input_frame.iloc[0].values,
                predict_fn=_lime_predict,
                num_features=int(attempt["num_features"]),
                num_samples=int(attempt["num_samples"]),
                top_labels=1,
            )

            available_labels = sorted(getattr(explanation, "local_exp", {}).keys())
            selected_label = 1 if 1 in available_labels else (available_labels[0] if available_labels else None)
            pairs = explanation.as_list(label=int(selected_label)) if selected_label is not None else []
            return (pd.DataFrame(pairs, columns=["rule", "weight"]) if pairs else pd.DataFrame(columns=["rule", "weight"])), None
        except Exception as exc:
            last_error = exc

    return None, f"LIME computation failed: {last_error}"
