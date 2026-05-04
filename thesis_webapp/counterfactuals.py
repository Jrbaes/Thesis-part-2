from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend import (
    RANGE_HINTS,
    build_input_values_from_widgets,
    feature_range,
    make_input_frame,
    predict_with_venn_abers,
    prepare_model_input,
)


NON_ACTIONABLE_FEATURES = {"age", "sex", "ethnicity", "ethnicity_group"}
REDUCE_ONLY_FEATURES = {"BMI", "bmi", "waist", "hip"}


def compute_counterfactuals(
    model: Any,
    preprocessor: Any,
    calibrator: Any,
    all_widget_values: dict,
    input_feature_names: list[str],
    dictionary_labels: dict[str, str],
    current_probability: float,
    top_n: int = 8,
) -> pd.DataFrame:
    from scipy.optimize import minimize_scalar  # type: ignore

    LAMBDA = 0.5
    Y_TARGET = 0.49

    scan_features = [fname for fname in all_widget_values if fname in RANGE_HINTS and fname not in NON_ACTIONABLE_FEATURES and not fname.startswith("epwt_fg") and not fname.startswith("fg")]

    def _predict_for_value(fname: str, val: float) -> float:
        test_widget = dict(all_widget_values)
        test_widget[fname] = val
        try:
            rebuilt = build_input_values_from_widgets(input_feature_names, test_widget)
            frame = make_input_frame(input_feature_names, rebuilt)
            model_frame = prepare_model_input(frame, preprocessor)
            pred_result = predict_with_venn_abers(model, model_frame, calibrator)
            return pred_result.calibrated_probability
        except Exception:
            return current_probability

    rows = []
    for fname in scan_features:
        current_val = all_widget_values.get(fname)
        if current_val is None or (isinstance(current_val, float) and np.isnan(current_val)):
            continue
        current_val = float(current_val)

        minimum, maximum, _ = feature_range(fname)
        base_range_width = max(maximum - minimum, 1e-6)
        if fname in REDUCE_ONLY_FEATURES:
            maximum = current_val
        if minimum >= maximum:
            continue

        feature_range_width = base_range_width

        def _wachter_loss(val: float, _fname=fname, _cur=current_val, _rng=feature_range_width) -> float:
            prob = _predict_for_value(_fname, val)
            pred_loss = (prob - Y_TARGET) ** 2
            proximity = ((val - _cur) / _rng) ** 2
            return LAMBDA * pred_loss + proximity

        try:
            opt = minimize_scalar(
                _wachter_loss,
                bounds=(minimum, maximum),
                method="bounded",
                options={"xatol": 1e-3, "maxiter": 200},
            )
            cf_val = float(np.clip(opt.x, minimum, maximum))
        except Exception:
            continue

        cf_prob = _predict_for_value(fname, cf_val)
        reduction = (current_probability - cf_prob) * 100.0

        if reduction <= 0.05 or np.isclose(cf_val, current_val, atol=1e-3):
            continue

        rows.append({"Feature": dictionary_labels.get(fname, fname), "Current Value": round(current_val, 2), "Suggested Value": round(cf_val, 2), "Current Risk": f"{current_probability * 100:.1f}%", "Projected Risk": f"{cf_prob * 100:.1f}%", "Risk Reduction": f"{reduction:.1f}%", "_delta": reduction})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("_delta", ascending=False).drop(columns=["_delta"]).head(top_n).reset_index(drop=True)
    return df
