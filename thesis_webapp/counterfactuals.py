from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app_constants import DISPLAY_LABEL_OVERRIDES
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


def _display_label(fname: str, dictionary_labels: dict[str, str]) -> str:
    """Return a clean display label — same priority order as the form UI."""
    if fname in DISPLAY_LABEL_OVERRIDES:
        return DISPLAY_LABEL_OVERRIDES[fname]
    raw = dictionary_labels.get(fname, "")
    if raw:
        return raw.split(":", 1)[0].strip().replace("  ", " ")
    return fname.replace("_", " ").strip().title()


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
    Y_TARGET = 0.34
    DECISION_BOUNDARY = 0.35
    GRID_STEPS = 60  # fallback grid resolution when Wachter can't cross 0.35

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

        # ── Stage 1: Wachter minimisation — minimal change to cross 0.5 ──
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

        # ── Stage 2: if Wachter didn't cross 0.5, grid-search for the
        #    value that achieves the greatest absolute risk reduction ──
        if cf_prob >= DECISION_BOUNDARY:
            grid = np.linspace(minimum, maximum, GRID_STEPS)
            grid_probs = np.array([_predict_for_value(fname, v) for v in grid])
            best_idx = int(np.argmin(grid_probs))
            grid_best_val = float(grid[best_idx])
            grid_best_prob = float(grid_probs[best_idx])
            if grid_best_prob < cf_prob:
                cf_val = grid_best_val
                cf_prob = grid_best_prob

        reduction = (current_probability - cf_prob) * 100.0

        if reduction <= 0.05 or np.isclose(cf_val, current_val, atol=1e-3):
            continue

        rows.append({
            "Feature": _display_label(fname, dictionary_labels),
            "Current Value": round(current_val, 2),
            "Suggested Value": round(cf_val, 2),
            "Current Risk": f"{current_probability * 100:.1f}%",
            "Projected Risk": f"{cf_prob * 100:.1f}%",
            "Risk Reduction": f"{reduction:.1f}%",
            "_delta": reduction,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("_delta", ascending=False).drop(columns=["_delta"]).head(top_n).reset_index(drop=True)
    return df


def compute_counterfactuals_safe(
    model: Any,
    preprocessor: Any,
    calibrator: Any,
    all_widget_values: dict,
    input_feature_names: list[str],
    dictionary_labels: dict[str, str],
    current_probability: float,
    top_n: int = 8,
    timeout: int = 180,
) -> pd.DataFrame:
    """Run compute_counterfactuals in an isolated subprocess.

    Prevents XGBoost's OpenMP thread pool from being initialised in the
    Streamlit main process, which would cause os._exit() on Windows when
    Streamlit's own threads try to run concurrently.

    Falls back to in-process computation if the model cannot be pickled.
    """
    import pathlib
    import pickle
    import subprocess
    import sys
    import tempfile
    import uuid

    _dir = pathlib.Path(__file__).parent
    worker = _dir / "_cf_worker.py"

    # Verify everything is picklable before spawning
    payload = {
        "model": model,
        "preprocessor": preprocessor,
        "calibrator": calibrator,
        "all_widget_values": all_widget_values,
        "input_feature_names": input_feature_names,
        "dictionary_labels": dictionary_labels,
        "current_probability": current_probability,
        "top_n": top_n,
    }
    try:
        _bytes = pickle.dumps(payload)
        del _bytes
    except Exception:
        # Not picklable — run in-process as fallback
        return compute_counterfactuals(
            model=model,
            preprocessor=preprocessor,
            calibrator=calibrator,
            all_widget_values=all_widget_values,
            input_feature_names=input_feature_names,
            dictionary_labels=dictionary_labels,
            current_probability=current_probability,
            top_n=top_n,
        )

    uid = uuid.uuid4().hex
    tmp_dir = pathlib.Path(tempfile.gettempdir())
    tmp_in = tmp_dir / f"_cf_in_{uid}.pkl"
    tmp_out = tmp_dir / f"_cf_out_{uid}.pkl"

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
            return pd.DataFrame()

        if not tmp_out.exists():
            return pd.DataFrame()

        with open(tmp_out, "rb") as fh:
            result = pickle.load(fh)

        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

    finally:
        for f in (tmp_in, tmp_out):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

