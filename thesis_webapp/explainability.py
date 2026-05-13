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


def _build_full_frame_background(base_input_frame: pd.DataFrame, rows: int = 20) -> pd.DataFrame:
    """Build a noisy background dataset in the full model-feature space."""
    rng = np.random.default_rng(42)
    base = base_input_frame.iloc[[0]].copy()
    records = []
    for _ in range(rows):
        row = base.copy()
        for col in base.columns:
            val = float(base.iloc[0][col])
            spread = max(abs(val) * 0.10, 0.05)
            row[col] = float(np.clip(rng.normal(val, spread), val - 3 * spread, val + 3 * spread))
        records.append(row)
    return pd.concat(records, ignore_index=True)


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

        # --- Try TreeExplainer first (XGBoost, LightGBM, CatBoost, RF, etc.) -----------
        # TreeExplainer reads tree structure directly — zero prediction calls,
        # no OpenMP thread-pool activation, no os._exit() risk, ~100x faster.
        _tree_ok = False
        try:
            _tree_exp = shap.TreeExplainer(model)
            _tree_ok = True
        except Exception:
            pass

        if _tree_ok:
            # Local: SHAP for the single test instance (use full model frame)
            _local_sv = np.asarray(_tree_exp.shap_values(base_input_frame))
            # Binary classifiers may return shape (2, n_samples, n_feat) or (n_samples, n_feat)
            if _local_sv.ndim == 3:
                _local_sv = _local_sv[1]          # class-1 slice
            _local_flat = _local_sv.reshape(-1)   # (n_all_features,)

            _all_cols = list(base_input_frame.columns)
            _col_idx = {c: i for i, c in enumerate(_all_cols)}
            _local_vals = np.array([
                _local_flat[_col_idx[f]] if f in _col_idx else 0.0
                for f in feature_names
            ])
            local_df = pd.DataFrame({
                "feature": feature_names,
                "shap_value": _local_vals,
                "abs_shap": np.abs(_local_vals),
            }).sort_values("abs_shap", ascending=False)

            # Global: mean |SHAP| over background rows
            _bg = _build_full_frame_background(base_input_frame, rows=20)
            _global_sv = np.asarray(_tree_exp.shap_values(_bg))
            if _global_sv.ndim == 3:
                _global_sv = _global_sv[1]
            _global_importance_full = np.mean(np.abs(_global_sv), axis=0)
            _global_vals = np.array([
                _global_importance_full[_col_idx[f]] if f in _col_idx else 0.0
                for f in feature_names
            ])
            global_df = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": _global_vals,
            }).sort_values("mean_abs_shap", ascending=False)

            return local_df, global_df, None

        # --- KernelExplainer fallback (non-tree models) --------------------------------
        background = _build_subset_background_samples(feature_names, base_input_frame, rows=20)
        predict_fn = _prediction_fn_for_subset_explainability(model, feature_names, base_input_frame)
        explainer = shap.KernelExplainer(predict_fn, background.values)

        local_shap_values = explainer.shap_values(input_frame.values, nsamples=50, silent=True)
        local_values = np.asarray(local_shap_values).reshape(-1)
        local_df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": local_values,
                "abs_shap": np.abs(local_values),
            }
        ).sort_values("abs_shap", ascending=False)

        global_shap_values = explainer.shap_values(background.values, nsamples=20, silent=True)
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
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            return None, None, f"SHAP computation failed: sys.exit({exc.code})"
        if not isinstance(exc, Exception):
            raise  # Re-raise StopException, RerunException, KeyboardInterrupt
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

    # Build background in model-feature space (i.e. the same standardised/OHE
    # scale that explain_input_frame lives in).  Using raw feature ranges here
    # is wrong because the model sees z-scored values; perturbing around the
    # actual test instance creates a valid local neighbourhood for LIME.
    rng_bg = np.random.default_rng(42)
    base_vals = input_frame.values  # shape (1, n_explain_features)
    noise = rng_bg.normal(0.0, 0.35, size=(50, len(feature_names)))
    bg_array = np.clip(base_vals + noise, -10.0, 10.0)
    background = pd.DataFrame(bg_array, columns=feature_names).replace([np.inf, -np.inf], np.nan)
    for name in feature_names:
        background[name] = pd.to_numeric(background[name], errors="coerce").fillna(float(input_frame.iloc[0].get(name, 0.0)))

    # Use the same full-frame predict approach as SHAP so the model always
    # receives all expected columns (non-explain columns filled from base_input_frame).
    _subset_predict_fn = _prediction_fn_for_subset_explainability(model, feature_names, base_input_frame)

    # Detect XGBoost booster so we can predict via DMatrix (single-threaded,
    # no OpenMP thread-pool activation, safe inside Streamlit on Windows).
    _xgb_booster = None
    _xgb_col_order: list[str] | None = None
    try:
        if hasattr(model, "get_booster"):
            import xgboost as _xgb  # type: ignore
            _xgb_booster = model.get_booster()
            _xgb_booster.set_param("nthread", 1)
            # Booster expects columns in the order it was trained on
            _xgb_col_order = list(base_input_frame.columns)
    except Exception:
        _xgb_booster = None

    def _lime_predict(nd_array: np.ndarray) -> np.ndarray:
        # Reconstruct full-feature frame from LIME-perturbed subset columns
        subset_frame = pd.DataFrame(nd_array, columns=feature_names).replace([np.inf, -np.inf], np.nan)
        full_frame = _repeat_base_frame(base_input_frame, len(subset_frame))
        for fname in feature_names:
            full_frame[fname] = pd.to_numeric(subset_frame[fname], errors="coerce").fillna(
                float(base_input_frame.iloc[0].get(fname, 0.0))
            )
        if _xgb_booster is not None:
            try:
                import xgboost as _xgb  # type: ignore
                if _xgb_col_order is not None:
                    full_frame = full_frame.reindex(columns=_xgb_col_order, fill_value=0.0)
                dm = _xgb.DMatrix(full_frame)
                proba_1 = _xgb_booster.predict(dm)
                return np.column_stack([1.0 - proba_1, proba_1])
            except Exception:
                pass
        proba_1 = np.asarray(model.predict_proba(full_frame))[:, 1]
        return np.column_stack([1.0 - proba_1, proba_1])

    lime_attempts = [
        {"discretize_continuous": True, "num_features": min(12, len(feature_names)), "num_samples": 500},
        {"discretize_continuous": False, "num_features": min(10, len(feature_names)), "num_samples": 300},
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
            if selected_label is None:
                last_error = ValueError("LIME produced no label explanations")
                continue
            pairs = explanation.as_list(label=int(selected_label))
            if not pairs:
                last_error = ValueError("LIME explanation returned zero feature contributions")
                continue
            return pd.DataFrame(pairs, columns=["rule", "weight"]), None
        except BaseException as exc:
            if isinstance(exc, SystemExit):
                last_error = RuntimeError(f"sys.exit({exc.code})")
                continue
            if not isinstance(exc, Exception):
                raise  # Re-raise StopException, RerunException, KeyboardInterrupt
            last_error = exc

    return None, f"LIME computation failed: {last_error}"


def compute_explainability_safe(
    model: Any,
    feature_names: list[str],
    input_frame: pd.DataFrame,
    base_input_frame: pd.DataFrame,
    timeout: int = 120,
) -> tuple:
    """Run SHAP + LIME in an isolated subprocess.

    If the subprocess crashes for any reason (os._exit, GPU OOM, C-level
    fault, tqdm writing to a corrupted pipe), the calling Streamlit process
    is completely unaffected.  Results are exchanged via temp files so the
    Streamlit file-watcher never picks them up.

    Falls back to in-process computation if the model object cannot be
    pickled (required for subprocess transport).
    """
    import pickle
    import pathlib
    import subprocess
    import sys
    import tempfile
    import uuid

    _dir = pathlib.Path(__file__).parent
    worker = _dir / "_exp_worker.py"

    # ── 1. Verify model is picklable (required for subprocess transport) ──
    try:
        _bytes = pickle.dumps(model)
        del _bytes
    except Exception:
        # Model not picklable — run in-process (no crash isolation)
        shap_local, shap_global, shap_err = try_compute_shap(
            model, feature_names, input_frame, base_input_frame
        )
        lime_df, lime_err = try_compute_lime(
            model, feature_names, input_frame, base_input_frame
        )
        return shap_local, shap_global, shap_err, lime_df, lime_err

    # ── 2. Write payload to system temp dir (NOT the webapp dir, so the
    #       Streamlit watchdog never picks up the temp files) ──
    uid = uuid.uuid4().hex
    tmp_dir = pathlib.Path(tempfile.gettempdir())
    tmp_in = tmp_dir / f"_shap_in_{uid}.pkl"
    tmp_out = tmp_dir / f"_shap_out_{uid}.pkl"

    payload = {
        "model": model,
        "feature_names": list(feature_names),
        "input_frame": input_frame.to_dict("list"),
        "base_input_frame": base_input_frame.to_dict("list"),
    }

    try:
        with open(tmp_in, "wb") as fh:
            pickle.dump(payload, fh)

        # ── 3. Spawn worker; suppress all output so nothing corrupts pipes ──
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
            msg = f"Explainability timed out after {timeout}s"
            return None, None, msg, None, msg

        if not tmp_out.exists():
            msg = f"Explainability subprocess exited (code {returncode}) without writing results"
            return None, None, msg, None, msg

        # ── 4. Read results ──
        with open(tmp_out, "rb") as fh:
            result = pickle.load(fh)

        if not result.get("ok"):
            err = result.get("error", "Unknown subprocess error")
            return None, None, err, None, err

        return (
            result["shap_local"],
            result["shap_global"],
            result["shap_error"],
            result["lime_df"],
            result["lime_error"],
        )

    except Exception as exc:
        msg = f"Explainability error: {exc}"
        return None, None, msg, None, msg

    finally:
        for _p in (tmp_in, tmp_out):
            try:
                _p.unlink(missing_ok=True)
            except Exception:
                pass
