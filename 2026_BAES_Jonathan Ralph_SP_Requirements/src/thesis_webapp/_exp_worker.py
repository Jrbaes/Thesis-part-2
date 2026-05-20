"""Standalone SHAP/LIME worker — always run as __main__ in a subprocess.

Any crash (os._exit, GPU OOM, C-level fault, tqdm stdout corruption) stays
contained here and never reaches the Streamlit server process.
"""
import sys
import pickle
import warnings

# Silence all warnings so nothing writes to stdout/stderr and corrupts pipes.
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    result: dict = {}

    try:
        import pandas as pd  # noqa: PLC0415
        from explainability import try_compute_shap, try_compute_lime  # noqa: PLC0415

        with open(in_path, "rb") as fh:
            payload = pickle.load(fh)

        model = payload["model"]
        feature_names: list = payload["feature_names"]
        input_frame = pd.DataFrame(payload["input_frame"])
        base_input_frame = pd.DataFrame(payload["base_input_frame"])

        shap_local_df, shap_global_df, shap_error = try_compute_shap(
            model, feature_names, input_frame, base_input_frame
        )
        lime_df, lime_error = try_compute_lime(
            model, feature_names, input_frame, base_input_frame
        )

        result = {
            "ok": True,
            "shap_local": shap_local_df,
            "shap_global": shap_global_df,
            "shap_error": shap_error,
            "lime_df": lime_df,
            "lime_error": lime_error,
        }

    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "tb": traceback.format_exc(),
        }

    try:
        with open(out_path, "wb") as fh:
            pickle.dump(result, fh)
    except Exception:  # noqa: BLE001
        pass  # Nothing more we can do
