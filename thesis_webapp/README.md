# Hypertension Risk Studio

Streamlit front end for the thesis model. It loads the stand-in joblib, exposes the model's feature schema, and scores risk with a Venn-Abers uncertainty band when a calibrator artifact is available.

## Run

```bash
streamlit run thesis_webapp/app.py
```

The default model path is `models/XGBoost__baseline_rank1.joblib`.
The default calibrator path is `main_2015_balanced_gpu_artifacts/models/venn_abers_calibrator.joblib`.

## Replace Later

Swap the model joblib and calibrator path in the sidebar, or overwrite those files with your final exported artifacts.