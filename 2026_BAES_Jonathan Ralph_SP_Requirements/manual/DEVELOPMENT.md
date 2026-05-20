# Development Guide

This document explains how to build and run the **Hypertension Risk Assessment** web application using Docker.

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Docker | 24+ |
| Git | any |

---

## Repository Layout

```
├── container/
│   └── Dockerfile           # Docker build definition
├── src/
│   ├── requirements.txt
│   └── thesis_webapp/
│       ├── app.py               # Streamlit entry point
│       ├── backend.py           # Model loading, inference, preprocessing
│       ├── app_constants.py     # Feature metadata (labels, units, overrides)
│       ├── counterfactuals.py   # CARLA-style counterfactual engine
│       ├── explainability.py    # SHAP / LIME computation
│       └── styles.py            # Global CSS applied to the Streamlit app
```

Training notebooks and saved model bundles live under `src/training/`.

---

## Build

Run the following from the **repository root** (the `Dockerfile` is inside `container/` and uses `../src/` as its copy source):

```bash
docker build -f container/Dockerfile -t hypertension-app .
```

The image uses `python:3.11-slim`, installs the system libraries required by LightGBM/XGBoost/OpenMP, and sets the Streamlit entry point to `thesis_webapp/app.py`.

---

## Run

```bash
docker run -p 8501:8501 hypertension-app
```

Open **http://localhost:8501** in your browser.

---

## Model Bundles

The application expects a `best_model_bundle.joblib` saved by the training notebooks. Each bundle is a `joblib`-serialised dict containing:

| Key | Contents |
|-----|----------|
| `model` | Fitted scikit-learn `Pipeline` (preprocessor + classifier) |
| `calibrator` | Fitted `VennAbers` calibrator |
| `preprocessor` | Fitted `ColumnTransformer` (also embedded in the pipeline) |

The bundle search order (highest `combined` metric wins) is:

1. `src/training/exp3_knn_rf/models/best_model_bundle.joblib`
2. `src/training/exp3_xgb_ada/models/best_model_bundle.joblib`
3. `src/training/exp3_logreg_cat/models/best_model_bundle.joblib`
4. `src/training/exp3_naive_bayes/models/best_model_bundle.joblib`

The production-pinned fallback is the **EXP B XGBoost** bundle (`exp3_xgb_ada`).

---

## Re-running Training Experiments

Open the notebooks in `src/training/` sequentially:

| Notebook | Purpose |
|----------|---------|
| `EXP 0 Create Merge Dataset.ipynb` | Merge Clinical, Dietary, Anthropometric datasets |
| `EXP A KNN_RF.ipynb` | K-NN and Random Forest experiments |
| `EXP B XGB_AdaBoost.ipynb` | XGBoost and AdaBoost experiments |
| `EXP C LogReg_CatBoost_LightGBM.ipynb` | Logistic Regression, CatBoost, LightGBM |
| `EXP D_Naive Bayes.ipynb` | Gaussian and Bernoulli Naive Bayes |
| `EXP E EDA.ipynb` | Exploratory data analysis |
| `EXP F COMPILE_RESULTS.ipynb` | Compile and compare all results |
| `EXP G sampling exp.ipynb` | Sampling strategy experiments |
| `EXP H threshold exp.ipynb` | Decision threshold experiments |

Each experiment notebook saves its model bundle and calibration CSV to the corresponding `exp3_*/` directory.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `COPY failed: file not found` during build | Ensure you run `docker build` from the **repository root**, not from inside `container/`. |
| Streamlit shows "No model bundle found" | Ensure a `best_model_bundle.joblib` exists in one of the `exp3_*/models/` paths under `src/training/`. |
| Port 8501 already in use | Run `docker run -p 8502:8501 hypertension-app` and open http://localhost:8502. |
