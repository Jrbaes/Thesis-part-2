# Hypertension Risk Assessment Using Machine Learning on Philippine Nutrition Survey Data

Author: Jonathan Ralph F. Baes  
Year: 2026  
Institution: University of the Philippines Manila

---

## Abstract

This project develops a web-based hypertension risk assessment tool using the 2015 Philippine National Nutrition Survey (NNS) datasets. The system integrates clinical, dietary, and anthropometric features to estimate an individual's probability of hypertension. The repository contains both:

1. A Streamlit deployment stack for end-user risk assessment.
2. A reproducible unified training pipeline that provides a consolidated execution path with notebook-faithful defaults.

Important: the unified pipeline does not replace or invalidate the original notebooks. `EXP 0` and `EXP A` through `EXP F` remain valid, runnable, and official experiment artifacts.

---

## Repository Structure

```text
attestation/
container/
journal article/
manual/
Proposal Docs/
SP Docs/
src/
    Dockerfile
    requirements.txt
    thesis_webapp/
        app.py
        backend.py
        app_constants.py
        counterfactuals.py
        explainability.py
        styles.py
    training/
        EXP 0 Create Merge Dataset.ipynb
        EXP A KNN_RF.ipynb
        EXP B XGB_AdaBoost.ipynb
        EXP C LogReg_CatBoost_LightGBM.ipynb
        EXP D_Naive Bayes.ipynb
        EXP E EDA.ipynb
        EXP F COMPILE_RESULTS.ipynb
        EXP unified_pipeline.py
README.md
```

---

## System Overview

The Streamlit web app accepts patient data across four domains:

| Domain | Examples |
|--------|----------|
| Demographic / Clinical | Age, sex, ethnicity, blood pressure history |
| Behavioral | Smoking status, alcohol consumption, binge drinking |
| Anthropometric | Weight, height, waist circumference, hip circumference |
| Dietary | Intake across 27 food groups (grams), derived energy and macronutrient totals |

The app then:

1. Preprocesses inputs through the saved preprocessing artifacts.
2. Produces calibrated hypertension probabilities.
3. Applies the selected operating threshold from training outputs.
4. Provides SHAP and LIME explainability outputs.
5. Generates counterfactual recommendations.

---

## Machine Learning Experiments

| Experiment | Models |
|------------|--------|
| EXP A | K-Nearest Neighbors, Random Forest |
| EXP B | XGBoost, AdaBoost |
| EXP C | Logistic Regression, CatBoost, LightGBM |
| EXP D | Naive Bayes |

All experiments use a two-stage search plus post-calibration evaluation flow.

---

## Unified Training Pipeline (Notebook Parity)

`EXP unified_pipeline.py` is the consolidated Python execution path for repeatable training runs.

It is intended to mirror notebook behavior while keeping the notebook workflow intact. `EXP 0` and `EXP A` through `EXP F` can still be run independently and are not deprecated.

Main script:

```bash
python "src/training/EXP unified_pipeline.py" --root src/training --out-dir exp3_unified
```

Default behavior is strict notebook-faithful execution:

1. Per-model Stage 1 and Stage 2 budgets come from the original EXP A-D configurations.
2. Threshold grid defaults to `np.round(np.arange(0.35, 0.70, 0.05), 2)`.
3. Search spaces match notebook search spaces for KNN, RF, XGBoost, AdaBoost, LogReg, CatBoost, LightGBM, and Naive Bayes.
4. Global stage CLI flags are optional overrides (`--s1-trials`, `--s1-folds`, `--s1-epochs`, `--s2-refine`, `--s2-folds`, `--s2-epochs`, `--top-k-s1`, `--top-k-s2`, `--final-epochs`). If omitted, notebook parity defaults are used.

Example override run:

```bash
python "src/training/EXP unified_pipeline.py" --root src/training --out-dir exp3_override --s1-trials 20 --s2-refine 3
```

Primary outputs (under `src/training/<out-dir>/`):

- `stage1_summary.csv`
- `stage2_summary.csv`
- `pre_calibration_results.csv`
- `post_calibration_results.csv`
- `models/best_model_bundle.joblib`
- `plots/` and explainability artifacts

---

## Dataset

- Source: 2015 Philippine National Nutrition Survey (NNS), Food and Nutrition Research Institute - DOST (FNRI-DOST)
- Domains: Clinical, Dietary, Anthropometric
- Features: Demographic, clinical, diet, body measurements, and behavior variables

---

## Technologies

| Category | Libraries / Tools |
|----------|-------------------|
| Web framework | Streamlit |
| ML / preprocessing | scikit-learn, XGBoost, CatBoost, LightGBM, imbalanced-learn |
| Calibration | venn-abers |
| Explainability | SHAP, LIME |
| Data | pandas, NumPy, SciPy |
| Container | Docker (Python 3.11-slim), port 8501 |

---

## Running the Application

See [manual/DEVELOPMENT.md](manual/DEVELOPMENT.md) for full instructions.

Quick start (Docker):

```bash
cd src
docker build -t hypertension-app .
docker run -p 8501:8501 hypertension-app
```

Open: `http://localhost:8501`

---

## Usage

See [manual/HELP.md](manual/HELP.md) for the user manual.

---

## Attestation

The signed ORI Attestation Form is in `attestation/`.

---

## License

For academic use only. Data obtained from FNRI-DOST under research data access terms.
