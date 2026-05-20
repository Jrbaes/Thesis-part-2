# User Manual — Hypertension Risk Assessment

This manual explains how to use the **Hypertension Risk Assessment** web application.

---

## Overview

The application accepts patient data and predicts the probability of hypertension using a machine learning model trained on the 2015 Philippine National Nutrition Survey. The tool also explains its prediction and suggests lifestyle changes that could reduce risk.

> **Medical Disclaimer:** This tool is for research and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding any medical condition.

---

## Accessing the Application

| Method | URL |
|--------|-----|
| Docker (local) | http://localhost:8501 |
| Local development server | http://localhost:8501 |

If the application is hosted on a remote server, use the URL provided by your administrator.

---

## Input Form

The form is divided into four sections. Fields marked **missing** accept blank input when data is unavailable; the model handles missing values internally.

### 1. Demographics

| Field | Description | Notes |
|-------|-------------|-------|
| Age | Exact age as of last birthday (years) | Integer, 0–120 |
| Sex | Sex of the respondent | 1 = Male, 2 = Female |
| Ethnicity | Ethnicity code | 0 = Not IP / no foreign blood; 1 = Indigenous People; 2 = ⅔ Filipino; 3 = ½ foreign blood |

### 2. Behavioral / Lifestyle

Smoking and alcohol fields are **cascaded**: selecting a top-level status (e.g., *Never smoker*) automatically fills in the dependent sub-fields and locks them.

**Smoking**

| Field | Options |
|-------|---------|
| Smoking Status | 0 Never, 1 Current, 2 Former |
| Current Smoking Frequency | Auto-filled when status = Never or Former |
| Ever Smoked | Auto-filled when status = Current |

**Alcohol**

| Field | Options |
|-------|---------|
| Alcohol Status | 0 Never, 1 Current, 2 Former |
| Ever consumed alcohol | Auto-filled based on status |
| Current drinker | Auto-filled based on status |
| Drank in past 30 days | Enabled only for current drinkers |
| Binge Drinking | Female ≥ 4 / Male ≥ 5 standard drinks in a row (past 30 days) |

### 3. Anthropometric Measurements

| Field | Unit | Typical range |
|-------|------|---------------|
| Weight | kg | 20–200 |
| Height | cm | 50–260 |
| Waist Circumference | cm | 40–200 |
| Hip Circumference | cm | 40–200 |

BMI and waist-to-hip ratio are computed automatically from these values.

### 4. Dietary Intake

Enter daily intake (in **grams**) for each of the 27 food groups. The following totals are computed automatically and are read-only:

- **Total Food Intake** — sum of all food group inputs
- **Total Energy (kcal)** — derived from carbohydrate, protein, and fat estimates
- **Total Protein (g)** — estimated from protein-rich food groups

You do not need to fill in all food groups; leave unknown fields blank.

---

## Running the Prediction

Click the **Predict** button at the bottom of the form.

The application will:
1. Validate all inputs (out-of-range entries are flagged with a warning; fix them before proceeding).
2. Compute derived features (BMI, food totals, engineered smoking/alcohol indicators).
3. Pass the data through the preprocessing pipeline and the calibrated model.
4. Display the results panel.

---

## Results Panel

### Risk Gauge

A circular gauge shows the predicted **hypertension probability** (0–100 %). The needle and number indicate the current score.

- **Green zone** (below 35 %): Not at Risk
- **Red zone** (35 % and above): At Risk of Hypertension

The black threshold marker is fixed at **35 %**.

### Risk Label

A clear text label — **"At Risk of Hypertension"** or **"Not at Risk"** — is shown alongside the gauge.

---

## Explainability

Below the gauge, three explainability tabs are available.

### SHAP — Local Explanation

A bar chart shows which features pushed the prediction **up** (towards risk) or **down** (away from risk) for this specific input. Longer bars indicate greater influence.

### LIME — Local Explanation

A table shows LIME feature attribution weights. Each row lists a feature rule, its contribution to the prediction, and the user's raw input value. Positive weights increase predicted risk; negative weights decrease it.

### SHAP — Global Feature Importance

A bar chart shows the average absolute SHAP value for each feature across a background sample, indicating which features the model relies on most overall.

---

## Counterfactual Suggestions

If the prediction is **At Risk**, a counterfactual panel recommends up to 8 actionable input changes that would individually reduce the predicted risk by the greatest amount.

The divergent bar chart shows:
- **Blue bars** — the feature should be *decreased* (e.g., reduce waist circumference)
- **Amber bars** — the feature should be *increased* (e.g., increase vegetable intake)

Each bar label shows the risk reduction and the suggested value range (e.g., `−12%  (88.0 → 78.0)`).

Non-actionable features (age, sex, ethnicity) are excluded from counterfactual suggestions.

---

## Tips

- For best accuracy, provide as many fields as possible, particularly weight, height, waist circumference, and dietary intake.
- After changing any input, click **Predict** again to refresh the results.
- All inputs are processed locally; no data is transmitted to external servers.
