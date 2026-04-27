from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_MODEL_PATH,
    build_input_values_from_widgets,
    feature_default,
    feature_range,
    group_feature_names,
    load_calibrator,
    load_feature_names,
    load_model,
    make_input_frame,
    predict_with_venn_abers,
)


st.set_page_config(
    page_title="Hypertension Risk Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DICTIONARY_PATHS = [
    PROJECT_ROOT / "Datasets2015" / "Clinical" / "Jonathan Ralph_Baes_2026-03-26141903_data-dictionary_clinical.csv",
    PROJECT_ROOT / "Datasets2015" / "Dietary" / "Jonathan Ralph_Baes_2026-03-26141801_data-dictionary_dietary.csv",
    PROJECT_ROOT / "Datasets2015" / "Anthropometric" / "Jonathan Ralph_Baes_2026-03-26141834_data-dictionary_anthrop.csv",
]

FEATURE_DICTIONARY_ALIASES = {
    "Total_Ener": ["Total_Energy"],
    "Total_Prot": ["Total_Protein"],
    "Total_Calc": ["Total_Calcium"],
    "Total_VitA": ["Total_VitaminA"],
    "Total_VitC": ["Total_VitaminC"],
    "Total_Thia": ["Total_Thiamin"],
    "Total_Ribo": ["Total_Riboflavin"],
    "Total_Nia": ["Total_Niacin"],
    "Total_Food_epwt": ["Total_FoodIntake"],
}

VARIABLE_DEFINITION_OVERRIDES = {
    "hhnum": "Household Number (unique identifier; merging variable): unique code assigned to each sample household. Shared by members of the same household and used with region/province codes for merging components.",
    "member_code": "Member Code (unique identifier; merging variable): pre-coded household membership number.",
    "age": "Age of Respondent: exact age as of last birthday.",
    "sex": "Sex of Respondent: sex of household member.",
    "psc": "Physiological Status Code: current physiological status among female members (pregnancy/lactation categories; 99 indicates male member).",
    "csc": "Civil Status Code: civil status of the household member.",
    "Ave_SBP": "Average Systolic BP (mmHg): pressure during heart contraction phase (for 10 years old and above).",
    "Ave_DBP": "Average Diastolic BP (mmHg): pressure during heart relaxation phase between beats (for 10 years old and above).",
    "pa_met": "Physical activity (MET-based indicator): higher values indicate greater activity energy expenditure.",
    "fbs": "Fasting Blood Sugar (mg/dL): blood glucose after fasting.",
    "chol": "Total Cholesterol (mg/dL): blood cholesterol concentration.",
    "tri": "Triglycerides (mg/dL): blood triglyceride concentration.",
    "hdl": "High-Density Lipoprotein (mg/dL): often referred to as protective cholesterol.",
    "ldl": "Low-Density Lipoprotein (mg/dL): often referred to as atherogenic cholesterol.",
    "ethnicity": "Ethnicity code: 0 Not IP/without foreign blood, 1 Indigenous People, 2 2/3 Filipino, 3 with 1/2 foreign blood.",
    "current_smoking": "Presently smoke cigarettes/cigars/pipes/tobacco products: current smoking frequency code.",
    "ever_smk": "Ever smoked in the past: former smoking behavior code.",
    "alcohol": "Ever consumed alcoholic drink such as beer, wine, or spirits.",
    "con_alcohol": "Consumed alcoholic drink within the past 12 months (current drinkers).",
    "drnk_30days": "Consumed alcoholic drink within the past 30 days.",
    "drnk_30d_num": "Total number of standard drinks in one single occasion in the past 30 days.",
    "smoke_status": "Smoking Status (Generated): 0 Never, 1 Current, 2 Former, 9 Not Applicable.",
    "alcohol_status": "Alcohol Status (Generated): 0 Never, 1 Current, 2 Former, 9 Not Applicable.",
    "binge_drink": "Binge Drinking Status (Generated): female >=4 standard drinks in a row; male >=5, among those who drank in past 30 days.",
    "weight": "Ave Weight (kg): body heaviness from muscle, fat, bone, organs and related conditions.",
    "height": "Ave Height (cm): standing height (or recumbent length for very young children in source survey).",
    "waist": "Ave Waist Circumference (cm): perimeter around natural waist/abdomen.",
    "hip": "Ave Hip Circumference (cm): distance around largest hip/buttocks area.",
    "measure_remarks": "Remarks in Measure: subjective notes to validate unusual measurement values.",
    "anthro_group": "Anthrop Group: classification by age and physiological status.",
    "mos_lactation": "Months of Lactation: duration of lactation in months.",
    "mos_preg": "Months of Pregnancy: pregnancy period in months.",
    "fg1": "Cereals and Cereal Products (g): includes rice/corn/other cereals.",
    "fg2": "Rice and Rice Products (g): rice and rice-based products.",
    "fg3": "Corn and Corn Products (g): corn and corn-based products.",
    "fg4": "Other Cereal Products (g): breads, biscuits, cakes, noodles, flour, etc.",
    "fg5": "Starchy Roots and Tubers (g): sweet potato, potato, cassava, and similar roots/tubers.",
    "fg6": "Sugar and Syrups (g): sugars, syrups, sweets, sweetened drinks and sugary foods.",
    "fg7": "Dried Beans, Nuts and Seeds (g): legumes, nuts, seeds and related products.",
    "fg8": "Vegetables (g): aggregate of green/yellow and other vegetables.",
    "fg9": "Green Leafy and Yellow Vegetables (g).",
    "fg10": "Other Vegetables (g).",
    "fg11": "Fruits (g): includes vitamin C-rich and other fruits.",
    "fg12": "Vitamin C-Rich Fruits (g).",
    "fg13": "Other Fruits (g).",
    "fg14": "Fish, Meat and Poultry (g).",
    "fg15": "Fish and Fish Products (g): fresh/dried/processed fish plus crustaceans/mollusks.",
    "fg16": "Meat and Meat Products (g): fresh, organ, and processed meats.",
    "fg17": "Poultry (g).",
    "fg18": "Eggs (g).",
    "fg19": "Milk and Milk Products (g).",
    "fg20": "Whole Milk (g): fresh/evaporated/recombined/powdered/condensed milk.",
    "fg21": "Milk Products (g): cheese, yogurt, cultured milk, ice cream, etc.",
    "fg23": "Fats and Oils (g): cooking oil, coconut fat, lard, butter, margarine, etc.",
    "fg24": "Miscellaneous (g): beverages, condiments/spices, and other misc.",
    "fg25": "Beverages (g): coffee, tea, alcoholic beverages, cacao/chocolate drinks, flavored drinks.",
    "fg26": "Condiments and Spices (g): salt, vinegar, catsup, and seasonings.",
    "fg27": "Other Miscellaneous (g): herbs and other miscellaneous ingredients.",
    "Total_FoodIntake": "Total Food Intake (g): total household intake across 27 food groups.",
    "Total_Energy": "Total Energy (kcal): total household energy intake.",
    "Total_Protein": "Total Protein (g): total household protein intake.",
    "Total_Calcium": "Total Calcium (mg): total household calcium intake.",
    "Total_Iron": "Total Iron (mg): total household iron intake.",
    "Total_VitaminA": "Total Vitamin A (mcg RE): total household vitamin A intake.",
    "Total_Thiamin": "Total Thiamin (mg): total household thiamin intake.",
    "Total_Riboflavin": "Total Riboflavin (mg): total household riboflavin intake.",
    "Total_Niacin": "Total Niacin (mg): total household niacin intake.",
    "Total_VitaminC": "Total Vitamin C (mg): total household vitamin C intake.",
    "Total_CHO": "Total Carbohydrates (g): total household carbohydrate intake.",
    "Total_Fat": "Total Fats (g): total household fat intake.",
}

VALUE_LABEL_OVERRIDES = {
    "sex": {"1": "Male", "2": "Female"},
    "ethnicity": {
        "0": "No, Not an IP/Without Foreign Blood",
        "1": "Yes, Indigenous People",
        "2": "Yes, 2/3 Filipino",
        "3": "Yes, with 1/2 Foreign Blood",
    },
    "current_smoking": {
        "0": "No, not at all",
        "1": "Yes, once a week",
        "2": "Yes, 2-6 times a week",
        "3": "Yes, every day, 7 times a week",
        "888888": "Not Applicable",
    },
    "ever_smk": {
        "0": "No, not at all",
        "1": "Yes, once a week",
        "2": "Yes, 2-6 times a week",
        "3": "Yes, every day",
        "4": "Yes, tried once",
        "5": "Yes, occasionally",
        "888888": "Not Applicable",
    },
    "alcohol": {
        "0": "No",
        "1": "Yes",
        "2": "Yes, occasionally, during socials",
        "888888": "Not Applicable",
    },
    "con_alcohol": {"0": "No", "1": "Yes", "999999": "Not Applicable"},
    "drnk_30days": {"0": "No", "1": "Yes", "999999": "Not Applicable"},
    "smoke_status": {"0": "Never", "1": "Current", "2": "Former", "9": "Not Applicable"},
    "alcohol_status": {"0": "Never", "1": "Current", "2": "Former", "9": "Not Applicable"},
    "binge_drink": {"0": "Non-binge drinker", "1": "Binge drinker", "99": "Not Applicable"},
    "anthro_group": {
        "1": "0-60 Months Old",
        "2": "61-120 Months Old",
        "3": "121-228 Months Old",
        "4": "Adults (19 Years Old and Above)",
        "7": "Pregnant Mothers",
        "8": "Lactating Women",
    },
    "mos_lactation": {"10": "Lactating, 0-6 Months", "11": "Lactating, 7-12 Months", "12": "Lactating, Over 1 Year"},
    "mos_preg": {
        "1": "Pregnant, 1 Month",
        "2": "Pregnant, 2 Months",
        "3": "Pregnant, 3 Months",
        "4": "Pregnant, 4 Months",
        "5": "Pregnant, 5 Months",
        "6": "Pregnant, 6 Months",
        "7": "Pregnant, 7 Months",
        "8": "Pregnant, 8 Months",
        "9": "Pregnant, 9 Months",
    },
    "psc": {
        "0": "Non-pregnant/Non-lactating female members",
        "1": "Pregnant, 1 month",
        "2": "Pregnant, 2 months",
        "3": "Pregnant, 3 months",
        "4": "Pregnant, 4 months",
        "5": "Pregnant, 5 months",
        "6": "Pregnant, 6 months",
        "7": "Pregnant, 7 months",
        "8": "Pregnant, 8 months",
        "9": "Pregnant, 9 months",
        "10": "Lactating, 0-6 months",
        "11": "Lactating, 7-12 months",
        "12": "Lactating, over 1 year",
        "99": "Male Member",
    },
}


st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(220, 38, 38, 0.12), transparent 32%),
          radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 28%),
          linear-gradient(180deg, #f8f5f2 0%, #f2ede8 45%, #fcfbfa 100%);
        color: #18212f;
      }
      .hero {
        padding: 1.1rem 1.4rem;
        border-radius: 28px;
        border: 1px solid rgba(24, 33, 47, 0.08);
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.92), rgba(127, 29, 29, 0.90));
        color: #fff;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.16);
      }
      .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: -0.03em;
      }
      .hero p {
        margin: 0.35rem 0 0;
        color: rgba(255, 255, 255, 0.84);
        max-width: 72ch;
      }
      .glass-card {
        padding: 1rem 1.1rem;
        border-radius: 22px;
        border: 1px solid rgba(24, 33, 47, 0.08);
        background: rgba(255, 255, 255, 0.76);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      }
      .kpi-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #64748b;
      }
      .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.15rem;
        color: #0f172a;
      }
      .kpi-subtitle {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: 0.18rem;
      }
      section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.82);
        border-right: 1px solid rgba(15, 23, 42, 0.08);
      }
      .block-container {
        padding-top: 1.1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, calibrator_path: str):
    model = load_model(Path(model_path))
    calibrator = load_calibrator(Path(calibrator_path))
    feature_names = load_feature_names(model)
    return model, calibrator, feature_names


@st.cache_data(show_spinner=False)
def load_dataset_dictionaries() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    labels: dict[str, str] = {}
    value_labels: dict[str, dict[str, str]] = {}

    for path in DICTIONARY_PATHS:
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        if not lines:
            continue

        for line in lines[1:]:
            if not line.strip():
                continue

            parts = line.split(",", 4)
            if len(parts) < 5:
                continue

            _, variable_name, variable_label, var_value, value_label = [part.strip() for part in parts]
            if not variable_name:
                continue

            labels.setdefault(variable_name, variable_label)

            if var_value:
                if variable_name not in value_labels:
                    value_labels[variable_name] = {}
                value_labels[variable_name].setdefault(var_value, value_label)

    return labels, value_labels


def dictionary_name_candidates(feature_name: str) -> list[str]:
    candidates = [feature_name]

    if feature_name.startswith("epwt_fg"):
        candidates.append(feature_name.replace("epwt_", ""))

    if feature_name in FEATURE_DICTIONARY_ALIASES:
        candidates.extend(FEATURE_DICTIONARY_ALIASES[feature_name])

    return candidates


def field_help_text(feature_name: str, dictionary_labels: dict[str, str]) -> str | None:
    for candidate in dictionary_name_candidates(feature_name):
        if candidate in VARIABLE_DEFINITION_OVERRIDES:
            return VARIABLE_DEFINITION_OVERRIDES[candidate]

    for candidate in dictionary_name_candidates(feature_name):
        if candidate in dictionary_labels and dictionary_labels[candidate]:
            return dictionary_labels[candidate]
    return None


def field_value_labels(feature_name: str, dictionary_value_labels: dict[str, dict[str, str]]) -> dict[str, str]:
    for candidate in dictionary_name_candidates(feature_name):
        if candidate in VALUE_LABEL_OVERRIDES:
            return VALUE_LABEL_OVERRIDES[candidate]

    for candidate in dictionary_name_candidates(feature_name):
        if candidate in dictionary_value_labels and dictionary_value_labels[candidate]:
            return dictionary_value_labels[candidate]

    return {}


def _sorted_value_label_keys(value_label_map: dict[str, str]) -> list[str]:
    def _key(value: str):
        try:
            return (0, int(float(value)))
        except ValueError:
            return (1, value)

    return sorted(value_label_map.keys(), key=_key)


def render_number_input(feature_name: str, dictionary_labels: dict[str, str], dictionary_value_labels: dict[str, dict[str, str]]):
    minimum, maximum, step = feature_range(feature_name)
    default_value = feature_default(feature_name)
    help_text = field_help_text(feature_name, dictionary_labels)

    value_label_map = field_value_labels(feature_name, dictionary_value_labels)
    if value_label_map and len(value_label_map) <= 30:
        option_values: list[str | None] = [None] + _sorted_value_label_keys(value_label_map)

        def _format_choice(choice: str | None) -> str:
            if choice is None:
                return "missing/use default"
            return f"{choice} - {value_label_map.get(choice, '')}".strip(" -")

        selected = st.selectbox(
            feature_name,
            options=option_values,
            index=0,
            key=f"input_{feature_name}",
            help=help_text,
            format_func=_format_choice,
        )
        if selected is None:
            return float(default_value)
        try:
            return float(int(float(selected)))
        except ValueError:
            return float(default_value)

    return st.number_input(
        feature_name,
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(default_value),
        step=float(step),
        format="%.2f",
        key=f"input_{feature_name}",
        help=help_text,
    )


def render_lifestyle_selectors(dictionary_labels: dict[str, str], dictionary_value_labels: dict[str, dict[str, str]]):
    st.caption("Provide original lifestyle variables. The backend computes engineered smoking/alcohol levels.")

    def _optional_code(variable_name: str, options: list[int], key: str, fallback_help: str):
        value_label_map = VALUE_LABEL_OVERRIDES.get(variable_name, dictionary_value_labels.get(variable_name, {}))
        variable_help = dictionary_labels.get(variable_name, fallback_help)

        option_values: list[int | None] = [None] + options

        def _format_choice(choice: int | None) -> str:
            if choice is None:
                return "missing"
            label = value_label_map.get(str(choice), "")
            if label:
                return f"{choice} - {label}"
            return str(choice)

        selected = st.selectbox(
            variable_name,
            options=option_values,
            index=0,
            key=key,
            format_func=_format_choice,
            help=variable_help,
        )
        return float("nan") if selected is None else float(int(selected))

    smoke_left, smoke_mid, smoke_right = st.columns(3)
    with smoke_left:
        smoke_status = _optional_code(
            variable_name="smoke_status",
            options=[0, 1, 2],
            key="raw_smoke_status",
            fallback_help="Smoking status code.",
        )
    with smoke_mid:
        current_smoking = _optional_code(
            variable_name="current_smoking",
            options=[0, 1, 2, 3, 888888],
            key="raw_current_smoking",
            fallback_help="Current smoking code used by the engineering logic.",
        )
    with smoke_right:
        ever_smk = _optional_code(
            variable_name="ever_smk",
            options=[0, 1, 2, 3, 4, 5, 888888],
            key="raw_ever_smk",
            fallback_help="Ever-smoking history code.",
        )

    alc_left, alc_mid, alc_right = st.columns(3)
    with alc_left:
        alcohol_status = _optional_code(
            variable_name="alcohol_status",
            options=[0, 1, 2],
            key="raw_alcohol_status",
            fallback_help="Alcohol status code.",
        )
    with alc_mid:
        alcohol = _optional_code(
            variable_name="alcohol",
            options=[0, 1, 2, 888888],
            key="raw_alcohol",
            fallback_help="Ever alcohol-use code used in fallback logic.",
        )
    with alc_right:
        con_alcohol = _optional_code(
            variable_name="con_alcohol",
            options=[0, 1, 999999],
            key="raw_con_alcohol",
            fallback_help="Current alcohol indicator for fallback logic.",
        )

    binge_left, binge_right = st.columns(2)
    with binge_left:
        drnk_30days = _optional_code(
            variable_name="drnk_30days",
            options=[0, 1, 999999],
            key="raw_drnk_30days",
            fallback_help="Drank an alcoholic drink within the past 30 days.",
        )
    with binge_right:
        binge_drink = _optional_code(
            variable_name="binge_drink",
            options=[0, 1, 99],
            key="raw_binge_drink",
            fallback_help="Binge-drink indicator used to set highest engineered level.",
        )

    st.caption("Provide anthropometric origins so BMI/WHR are computed with notebook-equivalent rules.")
    anthro_left, anthro_mid, anthro_right, anthro_last = st.columns(4)
    with anthro_left:
        weight = st.number_input(
            "weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=68.0,
            step=0.1,
            key="raw_weight",
            help=dictionary_labels.get("weight", "Average weight in kilograms."),
        )
    with anthro_mid:
        height = st.number_input(
            "height (cm or m)",
            min_value=0.8,
            max_value=260.0,
            value=165.0,
            step=0.1,
            key="raw_height",
            help=(dictionary_labels.get("height", "Average height.") + " Values >3 are treated as centimeters and converted to meters."),
        )
    with anthro_right:
        waist = st.number_input(
            "waist",
            min_value=30.0,
            max_value=200.0,
            value=84.0,
            step=0.1,
            key="raw_waist",
            help=dictionary_labels.get("waist", "Average waist circumference."),
        )
    with anthro_last:
        hip = st.number_input(
            "hip",
            min_value=30.0,
            max_value=200.0,
            value=96.0,
            step=0.1,
            key="raw_hip",
            help=dictionary_labels.get("hip", "Average hip circumference."),
        )

    return {
        "smoke_status": smoke_status,
        "current_smoking": current_smoking,
        "ever_smk": ever_smk,
        "alcohol_status": alcohol_status,
        "alcohol": alcohol,
        "con_alcohol": con_alcohol,
        "drnk_30days": drnk_30days,
        "binge_drink": binge_drink,
        "weight": weight,
        "height": height,
        "waist": waist,
        "hip": hip,
    }


def make_indicator_chart(score: float, lower: float, upper: float):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lower * 100.0, upper * 100.0],
            y=[0, 0],
            mode="lines",
            line=dict(color="#0f172a", width=20),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[score * 100.0],
            y=[0],
            mode="markers",
            marker=dict(color="#dc2626", size=18, line=dict(color="white", width=2)),
            hovertemplate="Risk score: %{x:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=170,
        margin=dict(l=10, r=10, t=12, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Probability (%)", range=[0, 100], showgrid=False, zeroline=False),
        yaxis=dict(visible=False),
    )
    return fig


def make_gauge_chart(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100.0,
            number={"suffix": "%", "font": {"size": 42, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#dc2626"},
                "steps": [
                    {"range": [0, 33], "color": "#eef2ff"},
                    {"range": [33, 67], "color": "#fee2e2"},
                    {"range": [67, 100], "color": "#fecaca"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 4}, "thickness": 0.75, "value": score * 100.0},
            },
        )
    )
    fig.update_layout(height=290, margin=dict(l=18, r=18, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _build_background_samples(feature_names: list[str], rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    records: list[dict[str, float]] = []

    for _ in range(rows):
        row: dict[str, float] = {}
        for feature_name in feature_names:
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


def _prediction_fn_for_explainability(model: Any, feature_names: list[str]):
    def _predict(nd_array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(nd_array, columns=feature_names)
        proba = model.predict_proba(frame)
        return np.asarray(proba)[:, 1]

    return _predict


def _try_compute_shap(model: Any, feature_names: list[str], input_frame: pd.DataFrame):
    try:
        import shap  # type: ignore
    except Exception:
        return None, None, "SHAP package is not installed. Install with: pip install shap"

    try:
        background = _build_background_samples(feature_names, rows=40)
        predict_fn = _prediction_fn_for_explainability(model, feature_names)
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


def _try_compute_lime(model: Any, feature_names: list[str], input_frame: pd.DataFrame):
    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    except Exception:
        return None, "LIME package is not installed. Install with: pip install lime"

    try:
        background = _build_background_samples(feature_names, rows=120)
        explainer = LimeTabularExplainer(
            training_data=background.values,
            feature_names=feature_names,
            class_names=["No HTN", "HTN"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        explanation = explainer.explain_instance(
            data_row=input_frame.iloc[0].values,
            predict_fn=model.predict_proba,
            num_features=min(12, len(feature_names)),
            top_labels=1,
        )

        pairs = explanation.as_list(label=1)
        lime_df = pd.DataFrame(pairs, columns=["rule", "weight"]) if pairs else pd.DataFrame(columns=["rule", "weight"])
        return lime_df, None
    except Exception as exc:
        return None, f"LIME computation failed: {exc}"


st.markdown(
    """
    <div class="hero">
      <h1>Hypertension Risk Studio</h1>
      <p>
        A focused front end for the thesis model: enter the same model variables, score risk, and surface a Venn-Abers uncertainty band from the backend.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


model_path = str(DEFAULT_MODEL_PATH)
calibrator_path = str(DEFAULT_CALIBRATOR_PATH)

with st.sidebar:
    st.subheader("Model artifacts")
    model_path = st.text_input("Stand-in model joblib", value=model_path)
    calibrator_path = st.text_input("Venn-Abers calibrator", value=calibrator_path)
    st.caption("Replace these paths later with your final exported artifacts.")
    st.divider()
    st.caption("The form is driven from the model's feature_names_in_ metadata.")
    st.caption("Optional explanations: install `shap` and `lime` to enable SHAP/LIME outputs.")


model, calibrator, feature_names = load_artifacts(model_path, calibrator_path)
grouped_features = group_feature_names(feature_names)
dictionary_labels, dictionary_value_labels = load_dataset_dictionaries()


left_col, right_col = st.columns([1.2, 0.8], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Input the model variables")
    with st.form("risk_form", clear_on_submit=False):
        widget_values: dict[str, float] = {}
        engineered_names = {"BMI", "bmi", "whr", "alcohol_level", "smoking_level"}
        has_engineered_features = any(
            name in engineered_names or name.startswith("alcohol_level_") or name.startswith("smoking_level_")
            for name in feature_names
        )

        if "Core clinical" in grouped_features:
            st.markdown("#### Core clinical")
            if "age" not in feature_names:
                st.caption("Note: `age` is not displayed because the currently loaded model does not include it in `feature_names_in_`.")
            core_cols = st.columns(3)
            for index, feature_name in enumerate(grouped_features["Core clinical"]):
                if feature_name in engineered_names:
                    continue
                with core_cols[index % 3]:
                    widget_values[feature_name] = render_number_input(feature_name, dictionary_labels, dictionary_value_labels)

        if "Anthropometrics" in grouped_features:
            st.markdown("#### Anthropometrics")
            anthro_cols = st.columns(3)
            for index, feature_name in enumerate(grouped_features["Anthropometrics"]):
                if feature_name in engineered_names:
                    continue
                with anthro_cols[index % 3]:
                    widget_values[feature_name] = render_number_input(feature_name, dictionary_labels, dictionary_value_labels)

        if "Dietary pattern" in grouped_features:
            with st.expander("Dietary pattern and nutrient signals", expanded=False):
                dietary_cols = st.columns(3)
                for index, feature_name in enumerate(grouped_features["Dietary pattern"]):
                    with dietary_cols[index % 3]:
                        widget_values[feature_name] = render_number_input(feature_name, dictionary_labels, dictionary_value_labels)

                dietary_meanings = []
                for feature_name in grouped_features["Dietary pattern"]:
                    dietary_meanings.append(
                        {
                            "variable": feature_name,
                            "meaning": field_help_text(feature_name, dictionary_labels) or "No definition found in loaded dictionaries.",
                        }
                    )

                st.markdown("##### Variable meanings from dataset dictionaries")
                st.dataframe(pd.DataFrame(dietary_meanings), use_container_width=True, hide_index=True)

                st.markdown("##### Example ordinary servings (approximate)")
                st.caption("These are quick, practical references to help with nutrient inputs.")
                st.markdown(
                    "- One cup cooked white rice has about 45 g carbohydrates (Total_CHO).\n"
                    "- One egg has about 6 g protein and 5 g fat (Total_Prot, Total_Fat).\n"
                    "- One medium banana has about 10 mg vitamin C and 27 g carbs (Total_VitC, Total_CHO).\n"
                    "- One cup milk has about 300 mg calcium and 8 g protein (Total_Calc, Total_Prot).\n"
                    "- One tablespoon cooking oil has about 14 g fat (Total_Fat).\n"
                    "- 100 g cooked chicken breast has about 31 g protein and 3.5 g fat (Total_Prot, Total_Fat).\n"
                    "- One medium orange has about 70 mg vitamin C (Total_VitC).\n"
                    "- One cup cooked mung beans has about 2.5 mg iron (Total_Iron).\n"
                    "- One cup cooked fortified cereal can provide around 4-8 mg niacin equivalents depending on brand (Total_Nia)."
                )

        if has_engineered_features:
            with st.expander("Raw variables for engineered features", expanded=True):
                widget_values.update(render_lifestyle_selectors(dictionary_labels, dictionary_value_labels))

        if "Other model inputs" in grouped_features:
            with st.expander("Other model inputs", expanded=False):
                other_cols = st.columns(3)
                for index, feature_name in enumerate(grouped_features["Other model inputs"]):
                    if feature_name in engineered_names:
                        continue
                    with other_cols[index % 3]:
                        widget_values[feature_name] = render_number_input(feature_name, dictionary_labels, dictionary_value_labels)

        submitted = st.form_submit_button("Score risk")
    st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Output")

    if submitted:
        input_values = build_input_values_from_widgets(feature_names, widget_values)
        input_frame = make_input_frame(feature_names, input_values)
        result = predict_with_venn_abers(model, input_frame, calibrator)

        kpi_left, kpi_mid, kpi_right = st.columns(3)
        with kpi_left:
            st.markdown(
                f"<div class='kpi-title'>Risk score</div><div class='kpi-value'>{result.calibrated_probability * 100:.1f}%</div><div class='kpi-subtitle'>Calibrated probability</div>",
                unsafe_allow_html=True,
            )
        with kpi_mid:
            st.markdown(
                f"<div class='kpi-title'>Interval</div><div class='kpi-value'>{result.lower_bound * 100:.1f}% - {result.upper_bound * 100:.1f}%</div><div class='kpi-subtitle'>Venn-Abers uncertainty band</div>",
                unsafe_allow_html=True,
            )
        with kpi_right:
            st.markdown(
                f"<div class='kpi-title'>Risk tier</div><div class='kpi-value'>{result.risk_label}</div><div class='kpi-subtitle'>Based on the calibrated score</div>",
                unsafe_allow_html=True,
            )

        st.plotly_chart(make_gauge_chart(result.calibrated_probability), use_container_width=True)
        st.plotly_chart(make_indicator_chart(result.calibrated_probability, result.lower_bound, result.upper_bound), use_container_width=True)

        summary_frame = pd.DataFrame(
            {
                "metric": ["raw probability", "calibrated probability", "lower bound", "upper bound", "uncertainty width"],
                "value": [
                    f"{result.raw_probability:.4f}",
                    f"{result.calibrated_probability:.4f}",
                    f"{result.lower_bound:.4f}",
                    f"{result.upper_bound:.4f}",
                    f"{result.uncertainty_width:.4f}",
                ],
            }
        )
        st.dataframe(summary_frame, use_container_width=True, hide_index=True)

        st.markdown("#### Explainability")
        st.caption("Local explanations for this single prediction. SHAP and LIME are generated automatically after scoring.")

        with st.spinner("Computing SHAP and LIME explanations..."):
            shap_local_df, shap_global_df, shap_error = _try_compute_shap(model, feature_names, input_frame)
            lime_df, lime_error = _try_compute_lime(model, feature_names, input_frame)

        shap_col, lime_col = st.columns(2)

        with shap_col:
            st.markdown("##### SHAP")
            if shap_error:
                st.warning(shap_error)
            elif shap_local_df is not None and not shap_local_df.empty:
                st.markdown("**Local SHAP (this case)**")
                st.dataframe(shap_local_df[["feature", "shap_value"]].head(12), use_container_width=True, hide_index=True)
                top_local = shap_local_df.head(10).iloc[::-1]
                st.plotly_chart(
                    go.Figure(
                        go.Bar(
                            x=top_local["shap_value"],
                            y=top_local["feature"],
                            orientation="h",
                            marker_color=["#dc2626" if value >= 0 else "#2563eb" for value in top_local["shap_value"]],
                        )
                    ).update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10)),
                    use_container_width=True,
                )

                st.markdown("**Global SHAP (overall)**")
                if shap_global_df is not None and not shap_global_df.empty:
                    st.dataframe(shap_global_df.head(12), use_container_width=True, hide_index=True)
                    top_global = shap_global_df.head(10).iloc[::-1]
                    st.plotly_chart(
                        go.Figure(
                            go.Bar(
                                x=top_global["mean_abs_shap"],
                                y=top_global["feature"],
                                orientation="h",
                                marker_color="#7c3aed",
                            )
                        ).update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10)),
                        use_container_width=True,
                    )
                else:
                    st.info("No global SHAP output produced.")

            else:
                st.info("No SHAP output produced.")

        with lime_col:
            st.markdown("##### LIME")
            if lime_error:
                st.warning(lime_error)
            elif lime_df is not None and not lime_df.empty:
                st.dataframe(lime_df, use_container_width=True, hide_index=True)
                top_lime = lime_df.head(10).iloc[::-1]
                st.plotly_chart(
                    go.Figure(
                        go.Bar(
                            x=top_lime["weight"],
                            y=top_lime["rule"],
                            orientation="h",
                            marker_color=["#dc2626" if value >= 0 else "#2563eb" for value in top_lime["weight"]],
                        )
                    ).update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10)),
                    use_container_width=True,
                )
            else:
                st.info("No LIME output produced.")
    else:
        st.info("Fill in the fields and press Score risk to generate a calibrated prediction.")

    st.markdown("</div>", unsafe_allow_html=True)


with st.expander("Model feature map", expanded=False):
    feature_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "default_value": [feature_default(feature_name) for feature_name in feature_names],
        }
    )
    st.dataframe(feature_frame, use_container_width=True, hide_index=True)
