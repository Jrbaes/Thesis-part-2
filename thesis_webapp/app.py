from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import streamlit.components.v1 as components

from backend import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    build_input_values_from_widgets,
    feature_default,
    feature_range,
    group_feature_names,
    load_calibrator,
    load_input_feature_names,
    load_model_feature_names,
    load_model,
    load_preprocessor,
    make_input_frame,
    prepare_model_input,
    predict_with_venn_abers,
    unwrap_model,
)
from app_constants import (
    AUTO_COMPUTED_TOTAL_FIELDS,
    CONDITIONALLY_ALLOWED_NA_CODES,
    DISPLAY_LABEL_OVERRIDES,
    FEATURE_DICTIONARY_ALIASES,
    FOOD_GROUP_COMPONENT_TOTALS,
    MISSING_INPUT_CODES,
    NO_HELP_FEATURES,
    VALUE_LABEL_OVERRIDES,
    VARIABLE_DEFINITION_OVERRIDES,
    get_dictionary_paths,
)
from counterfactuals import compute_counterfactuals
from explainability import try_compute_lime, try_compute_shap
from styles import apply_global_styles


PLOTLY_STATIC_CONFIG = {"displayModeBar": False}


st.set_page_config(page_title="Hypertension Risk Assesment", page_icon="", layout="wide", initial_sidebar_state="collapsed")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DICTIONARY_PATHS = get_dictionary_paths(PROJECT_ROOT)

apply_global_styles()


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, calibrator_path: str, preprocessor_path: str):
    model = load_model(Path(model_path))
    calibrator = load_calibrator(Path(calibrator_path))
    preprocessor = load_preprocessor(Path(preprocessor_path))
    input_feature_names = load_input_feature_names(model, preprocessor)
    model_feature_names = load_model_feature_names(model, preprocessor)
    return model, calibrator, preprocessor, input_feature_names, model_feature_names


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
    candidates = dictionary_name_candidates(feature_name)
    for candidate in candidates:
        if candidate in VARIABLE_DEFINITION_OVERRIDES:
            return VARIABLE_DEFINITION_OVERRIDES[candidate]

    for candidate in candidates:
        if candidate in dictionary_labels and dictionary_labels[candidate]:
            return dictionary_labels[candidate]
    return None


def field_value_labels(feature_name: str, dictionary_value_labels: dict[str, dict[str, str]]) -> dict[str, str]:
    candidates = dictionary_name_candidates(feature_name)
    for candidate in candidates:
        if candidate in VALUE_LABEL_OVERRIDES:
            return VALUE_LABEL_OVERRIDES[candidate]

    for candidate in candidates:
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


def _clean_label_text(text: str) -> str:
    return text.split(":", 1)[0].strip().replace("  ", " ")


def field_display_label(feature_name: str, dictionary_labels: dict[str, str]) -> str:
    candidates = dictionary_name_candidates(feature_name)
    for candidate in candidates:
        if candidate in DISPLAY_LABEL_OVERRIDES:
            return DISPLAY_LABEL_OVERRIDES[candidate]

    for candidate in candidates:
        if candidate in dictionary_labels and dictionary_labels[candidate]:
            return _clean_label_text(dictionary_labels[candidate])

    help_text = field_help_text(feature_name, dictionary_labels)
    if help_text:
        return _clean_label_text(help_text)

    fallback = feature_name.replace("epwt_", "").replace("_", " ").strip()
    return fallback.title()

def is_missing_input_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isnan(numeric_value) or numeric_value in MISSING_INPUT_CODES


def is_conditionally_allowed_na_value(feature_name: str, value: Any) -> bool:
    allowed_codes = CONDITIONALLY_ALLOWED_NA_CODES.get(feature_name)
    if not allowed_codes:
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return not np.isnan(numeric_value) and numeric_value in allowed_codes


def render_top_age_sex_fields(
    widget_values: dict[str, float],
    dictionary_labels: dict[str, str],
    feature_names: list[str],
    rendered_feature_names: set[str],
):
    top_cols = st.columns(3)
    if "age" in feature_names:
        with top_cols[0]:
            widget_values["age"] = render_number_input("age", dictionary_labels, {})
            rendered_feature_names.add("age")
    else:
        with top_cols[0]:
            widget_values["age"] = st.number_input("Age", min_value=0, max_value=120, value=40, step=1, help="Age in years.")
            rendered_feature_names.add("age")

    if "sex" in feature_names:
        with top_cols[1]:
            widget_values["sex"] = render_number_input("sex", dictionary_labels, VALUE_LABEL_OVERRIDES)
            rendered_feature_names.add("sex")
    else:
        with top_cols[1]:
            sex_choice = st.selectbox(
                "Sex",
                options=[None, 1, 2],
                index=0,
                format_func=lambda v: "missing" if v is None else ("Male" if v == 1 else "Female"),
                help="Biological sex.",
            )
            widget_values["sex"] = float("nan") if sex_choice is None else float(sex_choice)
            rendered_feature_names.add("sex")


def _format_numeric_text(value: float, step: float) -> str:
    if not np.isfinite(float(value)):
        return ""
    step_text = format(float(step), "f").rstrip("0").rstrip(".")
    decimals = 0
    if "." in step_text:
        decimals = len(step_text.split(".", 1)[1])
    return f"{float(value):.{decimals}f}" if decimals else str(int(round(float(value))))


def render_editable_numeric_input(
    label: str,
    minimum: float,
    maximum: float,
    default_value: float,
    step: float,
    widget_key: str,
    help_text: str | None,
):
    default_text = _format_numeric_text(float(default_value), float(step))
    existing_text = str(st.session_state.get(widget_key, default_text)).strip()

    text_value = st.text_input(
        label,
        value=existing_text if existing_text != "" else default_text,
        key=widget_key,
        help=help_text,
        placeholder=default_text,
    )

    cleaned_text = text_value.strip()
    if cleaned_text == "":
        return float(default_value)

    try:
        numeric_value = float(cleaned_text)
    except ValueError:
        st.warning(f"Please enter a valid numeric value for {label}.")
        return float("nan")

    if numeric_value < float(minimum) or numeric_value > float(maximum):
        st.warning(f"{label} must be between {minimum:g} and {maximum:g}.")
        return float("nan")

    return float(numeric_value)


def render_number_input(feature_name: str, dictionary_labels: dict[str, str], dictionary_value_labels: dict[str, dict[str, str]]):
    minimum, maximum, step = feature_range(feature_name)
    default_value = feature_default(feature_name)
    raw_help = field_help_text(feature_name, dictionary_labels)
    help_text = None if feature_name in NO_HELP_FEATURES else raw_help
    display_label = field_display_label(feature_name, dictionary_labels)
    widget_key = f"input_{feature_name}"
    is_food_group_field = feature_name.startswith("fg") or feature_name.startswith("epwt_fg")
    is_total_field = feature_name.startswith("Total_")
    force_numeric_input = is_food_group_field or is_total_field

    value_label_map = field_value_labels(feature_name, dictionary_value_labels)

    if feature_name == "age":
        age_text = st.text_input(
            display_label,
            value=st.session_state.get("input_age_text", ""),
            key="input_age_text",
            help=help_text,
            placeholder="missing",
        )
        if age_text is None or not age_text.strip():
            return float("nan")
        try:
            age_value = int(round(float(age_text)))
        except ValueError:
            st.warning("Please enter a valid numeric age or leave it blank.")
            return float("nan")
        if age_value < int(round(minimum)) or age_value > int(round(maximum)):
            st.warning(f"Age must be between {int(round(minimum))} and {int(round(maximum))}.")
            return float("nan")
        return float(age_value)

    if feature_name in AUTO_COMPUTED_TOTAL_FIELDS:
        st.number_input(
            display_label,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default_value),
            step=float(step),
            format="%.2f",
            key=f"input_{feature_name}",
            help=(help_text + " Auto-computed from dietary components above when Predict is pressed.") if help_text else "Auto-computed from dietary components above when Predict is pressed.",
            disabled=True,
        )
        st.caption("Auto-computed from prior dietary components above.")
        return float(default_value)

    if value_label_map and len(value_label_map) <= 30 and not force_numeric_input:
        if feature_name == "ethnicity":
            option_values_eth = _sorted_value_label_keys(value_label_map)
            selected_eth = st.selectbox(
                display_label,
                options=option_values_eth,
                index=option_values_eth.index("0") if "0" in option_values_eth else 0,
                key=f"input_{feature_name}",
                help=help_text,
                format_func=lambda choice: f"{choice} - {value_label_map.get(choice, '')}".strip(" -"),
            )
            try:
                return float(int(float(selected_eth)))
            except ValueError:
                return 0.0

        option_values: list[str | None] = [None] + _sorted_value_label_keys(value_label_map)

        def _format_choice(choice: str | None) -> str:
            if choice is None:
                return "missing"
            return f"{choice} - {value_label_map.get(choice, '')}".strip(" -")

        selected = st.selectbox(
            display_label,
            options=option_values,
            index=0,
            key=f"input_{feature_name}",
            help=help_text,
            format_func=_format_choice,
        )
        if selected is None:
            return float("nan")
        try:
            return float(int(float(selected)))
        except ValueError:
            return float("nan")

    numeric_value = render_editable_numeric_input(
        label=display_label,
        minimum=float(minimum),
        maximum=float(maximum),
        default_value=float(default_value),
        step=float(step),
        widget_key=widget_key,
        help_text=help_text,
    )

    return float(numeric_value)


def render_behavioral_selectors(dictionary_labels: dict[str, str], dictionary_value_labels: dict[str, dict[str, str]]):

    def _optional_code(
        variable_name: str,
        options: list[int],
        key: str,
        fallback_help: str,
        *,
        auto_value: int | None = None,
        disabled: bool = False,
    ):
        value_label_map = VALUE_LABEL_OVERRIDES.get(variable_name, dictionary_value_labels.get(variable_name, {}))
        variable_help = dictionary_labels.get(variable_name, fallback_help)

        option_values: list[int | None] = [None] + options

        if auto_value is not None:
            st.session_state[key] = auto_value
        elif key not in st.session_state or st.session_state[key] not in option_values:
            st.session_state[key] = None

        def _format_choice(choice: int | None) -> str:
            if choice is None:
                return "missing"
            label = value_label_map.get(str(choice), "")
            if "not applicable" in label.lower():
                return "N/A - Not Applicable"
            if label:
                return f"{choice} - {label}"
            return str(choice)

        selected = st.selectbox(
            field_display_label(variable_name, dictionary_labels),
            options=option_values,
            index=option_values.index(st.session_state.get(key)),
            key=key,
            format_func=_format_choice,
            help=variable_help,
            disabled=disabled,
        )
        return float("nan") if selected is None else float(int(selected))

    def _is_code(value: float, target: int) -> bool:
        return not (value is None or (isinstance(value, float) and np.isnan(value))) and int(value) == target

    smoke_left, smoke_mid, smoke_right = st.columns(3)
    with smoke_left:
        smoke_status = _optional_code(
            variable_name="smoke_status",
            options=[0, 1, 2],
            key="raw_smoke_status",
            fallback_help="Smoking status code.",
        )

    smoke_current_options = [0, 1, 2, 3, 888888]
    smoke_history_options = [0, 1, 2, 3, 4, 5, 888888]
    smoke_current_auto: int | None = None
    smoke_history_auto: int | None = None
    smoke_current_disabled = False
    smoke_history_disabled = False

    if _is_code(smoke_status, 0):
        smoke_current_auto = 0
        smoke_history_auto = 0
        smoke_current_disabled = True
        smoke_history_disabled = True
    elif _is_code(smoke_status, 1):
        smoke_current_options = [1, 2, 3]
        smoke_history_auto = 888888
        smoke_history_disabled = True
    elif _is_code(smoke_status, 2):
        smoke_current_auto = 0
        smoke_history_options = [1, 2, 3, 4, 5]
        smoke_current_disabled = True
        smoke_history_disabled = False

    with smoke_mid:
        current_smoking = _optional_code(
            variable_name="current_smoking",
            options=smoke_current_options,
            key="raw_current_smoking",
            fallback_help="Current smoking code used by the engineering logic.",
            auto_value=smoke_current_auto,
            disabled=smoke_current_disabled,
        )

    if smoke_history_auto is None and not smoke_history_disabled and _is_code(current_smoking, 1):
        smoke_history_auto = 1
        smoke_history_disabled = True
    if smoke_history_auto is None and not smoke_history_disabled and _is_code(current_smoking, 2):
        smoke_history_auto = 2
        smoke_history_disabled = True
    if smoke_history_auto is None and not smoke_history_disabled and _is_code(current_smoking, 3):
        smoke_history_auto = 3
        smoke_history_disabled = True

    with smoke_right:
        ever_smk = _optional_code(
            variable_name="ever_smk",
            options=smoke_history_options,
            key="raw_ever_smk",
            fallback_help="Ever-smoking history code.",
            auto_value=smoke_history_auto,
            disabled=smoke_history_disabled,
        )

    alc_left, alc_mid, alc_right = st.columns(3)
    with alc_left:
        alcohol_status = _optional_code(
            variable_name="alcohol_status",
            options=[0, 1, 2],
            key="raw_alcohol_status",
            fallback_help="Alcohol status code.",
        )

    alcohol_options = [0, 1, 2, 888888]
    con_alcohol_options = [0, 1, 999999]
    drnk_30days_options = [0, 1, 999999]
    alcohol_auto: int | None = None
    con_alcohol_auto: int | None = None
    drnk_30days_auto: int | None = None
    binge_auto: int | None = None
    alcohol_disabled = False
    con_alcohol_disabled = False
    drnk_30days_disabled = False
    binge_disabled = False

    if _is_code(alcohol_status, 0):
        alcohol_auto = 0
        con_alcohol_auto = 999999
        drnk_30days_auto = 999999
        binge_auto = 99
        alcohol_disabled = True
        con_alcohol_disabled = True
        drnk_30days_disabled = True
        binge_disabled = True
    elif _is_code(alcohol_status, 1):
        alcohol_auto = 1
        con_alcohol_auto = 1
        drnk_30days_options = [0, 1]
        alcohol_disabled = True
        con_alcohol_disabled = True
    elif _is_code(alcohol_status, 2):
        alcohol_auto = 1
        alcohol_disabled = True
        drnk_30days_auto = 999999
        binge_auto = 99
        drnk_30days_disabled = True
        binge_disabled = True

    with alc_mid:
        alcohol = _optional_code(
            variable_name="alcohol",
            options=alcohol_options,
            key="raw_alcohol",
            fallback_help="Ever alcohol-use code used in fallback logic.",
            auto_value=alcohol_auto,
            disabled=alcohol_disabled,
        )
    with alc_right:
        con_alcohol = _optional_code(
            variable_name="con_alcohol",
            options=con_alcohol_options,
            key="raw_con_alcohol",
            fallback_help="Current alcohol indicator for fallback logic.",
            auto_value=con_alcohol_auto,
            disabled=con_alcohol_disabled,
        )

    binge_left, binge_mid, _ = st.columns(3)
    with binge_left:
        drnk_30days = _optional_code(
            variable_name="drnk_30days",
            options=drnk_30days_options,
            key="raw_drnk_30days",
            fallback_help="Drank an alcoholic drink within the past 30 days.",
            auto_value=drnk_30days_auto,
            disabled=drnk_30days_disabled,
        )

    if not binge_disabled and _is_code(drnk_30days, 0):
        binge_auto = 0
        binge_disabled = True
    elif not binge_disabled and _is_code(drnk_30days, 999999):
        binge_auto = 99
        binge_disabled = True

    with binge_mid:
        binge_drink = _optional_code(
            variable_name="binge_drink",
            options=[0, 1, 99],
            key="raw_binge_drink",
            fallback_help="Binge-drink indicator used to set highest engineered level.",
            auto_value=binge_auto,
            disabled=binge_disabled,
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
    }


def render_anthro_origin_inputs(dictionary_labels: dict[str, str], rendered_feature_names: set[str]):
    raw_feature_order = ["weight", "height", "waist", "hip"]
    missing_raw = [feature for feature in raw_feature_order if feature not in rendered_feature_names]
    if not missing_raw:
        return {}

    cols = st.columns(3)
    values: dict[str, float] = {}

    for index, feature in enumerate(missing_raw):
        with cols[index % 3]:
            if feature == "weight":
                values["weight"] = render_editable_numeric_input(
                    label="Weight",
                    minimum=0.0,
                    maximum=300.0,
                    default_value=0.0,
                    step=0.1,
                    widget_key="raw_weight",
                    help_text=dictionary_labels.get("weight", "Weight in kilograms."),
                )
            elif feature == "height":
                values["height"] = render_editable_numeric_input(
                    label="Height",
                    minimum=0.0,
                    maximum=260.0,
                    default_value=0.0,
                    step=0.1,
                    widget_key="raw_height",
                    help_text=(dictionary_labels.get("height", "Height in cm or m.") + " Values >3 are treated as centimeters and converted to meters."),
                )
            elif feature == "waist":
                values["waist"] = render_editable_numeric_input(
                    label="Waist Circumference",
                    minimum=0.0,
                    maximum=200.0,
                    default_value=0.0,
                    step=0.1,
                    widget_key="raw_waist",
                    help_text=dictionary_labels.get("waist", "Waist circumference in cm."),
                )
            elif feature == "hip":
                values["hip"] = render_editable_numeric_input(
                    label="Hip Circumference",
                    minimum=0.0,
                    maximum=200.0,
                    default_value=0.0,
                    step=0.1,
                    widget_key="raw_hip",
                    help_text=dictionary_labels.get("hip", "Hip circumference in cm."),
                )

    return values


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
                    {"range": [0, 50], "color": "#dcfce7"},
                    {"range": [50, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 4}, "thickness": 0.75, "value": score * 100.0},
            },
        )
    )
    fig.update_layout(height=290, margin=dict(l=18, r=18, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _food_group_prefix_and_index(name: str) -> tuple[str, int] | None:
    for prefix in ("epwt_fg", "fg"):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            return (prefix, int(suffix)) if suffix.isdigit() else None
    return None


def _food_group_is_component_total(name: str) -> bool:
    return (parsed := _food_group_prefix_and_index(name)) is not None and parsed[1] in FOOD_GROUP_COMPONENT_TOTALS


def _food_group_is_addend(name: str) -> bool:
    return (parsed := _food_group_prefix_and_index(name)) is not None and parsed[1] in {num for nums in FOOD_GROUP_COMPONENT_TOTALS.values() for num in nums}


def _food_group_addend_names_for_total(total_name: str, available_names: set[str]) -> list[str]:
    if (parsed := _food_group_prefix_and_index(total_name)) is None:
        return []
    prefix, total_idx = parsed
    return [f"{prefix}{idx}" for idx in FOOD_GROUP_COMPONENT_TOTALS.get(total_idx, []) if f"{prefix}{idx}" in available_names]


def apply_dietary_derived_totals(widget_values: dict[str, float], feature_names: list[str]) -> dict[str, float]:
    values = dict(widget_values)
    feature_name_set = set(feature_names)

    # Enforce dictionary-defined subtotal fields from their component food-group inputs.
    for name in feature_names:
        if not _food_group_is_component_total(name):
            continue
        addends = _food_group_addend_names_for_total(name, feature_name_set)
        if not addends:
            continue
        total_val = 0.0
        for addend_name in addends:
            raw = values.get(addend_name)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            total_val += float(raw)
        values[name] = float(total_val)

    food_group_names = [
        name
        for name in feature_names
        if (name.startswith("epwt_fg") or name.startswith("fg")) and name not in {"fg", "epwt_fg"}
    ]
    food_group_total = 0.0
    for name in food_group_names:
        # Sum subtotal groups and standalone groups, but exclude addends already represented by subtotal groups.
        if _food_group_is_addend(name):
            continue
        raw = values.get(name)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        food_group_total += float(raw)

    if "Total_FoodIntake" in feature_names:
        values["Total_FoodIntake"] = float(food_group_total)
    if "Total_Food_epwt" in feature_names:
        values["Total_Food_epwt"] = float(food_group_total)

    cho = float(values.get("Total_CHO", 0.0) or 0.0)
    fat = float(values.get("Total_Fat", 0.0) or 0.0)

    # Approximate protein from protein-heavy food groups when explicit protein isn't available.
    protein_group_suffixes = {"7", "14", "15", "16", "17", "18", "19", "20", "21"}
    protein_source_total = 0.0
    for name in food_group_names:
        if _food_group_is_addend(name):
            continue
        suffix = ""
        if name.startswith("epwt_fg"):
            suffix = name.replace("epwt_fg", "")
        elif name.startswith("fg"):
            suffix = name.replace("fg", "")
        if suffix in protein_group_suffixes:
            raw = values.get(name)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            protein_source_total += float(raw)

    protein_val = 0.0
    if "Total_Protein" in feature_names and "Total_Protein" in values:
        protein_val = float(values.get("Total_Protein", 0.0) or 0.0)
    elif "Total_Prot" in feature_names and "Total_Prot" in values:
        protein_val = float(values.get("Total_Prot", 0.0) or 0.0)
    else:
        protein_val = float(protein_source_total * 0.18)

    if cho == 0.0 and fat == 0.0 and food_group_total > 0.0:
        cho = float(food_group_total * 0.30)
        fat = float(food_group_total * 0.04)

    energy_val = float((4.0 * cho) + (4.0 * protein_val) + (9.0 * fat))
    if "Total_Energy" in feature_names:
        values["Total_Energy"] = energy_val
    if "Total_Ener" in feature_names:
        values["Total_Ener"] = energy_val

    if "Total_Protein" in feature_names:
        values["Total_Protein"] = protein_val
    if "Total_Prot" in feature_names:
        values["Total_Prot"] = protein_val

    if "Total_Protein" in feature_names and "Total_Protein" not in values and "Total_Prot" in values:
        values["Total_Protein"] = float(values.get("Total_Prot", 0.0) or 0.0)
    if "Total_Prot" in feature_names and "Total_Prot" not in values and "Total_Protein" in values:
        values["Total_Prot"] = float(values.get("Total_Protein", 0.0) or 0.0)

    return values


def risk_label_from_score(probability: float) -> str:
    return "At Risk of Hypertension" if probability > 0.5 else "Not at Risk"


def _model_column_source_input_feature(model_column: str, input_features: list[str]) -> str | None:
    # Map transformed model columns (e.g., num__age, cat__ethnicity_1.0) back to raw input features.
    token = model_column.split("__", 1)[1] if "__" in model_column else model_column
    lowered_features = sorted({str(name).lower() for name in input_features}, key=len, reverse=True)
    token_lc = token.lower()

    if token_lc in lowered_features:
        return token_lc

    for feature_name in lowered_features:
        if token_lc.startswith(f"{feature_name}_"):
            return feature_name

    return None


def resolve_explainability_columns(
    model_columns: list[str],
    input_features: list[str],
    rendered_input_features: set[str],
) -> list[str]:
    rendered_lc = {str(name).lower() for name in rendered_input_features}
    return [str(c) for c in model_columns if (s := _model_column_source_input_feature(str(c), input_features)) and s in rendered_lc]

if "show_form" not in st.session_state:
    st.session_state.show_form = False
if "scored" not in st.session_state:
    st.session_state.scored = False


model_path = str(DEFAULT_MODEL_PATH)
calibrator_path = str(DEFAULT_CALIBRATOR_PATH)
preprocessor_path = str(DEFAULT_PREPROCESSOR_PATH)

with st.sidebar:
    st.subheader("Model artifacts")
    model_path = st.text_input("Stand-in model joblib", value=model_path)
    preprocessor_path = st.text_input("Preprocessor joblib", value=preprocessor_path)
    calibrator_path = st.text_input("Venn-Abers calibrator", value=calibrator_path)
    st.caption("Replace these paths when your final exported artifacts are ready.")
    st.divider()
    st.caption("Form is driven from the model's `feature_names_in_` metadata.")
    st.caption("Install `shap` and `lime` to enable explainability outputs.")


model, calibrator, preprocessor, input_feature_names, model_feature_names = load_artifacts(
    model_path,
    calibrator_path,
    preprocessor_path,
)
explain_model = unwrap_model(model)
feature_names = input_feature_names
grouped_features = group_feature_names(feature_names)
dictionary_labels, dictionary_value_labels = load_dataset_dictionaries()


if not st.session_state.show_form:
    st.markdown(
        """
        <div class="landing-hero">
          <div class="landing-badge">Philippine 2015 DOST-FNRI Thesis Model</div>
                    <h1>HRP-AI: A WEB APPLICATION FOR HYPERTENSION<br>RISK PREDICTION WITH CALIBRATED<br>EXPLAINABLE AI</h1>
          <p class="landing-sub" style="text-align:center;margin:0 auto;width:100%;">
                        A thesis-focused decision support interface for estimating hypertension risk using dietary, anthropometric, and clinical inputs, with calibrated probabilities and transparent explainability outputs. The AI model was trained on the DOST-FNRI 2015 NNS dataset.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, landing_cta_col, _ = st.columns([1.2, 2.6, 1.2])
    with landing_cta_col:
        st.markdown(
                        """
                        <div style="width:100%;margin:0.5rem 0 2.6rem;background:rgba(255,255,255,0.82);
                                                border:1px solid rgba(24,33,47,0.09);border-radius:22px;
                                                padding:1.6rem 2rem;box-shadow:0 8px 30px rgba(15,23,42,0.07);">
                            <div style="text-align:center;font-size:0.74rem;text-transform:uppercase;
                                                    letter-spacing:0.15em;color:#64748b;margin-bottom:1rem;">Model Performance</div>
                            <div style="display:flex;justify-content:center;gap:3rem;margin-bottom:1.2rem;">
                                <div style="text-align:center;">
                                    <div style="font-size:2rem;font-weight:800;color:#0f172a;">76.3%</div>
                                    <div style="font-size:0.82rem;color:#64748b;margin-top:0.2rem;">Accuracy</div>
                                </div>
                                <div style="text-align:center;">
                                    <div style="font-size:2rem;font-weight:800;color:#dc2626;">85.3%</div>
                                    <div style="font-size:0.82rem;color:#64748b;margin-top:0.2rem;">Recall</div>
                                </div>
                            </div>
                            <div style="font-size:0.83rem;color:#475569;line-height:1.65;text-align:justify;text-justify:inter-word;">
                                <strong>Accuracy</strong> is the share of all individuals the model classifies correctly.
                                <strong>Recall</strong> (sensitivity) is the share of true hypertensive cases the model catches —
                                prioritised here because missing a true positive in health screening carries greater risk than a false alarm.
                                Metrics are from the calibrated CatBoost (isotonic, threshold 0.35) evaluated on the held-out test set.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
        )

        _begin = st.button("Begin Assessment", use_container_width=True, type="primary")
    if _begin:
        st.session_state.show_form = True
        st.rerun()

    st.markdown(
        """
        <div class="landing-disclaimer">
          For research and academic use only. This interface does not replace clinical judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    nav_left, nav_right = st.columns([5, 1])
    with nav_left:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:0.7rem;padding:0.5rem 0;'>"
            "<span style='font-size:1.2rem;font-weight:700;color:#0f172a;'>Hypertension Risk Assesment</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button("Back Home"):
            st.session_state.show_form = False
            st.session_state.scored = False
            st.rerun()

    st.divider()

    st.markdown(
        "<div id='input-section' class='section-header'>"
        "<h2>Patient Details</h2>"
        "<p>Complete all sections below. Use the question-mark hover next to each field for definitions and guidance."
        " Click Predict My Risk to see the output below this form on the same page.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # No st.form wrapper – inputs update live so dietary totals auto-compute on every re-render.
    submitted = False
    widget_values: dict[str, float] = {}
    dietary_addend_fields: list[str] = []
    required_field_labels: dict[str, str] = {}
    rendered_feature_names: set[str] = set()
    engineered_names = {"BMI", "bmi", "whr", "alcohol_level", "smoking_level"}
    has_engineered_features = any(
        name in engineered_names or name.startswith("alcohol_level_") or name.startswith("smoking_level_")
        for name in feature_names
    )

    st.markdown("#### Quick Identity and Key Inputs")
    render_top_age_sex_fields(widget_values, dictionary_labels, feature_names, rendered_feature_names)
    required_field_labels["age"] = field_display_label("age", dictionary_labels)
    required_field_labels["sex"] = field_display_label("sex", dictionary_labels)

    if "Core clinical" in grouped_features:
        st.markdown("#### Clinical")
        core_cols = st.columns(3)

        priority = ["hemoglobin"]
        clinical_features = [f for f in grouped_features["Core clinical"] if f not in engineered_names and f not in {"age", "sex"}]
        if "hemoglobin" in feature_names and "hemoglobin" not in clinical_features:
            clinical_features.append("hemoglobin")
        ordered_features: list[str] = [f for f in priority if f in clinical_features]
        ordered_features.extend([f for f in clinical_features if f not in ordered_features])

        for index, feature_name in enumerate(ordered_features):
            if feature_name in engineered_names:
                continue
            if feature_name in rendered_feature_names:
                continue
            with core_cols[index % 3]:
                widget_values[feature_name] = render_number_input(feature_name, dictionary_labels, dictionary_value_labels)
                rendered_feature_names.add(feature_name)
                required_field_labels[feature_name] = field_display_label(feature_name, dictionary_labels)

    if has_engineered_features:
        st.markdown("#### Behavioral (Smoking and Alcohol)")
        st.caption("Provide original smoking and alcohol variables. The backend computes engineered levels.")
        widget_values.update(render_behavioral_selectors(dictionary_labels, dictionary_value_labels))
        for feature_name in ("smoke_status", "current_smoking", "ever_smk", "alcohol_status", "alcohol", "con_alcohol", "drnk_30days", "binge_drink"):
            required_field_labels[feature_name] = field_display_label(feature_name, dictionary_labels)

    if "Anthropometrics" in grouped_features:
        st.markdown("#### Anthropometrics")
        anthro_cols = st.columns(3)
        for index, feature_name in enumerate(grouped_features["Anthropometrics"]):
            if feature_name in engineered_names:
                continue
            if feature_name in rendered_feature_names:
                continue
            with anthro_cols[index % 3]:
                widget_values[feature_name] = render_number_input(feature_name, dictionary_labels, dictionary_value_labels)
                rendered_feature_names.add(feature_name)
                required_field_labels[feature_name] = field_display_label(feature_name, dictionary_labels)

    if has_engineered_features:
        st.caption("For engineered BMI and WHR, provide these origin anthropometric measurements.")
        widget_values.update(render_anthro_origin_inputs(dictionary_labels, rendered_feature_names))

    if "Dietary pattern" in grouped_features:
        st.markdown("#### Dietary")
        st.caption("Enter each dietary value as your daily intake. Definitions for each dietary component are available in the field hover tooltips (?).")
        st.markdown("##### Example ordinary servings (approximate)")
        st.caption("Use these examples as quick references before entering daily dietary values.")
        st.markdown(
            "- One cup cooked white rice ≈ 45 g carbohydrates (Total_CHO).\n"
            "- One egg ≈ 6 g protein and 5 g fat (Total_Prot, Total_Fat).\n"
            "- One medium banana ≈ 10 mg vitamin C and 27 g carbs (Total_VitC, Total_CHO).\n"
            "- One cup milk ≈ 300 mg calcium and 8 g protein (Total_Calc, Total_Prot).\n"
            "- One tablespoon cooking oil ≈ 14 g fat (Total_Fat)."
        )
        dietary_features = list(grouped_features["Dietary pattern"])
        if "vita" in feature_names and "vita" not in dietary_features:
            dietary_features.append("vita")

        def _food_group_position_key(name: str) -> float | None:
            parsed = _food_group_prefix_and_index(name)
            if parsed is None:
                return None
            _, idx = parsed
            if idx in FOOD_GROUP_COMPONENT_TOTALS:
                return float(max(FOOD_GROUP_COMPONENT_TOTALS[idx])) + 0.5
            return float(idx)

        # Match dictionary-style ordering, with subtotal fields moved after their component addends.
        original_positions = {name: idx for idx, name in enumerate(dietary_features)}
        dietary_features = sorted(
            dietary_features,
            key=lambda name: (
                0 if _food_group_position_key(name) is not None else 1,
                _food_group_position_key(name) if _food_group_position_key(name) is not None else 10_000,
                original_positions[name],
            ),
        )

        dietary_input_features: list[str] = []
        for feature_name in dietary_features:
            if feature_name in rendered_feature_names:
                continue
            if feature_name in AUTO_COMPUTED_TOTAL_FIELDS:
                widget_values[feature_name] = float(feature_default(feature_name))
                rendered_feature_names.add(feature_name)
                continue
            dietary_input_features.append(feature_name)

        dietary_addend_fields = [
            name
            for name in dietary_input_features
            if _food_group_prefix_and_index(name) is not None and not _food_group_is_component_total(name)
        ]

        for start_index in range(0, len(dietary_input_features), 3):
            dietary_cols = st.columns(3)
            for offset, feature_name in enumerate(dietary_input_features[start_index:start_index + 3]):
                with dietary_cols[offset]:
                    if _food_group_is_component_total(feature_name):
                        addend_names = _food_group_addend_names_for_total(feature_name, set(dietary_features))
                        subtotal_value = 0.0
                        for addend_name in addend_names:
                            raw = widget_values.get(addend_name)
                            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                                continue
                            subtotal_value += float(raw)

                        minimum, maximum, step = feature_range(feature_name)
                        display_label = field_display_label(feature_name, dictionary_labels)
                        help_text = field_help_text(feature_name, dictionary_labels)
                        auto_key = f"auto_{feature_name}"
                        st.session_state[auto_key] = float(subtotal_value)
                        st.number_input(
                            display_label,
                            min_value=float(minimum),
                            max_value=float(maximum),
                            value=float(subtotal_value),
                            step=float(step),
                            format="%.2f",
                            key=auto_key,
                            disabled=True,
                            help=(help_text + " Auto-computed from the component food groups.") if help_text else "Auto-computed from the component food groups.",
                        )
                        st.caption("Auto-summed from the food-group inputs above.")
                        widget_values[feature_name] = float(subtotal_value)
                    else:
                        widget_values[feature_name] = render_number_input(feature_name, dictionary_labels, dictionary_value_labels)
                    rendered_feature_names.add(feature_name)
                    required_field_labels[feature_name] = field_display_label(feature_name, dictionary_labels)
                    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        _protein_group_sfx = {"7", "14", "15", "16", "17", "18", "19", "20", "21"}
        _live_food = 0.0
        _live_protein_src = 0.0
        for _fn in dietary_features:
            if _fn not in widget_values:
                continue
            _raw = widget_values[_fn]
            if _raw is None or (isinstance(_raw, float) and np.isnan(_raw)):
                continue
            _is_fg = _fn.startswith("epwt_fg") or (_fn.startswith("fg") and not _fn.startswith("epwt_"))
            if _is_fg:
                if _food_group_is_addend(_fn):
                    continue
                _live_food += float(_raw)
                _sfx = _fn.replace("epwt_fg", "").replace("fg", "")
                if _sfx in _protein_group_sfx:
                    _live_protein_src += float(_raw)
        _live_protein = _live_protein_src * 0.18
        _live_cho = float(widget_values.get("Total_CHO") or 0.0)
        _live_fat = float(widget_values.get("Total_Fat") or 0.0)
        if _live_cho == 0.0 and _live_fat == 0.0 and _live_food > 0.0:
            _live_cho = _live_food * 0.30
            _live_fat = _live_food * 0.04
        _live_energy = 4.0 * _live_cho + 4.0 * _live_protein + 9.0 * _live_fat
        # Store live values so submit handler picks them up
        for _alias in ("Total_FoodIntake", "Total_Food_epwt"):
            if _alias in feature_names:
                widget_values[_alias] = _live_food
        for _alias in ("Total_Energy", "Total_Ener"):
            if _alias in feature_names:
                widget_values[_alias] = _live_energy
        for _alias in ("Total_Protein", "Total_Prot"):
            if _alias in feature_names:
                widget_values[_alias] = _live_protein

        _tot_a, _tot_b, _tot_c = st.columns(3)
        with _tot_a:
            st.session_state["_disp_food"] = round(_live_food, 1)
            st.number_input("Total Food Intake (g)", value=round(_live_food, 1), disabled=True, key="_disp_food")
            st.caption("Auto-updates from the dietary values above.")
        with _tot_b:
            st.session_state["_disp_energy"] = round(_live_energy, 1)
            st.number_input("Total Energy (kcal)", value=round(_live_energy, 1), disabled=True, key="_disp_energy")
            st.caption("Auto-updates from the dietary values above.")
        with _tot_c:
            st.session_state["_disp_protein"] = round(_live_protein, 1)
            st.number_input("Total Protein (g)", value=round(_live_protein, 1), disabled=True, key="_disp_protein")
            st.caption("Auto-updates from the dietary values above.")

    missing_field_labels = [
        label
        for feature_name, label in required_field_labels.items()
        if is_missing_input_value(widget_values.get(feature_name))
        and not is_conditionally_allowed_na_value(feature_name, widget_values.get(feature_name))
    ]
    if missing_field_labels:
        missing_preview = ", ".join(missing_field_labels[:8])
        if len(missing_field_labels) > 8:
            missing_preview += f", and {len(missing_field_labels) - 8} more"
        st.warning(f"Complete all missing inputs before predicting: {missing_preview}.")
        st.session_state.scored = False
        st.session_state.pop("_result", None)
        st.session_state.pop("_input_frame", None)
        st.session_state.pop("_model_feature_names", None)
        st.session_state.pop("_rendered_input_features", None)

    dietary_all_zero = False
    if dietary_addend_fields:
        dietary_all_zero = all(
            float(widget_values.get(name, 0.0) or 0.0) == 0.0 for name in dietary_addend_fields
        )
        if dietary_all_zero:
            st.warning("Enter at least one non-zero food-group dietary intake value before predicting.")
            st.session_state.scored = False
            st.session_state.pop("_result", None)
            st.session_state.pop("_input_frame", None)
            st.session_state.pop("_model_feature_names", None)
            st.session_state.pop("_rendered_input_features", None)

    st.markdown("<br>", unsafe_allow_html=True)
    _, submit_col, _ = st.columns([1.3, 1.2, 1.3])
    with submit_col:
        submitted = st.button("Predict My Risk", use_container_width=True, type="primary", disabled=bool(missing_field_labels) or dietary_all_zero)

    components.html(
                """
                <script>
                (function() {
                    const doc = window.parent.document;
                    const submitButtons = Array.from(doc.querySelectorAll('button')).filter(
                        (btn) => btn.innerText && btn.innerText.trim() === 'Predict My Risk'
                    );
                    if (!submitButtons.length) return;

                    const submitBtn = submitButtons[submitButtons.length - 1];
                    const formContainer = doc;
                    const selector = 'input:not([type="hidden"]):not([disabled]), textarea';
                    const numericPlaceholderPattern = /^-?\\d+(?:\\.\\d+)?$/;

                    function visibleFields() {
                        return Array.from(formContainer.querySelectorAll(selector)).filter((el) => el.offsetParent !== null);
                    }

                    function bindClearOnFocus(el) {
                        if (el.dataset.clearOnFocusBound === '1') return;
                        if (el.tagName !== 'INPUT' || el.type !== 'text') return;
                        const placeholder = (el.getAttribute('placeholder') || '').trim();
                        if (!numericPlaceholderPattern.test(placeholder)) return;
                        el.dataset.clearOnFocusBound = '1';
                        el.addEventListener('focus', function() {
                            el.value = '';
                        });
                        el.addEventListener('blur', function() {
                            if (el.value.trim() !== '') return;
                            el.value = placeholder;
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        });
                    }

                    visibleFields().forEach((el) => {
                        bindClearOnFocus(el);
                        if (el.dataset.enterNavBound === '1') return;
                        el.dataset.enterNavBound = '1';
                        el.addEventListener('keydown', function(e) {
                            if (e.key !== 'Enter' || e.shiftKey) return;
                            e.preventDefault();
                            const fields = visibleFields();
                            const idx = fields.indexOf(el);
                            if (idx >= 0 && idx < fields.length - 1) {
                                fields[idx + 1].focus();
                                if (typeof fields[idx + 1].select === 'function') {
                                    fields[idx + 1].select();
                                }
                            } else {
                                submitBtn.click();
                            }
                        });
                    });
                })();
                </script>
                """,
                height=0,
        )

    if submitted and not missing_field_labels and not dietary_all_zero:
        derived_widget_values = apply_dietary_derived_totals(widget_values, feature_names)
        input_values = build_input_values_from_widgets(feature_names, derived_widget_values)
        raw_input_frame = make_input_frame(feature_names, input_values)
        model_input_frame = prepare_model_input(raw_input_frame, preprocessor)
        result = predict_with_venn_abers(model, model_input_frame, calibrator)
        st.session_state.scored = True
        st.session_state._result = result
        st.session_state._input_frame = model_input_frame
        st.session_state._model_feature_names = list(model_input_frame.columns)
        st.session_state._input_feature_names = feature_names
        st.session_state._rendered_input_features = sorted(rendered_feature_names)
        st.session_state._all_widget_values = dict(derived_widget_values)
        st.session_state._scroll_to_output = True

    if st.session_state.scored and hasattr(st.session_state, "_result"):
        result = st.session_state._result
        input_frame = st.session_state._input_frame

        st.markdown('<div id="output-section"></div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='section-header' style='margin-top:2.6rem;'>"
            "<h2>Output</h2>"
            "<p>The form stays above. Scroll up anytime to edit the inputs.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        score_pct = result.calibrated_probability * 100.0
        mapped_risk_label = risk_label_from_score(result.calibrated_probability)
        tier_css = "risk-at-risk" if result.calibrated_probability > 0.5 else "risk-not-at-risk"

        kpi_a, kpi_b, kpi_c = st.columns(3)
        with kpi_a:
            st.markdown(
                f"<div class='output-hero'><div class='oh-label'>Risk Score</div><div class='oh-value'>{score_pct:.1f}%</div><div class='oh-sub'>Calibrated probability</div></div>",
                unsafe_allow_html=True,
            )
        with kpi_b:
            st.markdown(
                f"<div class='output-hero'><div class='oh-label'>Uncertainty Interval</div><div class='oh-value' style='font-size:1.6rem;'>{result.lower_bound*100:.1f}% - {result.upper_bound*100:.1f}%</div><div class='oh-sub'>Venn-Abers interval</div></div>",
                unsafe_allow_html=True,
            )
        with kpi_c:
            st.markdown(
                f"<div class='output-hero'><div class='oh-label'>Risk Classification</div><div class='oh-value' style='font-size:1.15rem;'><span class='risk-badge {tier_css}'>{mapped_risk_label}</span></div><div class='oh-sub'>Based on calibrated probability</div></div>",
                unsafe_allow_html=True,
            )

        chart_left, chart_right = st.columns([1, 1.5])
        with chart_left:
            st.markdown("**Risk Gauge**")
            st.plotly_chart(make_gauge_chart(result.calibrated_probability), use_container_width=True, config=PLOTLY_STATIC_CONFIG)
        with chart_right:
            st.markdown("**Prediction Summary**")
            summary_frame = pd.DataFrame(
                {
                    "metric": [
                        "Calibrated Probability",
                        "Lower Uncertainty Interval Bound",
                        "Higher Uncertainty Interval Bound",
                        "Uncertainty Interval Width / Range",
                    ],
                    "value": [
                        f"{result.calibrated_probability:.4f}",
                        f"{result.lower_bound:.4f}",
                        f"{result.upper_bound:.4f}",
                        f"{result.uncertainty_width:.4f}",
                    ],
                }
            )
            st.table(summary_frame)

        st.markdown(
            "<div class='section-header' style='margin-top:1.8rem;'>"
            "<h2>Counterfactual Analysis</h2>"
            "<p>What single-feature changes would most reduce your hypertension risk score?</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        _all_wvals = st.session_state.get("_all_widget_values", {})
        _input_fnames = st.session_state.get("_input_feature_names", [])
        if _all_wvals and _input_fnames:
            with st.spinner("Computing counterfactuals..."):
                cf_df = compute_counterfactuals(
                    model=unwrap_model(model),
                    preprocessor=preprocessor,
                    calibrator=calibrator,
                    all_widget_values=_all_wvals,
                    input_feature_names=_input_fnames,
                    dictionary_labels=dictionary_labels,
                    current_probability=result.calibrated_probability,
                )
            if cf_df.empty:
                st.info("No single-feature change was found to reduce risk further.")
            else:
                st.caption(
                    "Wachter-style counterfactuals: each row shows the optimal value for one actionable feature "
                    "found by minimising a proximity-weighted loss that balances pushing the predicted probability "
                    "below the decision boundary while staying as close as possible to the original input. "
                    "All other features are held constant."
                )
                st.dataframe(cf_df, use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='section-header' style='margin-top:1.8rem;'>"
            "<h2>Explainability</h2>"
            "<p>Local explanations for this prediction.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Computing SHAP and LIME explanations..."):
            explain_feature_names = st.session_state.get("_model_feature_names", model_feature_names)
            rendered_inputs = set(st.session_state.get("_rendered_input_features", []))
            input_features_for_mapping = st.session_state.get("_input_feature_names", feature_names)
            explain_feature_names = resolve_explainability_columns(
                model_columns=list(explain_feature_names),
                input_features=list(input_features_for_mapping),
                rendered_input_features=rendered_inputs,
            )

            if explain_feature_names:
                explain_input_frame = input_frame.loc[:, explain_feature_names].copy()
                shap_local_df, shap_global_df, shap_error = try_compute_shap(
                    explain_model,
                    explain_feature_names,
                    explain_input_frame,
                    base_input_frame=input_frame,
                )
                lime_df, lime_error = try_compute_lime(
                    explain_model,
                    explain_feature_names,
                    explain_input_frame,
                    base_input_frame=input_frame,
                )
            else:
                shap_local_df, shap_global_df, shap_error = None, None, "No explainability columns are currently mapped to visible webpage inputs."
                lime_df, lime_error = None, "No explainability columns are currently mapped to visible webpage inputs."

        exp_a, exp_b, exp_c = st.columns(3)

        with exp_a:
            st.markdown("##### SHAP Local")
            if shap_error:
                st.warning(shap_error)
            elif shap_local_df is not None and not shap_local_df.empty:
                local_top = shap_local_df.head(12)
                st.table(local_top[["feature", "shap_value"]])
                chart_df = local_top.head(10).iloc[::-1]
                st.plotly_chart(
                    go.Figure(
                        go.Bar(
                            x=chart_df["shap_value"],
                            y=chart_df["feature"],
                            orientation="h",
                            marker_color=["#dc2626" if value >= 0 else "#2563eb" for value in chart_df["shap_value"]],
                        )
                    ).update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10)),
                    use_container_width=True,
                    config=PLOTLY_STATIC_CONFIG,
                )
            else:
                st.info("No SHAP local output produced.")

        with exp_b:
            st.markdown("##### SHAP Global")
            if shap_error:
                st.warning(shap_error)
            elif shap_global_df is not None and not shap_global_df.empty:
                global_top = shap_global_df.head(12)
                st.table(global_top)
                chart_df = global_top.head(10)
                st.plotly_chart(
                    go.Figure(
                        go.Waterfall(
                            orientation="v",
                            measure=["relative"] * len(chart_df),
                            x=chart_df["feature"],
                            y=chart_df["mean_abs_shap"],
                            connector={"line": {"color": "#9ca3af"}},
                            increasing={"marker": {"color": "#7c3aed"}},
                            decreasing={"marker": {"color": "#7c3aed"}},
                            text=[f"{v:.4f}" for v in chart_df["mean_abs_shap"]],
                            textposition="outside",
                        )
                    ).update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=60),
                        xaxis_tickangle=-40,
                        showlegend=False,
                    ),
                    use_container_width=True,
                    config=PLOTLY_STATIC_CONFIG,
                )
            else:
                st.info("No SHAP global output produced.")

        with exp_c:
            st.markdown("##### LIME Local")
            if lime_error:
                st.warning(lime_error)
            elif lime_df is not None and not lime_df.empty:
                lime_top = lime_df.head(12)
                st.table(lime_top)
                chart_df = lime_top.head(10).iloc[::-1]
                st.plotly_chart(
                    go.Figure(
                        go.Bar(
                            x=chart_df["weight"],
                            y=chart_df["rule"],
                            orientation="h",
                            marker_color=["#dc2626" if value >= 0 else "#2563eb" for value in chart_df["weight"]],
                        )
                    ).update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10)),
                    use_container_width=True,
                    config=PLOTLY_STATIC_CONFIG,
                )
            else:
                st.info("No LIME local output produced.")

        if st.session_state.get("_scroll_to_output", False):
            components.html(
                """
                <script>
                (function() {
                  const tries = { count: 0 };
                  function scrollToOutput() {
                    const anchors = window.parent.document.querySelectorAll('[id="output-section"]');
                    const el = anchors.length ? anchors[anchors.length - 1] : null;
                    if (el) {
                      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    } else if (tries.count < 20) {
                      tries.count += 1;
                      setTimeout(scrollToOutput, 200);
                    }
                  }
                  setTimeout(scrollToOutput, 250);
                })();
                </script>
                """,
                height=0,
            )
            st.session_state._scroll_to_output = False

