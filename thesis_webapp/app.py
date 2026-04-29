from __future__ import annotations

import math
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
    page_title="Hypertension Risk Assesment",
    page_icon="",
    layout="wide",
        initial_sidebar_state="collapsed",
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
        "0": "No, Not an IP/Without Foreign Blood (default)",
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

DISPLAY_LABEL_OVERRIDES = {
    "age": "Age",
    "sex": "Sex",
    "uic": "Urinary Iodine Concentration (UIC)",
    "vita": "Vitamin A",
    "hemoglobin": "Hemoglobin",
    "waist": "Waist Circumference",
    "hip": "Hip Circumference",
    "weight": "Weight",
    "height": "Height",
    "epwt_fg1": "Cereal and Cereal Products",
    "smoke_status": "Smoking Status",
    "current_smoking": "Current Smoking Frequency",
    "ever_smk": "Smoking History",
    "alcohol_status": "Alcohol Use Status",
    "alcohol": "Alcohol Consumption",
    "con_alcohol": "Alcohol Use in Past 12 Months",
    "drnk_30days": "Alcohol Use in Past 30 Days",
    "binge_drink": "Binge Drinking Status",
}

AUTO_COMPUTED_TOTAL_FIELDS = {
    "Total_FoodIntake",
    "Total_Food_epwt",
    "Total_Energy",
    "Total_Ener",
    "Total_Protein",
    "Total_Prot",
}

AVERAGE_DEFAULT_NOTICE_FIELDS = {
    "Total_Calcium",
    "Total_Calc",
    "Total_Iron",
    "Total_VitaminA",
    "Total_VitA",
    "Total_VitaminC",
    "Total_VitC",
    "Total_Thiamin",
    "Total_Thia",
    "Total_Riboflavin",
    "Total_Ribo",
    "Total_Niacin",
    "Total_Nia",
    "Total_CHO",
    "Total_Fat",
}

DIETARY_COMMON_FOODS = [
    {"food": "Pandesal", "serving": "1 piece", "group": "Bread/Merienda", "carbs_g": 16, "protein_g": 3, "fat_g": 1.5, "sugar_g": 2, "note": "Approximate per piece"},
    {"food": "Cooked white rice (kanin)", "serving": "1 cup", "group": "Staple/Starch", "carbs_g": 45, "protein_g": 4, "fat_g": 0.4, "sugar_g": 0.1, "note": "Approximate per cup"},
    {"food": "Garlic fried rice (sinangag)", "serving": "1 cup", "group": "Staple/Starch", "carbs_g": 50, "protein_g": 5, "fat_g": 5, "sugar_g": 0.5, "note": "Oil affects fat"},
    {"food": "Brown rice", "serving": "1 cup", "group": "Staple/Starch", "carbs_g": 45, "protein_g": 5, "fat_g": 1.8, "sugar_g": 0.4, "note": "Higher fiber"},
    {"food": "Boiled saba", "serving": "1 medium", "group": "Staple/Starch", "carbs_g": 27, "protein_g": 1.2, "fat_g": 0.3, "sugar_g": 14, "note": "Common merienda"},
    {"food": "Sweet potato (kamote)", "serving": "1 medium", "group": "Staple/Starch", "carbs_g": 24, "protein_g": 2, "fat_g": 0.1, "sugar_g": 7, "note": "Root crop"},
    {"food": "Cassava", "serving": "1/2 cup cooked", "group": "Staple/Starch", "carbs_g": 38, "protein_g": 1.4, "fat_g": 0.3, "sugar_g": 1.7, "note": "Approximate"},
    {"food": "Pancit (cooked)", "serving": "1 cup", "group": "Staple/Starch", "carbs_g": 35, "protein_g": 7, "fat_g": 7, "sugar_g": 2, "note": "Varies by recipe"},
    {"food": "Banana", "serving": "1 medium", "group": "Fruit", "carbs_g": 27, "protein_g": 1.3, "fat_g": 0.4, "sugar_g": 14, "note": "Common daily fruit"},
    {"food": "Mango", "serving": "1 cup slices", "group": "Fruit", "carbs_g": 25, "protein_g": 1.4, "fat_g": 0.6, "sugar_g": 23, "note": "Seasonal tropical fruit"},
    {"food": "Papaya", "serving": "1 cup cubes", "group": "Fruit", "carbs_g": 16, "protein_g": 0.9, "fat_g": 0.4, "sugar_g": 11, "note": "Breakfast fruit"},
    {"food": "Pineapple", "serving": "1 cup chunks", "group": "Fruit", "carbs_g": 22, "protein_g": 0.9, "fat_g": 0.2, "sugar_g": 16, "note": "Approximate"},
    {"food": "Watermelon", "serving": "1 cup cubes", "group": "Fruit", "carbs_g": 12, "protein_g": 0.9, "fat_g": 0.2, "sugar_g": 9, "note": "Hydrating fruit"},
    {"food": "Calamansi juice (sweetened)", "serving": "1 glass", "group": "Beverage", "carbs_g": 18, "protein_g": 0, "fat_g": 0, "sugar_g": 16, "note": "Depends on added sugar"},
    {"food": "Malunggay leaves", "serving": "1/2 cup cooked", "group": "Vegetable", "carbs_g": 4, "protein_g": 2.5, "fat_g": 0.5, "sugar_g": 1, "note": "Leafy vegetable"},
    {"food": "Kangkong", "serving": "1/2 cup cooked", "group": "Vegetable", "carbs_g": 3, "protein_g": 2, "fat_g": 0.2, "sugar_g": 0.4, "note": "Leafy vegetable"},
    {"food": "Pechay", "serving": "1/2 cup cooked", "group": "Vegetable", "carbs_g": 2.5, "protein_g": 1.2, "fat_g": 0.1, "sugar_g": 1, "note": "Common in soups"},
    {"food": "Ampalaya", "serving": "1/2 cup cooked", "group": "Vegetable", "carbs_g": 4, "protein_g": 1, "fat_g": 0.1, "sugar_g": 1.8, "note": "Bitter gourd"},
    {"food": "Eggplant (talong)", "serving": "1/2 cup cooked", "group": "Vegetable", "carbs_g": 4, "protein_g": 0.8, "fat_g": 0.2, "sugar_g": 2.2, "note": "Common in pinakbet"},
    {"food": "Sitaw", "serving": "1/2 cup cooked", "group": "Vegetable", "carbs_g": 5, "protein_g": 1.3, "fat_g": 0.2, "sugar_g": 2, "note": "String beans"},
    {"food": "Chicken adobo", "serving": "1 serving (90-100 g)", "group": "Protein", "carbs_g": 2, "protein_g": 24, "fat_g": 12, "sugar_g": 1, "note": "Sauce and skin vary"},
    {"food": "Pork adobo", "serving": "1 serving (90 g)", "group": "Protein", "carbs_g": 2, "protein_g": 20, "fat_g": 18, "sugar_g": 1, "note": "Choose lean cuts"},
    {"food": "Tinolang manok", "serving": "1 bowl", "group": "Protein", "carbs_g": 4, "protein_g": 20, "fat_g": 8, "sugar_g": 2, "note": "Includes broth/veg"},
    {"food": "Sinigang na isda", "serving": "1 bowl", "group": "Protein", "carbs_g": 3, "protein_g": 18, "fat_g": 5, "sugar_g": 1, "note": "Fish + vegetables"},
    {"food": "Grilled bangus", "serving": "1 palm-size piece", "group": "Protein", "carbs_g": 0, "protein_g": 22, "fat_g": 10, "sugar_g": 0, "note": "Fish option"},
    {"food": "Galunggong", "serving": "1 medium fish", "group": "Protein", "carbs_g": 0, "protein_g": 18, "fat_g": 8, "sugar_g": 0, "note": "Affordable fish"},
    {"food": "Canned sardines", "serving": "1 small can", "group": "Protein", "carbs_g": 2, "protein_g": 20, "fat_g": 10, "sugar_g": 0, "note": "Watch sodium"},
    {"food": "Egg", "serving": "1 piece", "group": "Protein", "carbs_g": 0.6, "protein_g": 6.3, "fat_g": 5.3, "sugar_g": 0.2, "note": "Whole egg"},
    {"food": "Tokwa (tofu)", "serving": "1/2 cup", "group": "Protein", "carbs_g": 2, "protein_g": 10, "fat_g": 6, "sugar_g": 0.3, "note": "Plant protein"},
    {"food": "Ginisang monggo", "serving": "1 cup", "group": "Protein", "carbs_g": 19, "protein_g": 14, "fat_g": 4, "sugar_g": 3, "note": "Legume dish"},
    {"food": "Milk", "serving": "1 cup", "group": "Dairy", "carbs_g": 12, "protein_g": 8, "fat_g": 8, "sugar_g": 12, "note": "Regular milk"},
    {"food": "Yogurt (plain)", "serving": "3/4 cup", "group": "Dairy", "carbs_g": 9, "protein_g": 8, "fat_g": 3, "sugar_g": 8, "note": "Unsweetened preferred"},
    {"food": "Cheese", "serving": "1 slice (30 g)", "group": "Dairy", "carbs_g": 1, "protein_g": 7, "fat_g": 9, "sugar_g": 0.5, "note": "Use in moderation"},
    {"food": "Turon", "serving": "1 piece", "group": "Merienda", "carbs_g": 35, "protein_g": 3, "fat_g": 9, "sugar_g": 14, "note": "Fried snack"},
    {"food": "Banana cue", "serving": "1 stick", "group": "Merienda", "carbs_g": 40, "protein_g": 1.5, "fat_g": 8, "sugar_g": 20, "note": "Sugared fried banana"},
    {"food": "Camote cue", "serving": "1 stick", "group": "Merienda", "carbs_g": 42, "protein_g": 2, "fat_g": 8, "sugar_g": 18, "note": "Sugared fried root crop"},
    {"food": "Lugaw/arroz caldo", "serving": "1 bowl", "group": "Merienda", "carbs_g": 30, "protein_g": 6, "fat_g": 4, "sugar_g": 1, "note": "Rice porridge"},
    {"food": "Champorado", "serving": "1 bowl", "group": "Merienda", "carbs_g": 45, "protein_g": 5, "fat_g": 6, "sugar_g": 20, "note": "Sweet cocoa porridge"},
    {"food": "Puto", "serving": "2 pieces", "group": "Merienda", "carbs_g": 22, "protein_g": 2.5, "fat_g": 1.5, "sugar_g": 8, "note": "Rice cake"},
    {"food": "Kutsinta", "serving": "2 pieces", "group": "Merienda", "carbs_g": 24, "protein_g": 1.5, "fat_g": 0.5, "sugar_g": 10, "note": "Rice snack"},
    {"food": "Biko", "serving": "1 small slice", "group": "Merienda", "carbs_g": 38, "protein_g": 3, "fat_g": 7, "sugar_g": 16, "note": "Sticky rice dessert"},
    {"food": "Halo-halo", "serving": "1 glass", "group": "Merienda", "carbs_g": 60, "protein_g": 6, "fat_g": 9, "sugar_g": 40, "note": "Dessert; high sugar"},
    {"food": "Milk tea", "serving": "1 cup", "group": "Sugary drinks", "carbs_g": 45, "protein_g": 2, "fat_g": 7, "sugar_g": 35, "note": "Depends on syrup/toppings"},
    {"food": "3-in-1 coffee", "serving": "1 sachet cup", "group": "Sugary drinks", "carbs_g": 12, "protein_g": 1, "fat_g": 3, "sugar_g": 10, "note": "Contains sugar/creamer"},
    {"food": "Softdrink", "serving": "330 ml can", "group": "Sugary drinks", "carbs_g": 35, "protein_g": 0, "fat_g": 0, "sugar_g": 35, "note": "Limit intake"},
]

DIETARY_FAQ = [
    ("How do I estimate grams if I do not weigh food?", "Use household measures (cup, spoon, piece) and choose the closest sample food in the guide."),
    ("Should I include mixed dishes?", "Yes. Break mixed dishes into main ingredients (rice, meat/fish, vegetables, oil)."),
    ("Do beverages count?", "Yes. Sweet drinks and milk-based drinks should be included in dietary estimates."),
    ("Do I enter raw or cooked amounts?", "Prefer cooked edible portions for consistency with typical intake reporting."),
    ("What if I am unsure of exact intake?", "Enter your best estimate based on usual intake over recent days."),
    ("How should I record merienda?", "Include all snacks and sweet drinks consumed between meals; they are part of total dietary intake."),
]


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
st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(220, 38, 38, 0.10), transparent 30%),
                    radial-gradient(circle at top right, rgba(37, 99, 235, 0.08), transparent 26%),
                    linear-gradient(180deg, #f8f5f2 0%, #f2ede8 45%, #fcfbfa 100%);
                color: #18212f;
            }
            #MainMenu,
            header,
            footer,
            .stAppHeader,
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"] {
                visibility: hidden !important;
                display: none !important;
                height: 0 !important;
            }
            .block-container { padding-top: 0 !important; }
            [data-testid="stAppViewContainer"] > .main {
                padding-top: 0 !important;
            }
            [data-testid="stAppViewContainer"] .main .block-container {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
            /* ── Landing ── */
            .landing-hero {
                background: linear-gradient(145deg, #0f172a 0%, #7f1d1d 55%, #1e3a5f 100%);
                color: #fff;
                padding: 6.75rem 2rem 5.75rem;
                text-align: center;
                position: relative;
                overflow: hidden;
                border-radius: 0 0 56px 56px;
                margin: -2.75rem -1rem 2.6rem;
                min-height: 29rem;
            }
            .landing-hero::before {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    radial-gradient(circle at 18% 45%, rgba(220,38,38,0.28) 0%, transparent 52%),
                    radial-gradient(circle at 82% 18%, rgba(37,99,235,0.22) 0%, transparent 44%);
                pointer-events: none;
            }
            .landing-hero h1 {
                font-size: 2.5rem; font-weight: 800; letter-spacing: -0.025em;
                margin: 0 0 0.9rem; position: relative;
                line-height: 1.2;
            }
            .landing-badge {
                display: inline-block;
                background: rgba(255,255,255,0.13);
                border: 1px solid rgba(255,255,255,0.24);
                border-radius: 999px;
                padding: 0.3rem 1.1rem;
                font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase;
                margin-bottom: 1.4rem; position: relative;
            }
            .landing-sub {
                font-size: 1.08rem; color: rgba(255,255,255,0.90);
                max-width: 760px; margin: 0 auto; line-height: 1.72; position: relative;
                text-align: center;
                display: block;
                width: 100%;
            }
            .feature-cards {
                display: flex; gap: 1.2rem; justify-content: center;
                flex-wrap: wrap; padding: 2.4rem 1rem 0.4rem;
            }
            .feature-card {
                background: rgba(255,255,255,0.82);
                border: 1px solid rgba(24,33,47,0.09);
                border-radius: 20px; padding: 1.55rem 1.6rem; width: 210px;
                box-shadow: 0 8px 30px rgba(15,23,42,0.07); text-align: center;
            }
            .feature-card .fc-icon { font-size: 2rem; margin-bottom: 0.55rem; }
            .feature-card .fc-title { font-weight: 700; font-size: 0.96rem; color: #0f172a; margin-bottom: 0.3rem; }
            .feature-card .fc-desc { font-size: 0.83rem; color: #475569; line-height: 1.5; }
            .landing-disclaimer {
                text-align: center; color: #94a3b8; font-size: 0.78rem;
                margin-top: 1.6rem; padding-bottom: 1.4rem;
            }
            /* ── App page ── */
            .glass-card {
                padding: 1rem 1.1rem; border-radius: 22px;
                border: 1px solid rgba(24,33,47,0.08);
                background: rgba(255,255,255,0.78);
                box-shadow: 0 18px 40px rgba(15,23,42,0.07);
                margin-bottom: 1rem;
            }
            .section-header {
                padding: 1.6rem 0 0.4rem;
                border-bottom: 2px solid rgba(220,38,38,0.18);
                margin-bottom: 1.3rem;
            }
            .section-header h2 { font-size: 1.55rem; font-weight: 700; margin: 0; color: #0f172a; }
            .section-header p { color: #64748b; margin: 0.25rem 0 0; font-size: 0.93rem; }
            .output-hero {
                background: linear-gradient(135deg, rgba(17,24,39,0.94), rgba(127,29,29,0.88));
                border-radius: 22px; padding: 1.4rem 1.8rem; color: #fff; margin-bottom: 0;
            }
            .output-hero .oh-label { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.15em; color: rgba(255,255,255,0.62); }
            .output-hero .oh-value { font-size: 2.1rem; font-weight: 800; margin-top: 0.1rem; }
            .output-hero .oh-sub { font-size: 0.87rem; color: rgba(255,255,255,0.70); margin-top: 0.15rem; }
            .risk-badge { display: inline-block; border-radius: 999px; padding: 0.32rem 1rem; font-size: 0.95rem; font-weight: 700; }
            .risk-low    { background: #dcfce7; color: #15803d; }
            .risk-medium { background: #fef9c3; color: #92400e; }
            .risk-high   { background: #fee2e2; color: #b91c1c; }
            .kpi-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.18em; color: #64748b; }
            .kpi-value { font-size: 1.8rem; font-weight: 700; margin-top: 0.15rem; color: #0f172a; }
            .kpi-subtitle { color: #64748b; font-size: 0.92rem; margin-top: 0.18rem; }
            section[data-testid="stSidebar"] {
                background: rgba(255,255,255,0.82);
                border-right: 1px solid rgba(15,23,42,0.08);
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


def _clean_label_text(text: str) -> str:
    primary = text.split(":", 1)[0].strip()
    return primary.replace("  ", " ")


def field_display_label(feature_name: str, dictionary_labels: dict[str, str]) -> str:
    for candidate in dictionary_name_candidates(feature_name):
        if candidate in DISPLAY_LABEL_OVERRIDES:
            return DISPLAY_LABEL_OVERRIDES[candidate]

    for candidate in dictionary_name_candidates(feature_name):
        if candidate in dictionary_labels and dictionary_labels[candidate]:
            return _clean_label_text(dictionary_labels[candidate])

    help_text = field_help_text(feature_name, dictionary_labels)
    if help_text:
        return _clean_label_text(help_text)

    fallback = feature_name.replace("epwt_", "").replace("_", " ").strip()
    return fallback.title()


MISSING_INPUT_CODES = {9.0, 99.0, 888888.0, 999999.0}


def is_missing_input_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    if np.isnan(numeric_value):
        return True
    return numeric_value in MISSING_INPUT_CODES


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


def render_dietary_quick_guide_popup():
    container = st.popover("Open Dietary Sample Foods & FAQ") if hasattr(st, "popover") else st.expander("Open Dietary Sample Foods & FAQ", expanded=False)
    with container:
        st.markdown("##### Common Food Samples (Web-informed)")
        st.caption("Compiled from public dietary guidance patterns (NHS Eatwell categories) and common household food portions.")
        st.dataframe(pd.DataFrame(DIETARY_COMMON_FOODS), use_container_width=True, hide_index=True)
        st.markdown("##### FAQ")
        for question, answer in DIETARY_FAQ:
            st.markdown(f"- **{question}** {answer}")


def render_number_input(feature_name: str, dictionary_labels: dict[str, str], dictionary_value_labels: dict[str, dict[str, str]]):
    minimum, maximum, step = feature_range(feature_name)
    default_value = feature_default(feature_name)
    help_text = field_help_text(feature_name, dictionary_labels)
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
            placeholder="Leave blank for missing",
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

    numeric_value = st.number_input(
        (
            f"{display_label} (Filipino Average {display_label} Value)"
            if np.isfinite(float(default_value))
            and np.isclose(float(st.session_state.get(widget_key, default_value)), float(default_value))
            else display_label
        ),
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(default_value),
        step=float(step),
        format="%.2f",
        key=widget_key,
        help=help_text,
    )

    return float(numeric_value)


def render_behavioral_selectors(dictionary_labels: dict[str, str], dictionary_value_labels: dict[str, dict[str, str]]):

    def _optional_code(variable_name: str, options: list[int], key: str, fallback_help: str):
        value_label_map = VALUE_LABEL_OVERRIDES.get(variable_name, dictionary_value_labels.get(variable_name, {}))
        variable_help = dictionary_labels.get(variable_name, fallback_help)

        option_values: list[int | None] = [None] + options

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

    binge_left, binge_mid, _ = st.columns(3)
    with binge_left:
        drnk_30days = _optional_code(
            variable_name="drnk_30days",
            options=[0, 1, 999999],
            key="raw_drnk_30days",
            fallback_help="Drank an alcoholic drink within the past 30 days.",
        )
    with binge_mid:
        binge_drink = _optional_code(
            variable_name="binge_drink",
            options=[0, 1, 99],
            key="raw_binge_drink",
            fallback_help="Binge-drink indicator used to set highest engineered level.",
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
                values["weight"] = st.number_input(
                    "Weight (Filipino Average Weight Value)" if np.isclose(float(st.session_state.get("raw_weight", 68.0)), 68.0) else "Weight",
                    min_value=20.0,
                    max_value=300.0,
                    value=68.0,
                    step=0.1,
                    key="raw_weight",
                    help=dictionary_labels.get("weight", "Average weight in kilograms."),
                )
            elif feature == "height":
                values["height"] = st.number_input(
                    "Height (Filipino Average Height Value)" if np.isclose(float(st.session_state.get("raw_height", 165.0)), 165.0) else "Height",
                    min_value=0.8,
                    max_value=260.0,
                    value=165.0,
                    step=0.1,
                    key="raw_height",
                    help=(dictionary_labels.get("height", "Average height.") + " Values >3 are treated as centimeters and converted to meters."),
                )
            elif feature == "waist":
                values["waist"] = st.number_input(
                    "Waist Circumference (Filipino Average Waist Circumference Value)" if np.isclose(float(st.session_state.get("raw_waist", 84.0)), 84.0) else "Waist Circumference",
                    min_value=30.0,
                    max_value=200.0,
                    value=84.0,
                    step=0.1,
                    key="raw_waist",
                    help=dictionary_labels.get("waist", "Average waist circumference."),
                )
            elif feature == "hip":
                values["hip"] = st.number_input(
                    "Hip Circumference (Filipino Average Hip Circumference Value)" if np.isclose(float(st.session_state.get("raw_hip", 96.0)), 96.0) else "Hip Circumference",
                    min_value=30.0,
                    max_value=200.0,
                    value=96.0,
                    step=0.1,
                    key="raw_hip",
                    help=dictionary_labels.get("hip", "Average hip circumference."),
                )

    return values


def make_indicator_chart(score: float, lower: float, upper: float):
    lower_pct = float(lower * 100.0)
    upper_pct = float(upper * 100.0)
    score_pct = float(score * 100.0)

    # Ensure the interval is visually readable when bounds nearly coincide.
    width = upper_pct - lower_pct
    if width < 2.0:
        pad = (2.0 - width) / 2.0
        lower_pct = max(0.0, lower_pct - pad)
        upper_pct = min(100.0, upper_pct + pad)

    # Auto-zoom: score ± 10 unless the uncertainty interval is wider than 20 pts.
    interval_range = upper_pct - lower_pct
    if interval_range <= 20.0:
        x_min = max(0.0, score_pct - 10.0)
        x_max = min(100.0, score_pct + 10.0)
    else:
        _pad_ext = 5.0
        x_min = max(0.0, lower_pct - _pad_ext)
        x_max = min(100.0, math.ceil((upper_pct + _pad_ext) / 10.0) * 10.0)

    dtick_val = 5 if (x_max - x_min) <= 25 else 10

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lower_pct, upper_pct],
            y=[0, 0],
            mode="lines",
            line=dict(color="#0f172a", width=12),
            hovertemplate="Interval: %{x:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lower_pct, upper_pct],
            y=[0, 0],
            mode="markers",
            marker=dict(color="#0f172a", size=10),
            hovertemplate="Bound: %{x:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[score_pct],
            y=[0],
            mode="markers",
            marker=dict(color="#dc2626", size=16, line=dict(color="white", width=2)),
            hovertemplate="Risk score: %{x:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=score_pct,
        y=0.28,
        text=f"{score_pct:.1f}%",
        showarrow=False,
        font=dict(size=12, color="#0f172a"),
    )
    fig.update_layout(
        height=210,
        margin=dict(l=10, r=10, t=16, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Probability (%)",
            range=[x_min, x_max],
            showgrid=True,
            gridcolor="rgba(100,116,139,0.25)",
            tickmode="linear",
            dtick=dtick_val,
            zeroline=False,
        ),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
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


def apply_dietary_derived_totals(widget_values: dict[str, float], feature_names: list[str]) -> dict[str, float]:
    values = dict(widget_values)

    food_group_names = [
        name
        for name in feature_names
        if (name.startswith("epwt_fg") or name.startswith("fg")) and name not in {"fg", "epwt_fg"}
    ]
    food_group_total = 0.0
    for name in food_group_names:
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
    pct = probability * 100.0
    if pct <= 33.0:
        return "Low Risk"
    if pct <= 66.0:
        return "Medium Risk"
    return "High Risk"


if "show_form" not in st.session_state:
    st.session_state.show_form = False
if "scored" not in st.session_state:
    st.session_state.scored = False


model_path = str(DEFAULT_MODEL_PATH)
calibrator_path = str(DEFAULT_CALIBRATOR_PATH)

with st.sidebar:
    st.subheader("Model artifacts")
    model_path = st.text_input("Stand-in model joblib", value=model_path)
    calibrator_path = st.text_input("Venn-Abers calibrator", value=calibrator_path)
    st.caption("Replace these paths when your final exported artifacts are ready.")
    st.divider()
    st.caption("Form is driven from the model's `feature_names_in_` metadata.")
    st.caption("Install `shap` and `lime` to enable explainability outputs.")


model, calibrator, feature_names = load_artifacts(model_path, calibrator_path)
grouped_features = group_feature_names(feature_names)
dictionary_labels, dictionary_value_labels = load_dataset_dictionaries()


if not st.session_state.show_form:
    st.markdown(
        """
        <div class="landing-hero">
          <div class="landing-badge">Philippine 2015 DOST-FNRI Thesis Model</div>
                    <h1>HRP-AI: A WEB APPLICATION FOR HYPERTENSION<br>RISK PREDICTION WITH CALIBRATED<br>EXPLAINABLE AI</h1>
          <p class="landing-sub" style="text-align:center;margin:0 auto;width:100%;">
            A thesis-focused decision support interface for estimating hypertension risk using dietary, anthropometric, and clinical inputs, with calibrated probabilities and transparent explainability outputs.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, cta_col, _ = st.columns([2, 1.2, 2])
    with cta_col:
        if st.button("Begin Assessment", use_container_width=True, type="primary"):
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
        clinical_features = [f for f in grouped_features["Core clinical"] if f not in engineered_names and f not in {"age", "sex", "uic", "pa_met", "vita"}]
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
        st.caption("Definitions for each dietary component are available in the field hover tooltips (?).")
        render_dietary_quick_guide_popup()
        st.markdown("##### Example ordinary servings (approximate)")
        st.caption("Use these examples as quick references before entering dietary values.")
        st.markdown(
            "- One cup cooked white rice ≈ 45 g carbohydrates (Total_CHO).\n"
            "- One egg ≈ 6 g protein and 5 g fat (Total_Prot, Total_Fat).\n"
            "- One medium banana ≈ 10 mg vitamin C and 27 g carbs (Total_VitC, Total_CHO).\n"
            "- One cup milk ≈ 300 mg calcium and 8 g protein (Total_Calc, Total_Prot).\n"
            "- One tablespoon cooking oil ≈ 14 g fat (Total_Fat)."
        )
        dietary_features = list(grouped_features["Dietary pattern"])
        if "vita" in feature_names and "vita" not in dietary_features:
            dietary_features.insert(0, "vita")

        dietary_cols = st.columns(3)
        _diet_col_idx = 0
        for feature_name in dietary_features:
            if feature_name in rendered_feature_names:
                continue
            # Skip auto-computed totals from input – show as live metrics below
            if feature_name in AUTO_COMPUTED_TOTAL_FIELDS:
                widget_values[feature_name] = float(feature_default(feature_name))
                rendered_feature_names.add(feature_name)
                continue
            with dietary_cols[_diet_col_idx % 3]:
                widget_values[feature_name] = render_number_input(feature_name, dictionary_labels, dictionary_value_labels)
                rendered_feature_names.add(feature_name)
                required_field_labels[feature_name] = field_display_label(feature_name, dictionary_labels)
            _diet_col_idx += 1

        # ── Live-computed dietary totals (auto-updates with every input change) ──
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

        st.divider()
        st.markdown("##### Computed Dietary Totals (updates as you fill in values above)")
        _tot_a, _tot_b, _tot_c = st.columns(3)
        with _tot_a:
            st.number_input("Total Food Intake (g)", value=round(_live_food, 1), disabled=True, key="_disp_food")
            st.caption("Sum of all food-group inputs above")
        with _tot_b:
            st.number_input("Total Energy (kcal)", value=round(_live_energy, 1), disabled=True, key="_disp_energy")
            st.caption("4 × carbs + 4 × protein + 9 × fat (estimated)")
        with _tot_c:
            st.number_input("Total Protein (g)", value=round(_live_protein, 1), disabled=True, key="_disp_protein")
            st.caption("Estimated from protein-heavy food groups")

    missing_field_labels = [
        label for feature_name, label in required_field_labels.items() if is_missing_input_value(widget_values.get(feature_name))
    ]
    if missing_field_labels:
        missing_preview = ", ".join(missing_field_labels[:8])
        if len(missing_field_labels) > 8:
            missing_preview += f", and {len(missing_field_labels) - 8} more"
        st.warning(f"Complete all missing inputs before predicting: {missing_preview}.")
        st.session_state.scored = False
        st.session_state.pop("_result", None)
        st.session_state.pop("_input_frame", None)

    st.markdown("<br>", unsafe_allow_html=True)
    _, submit_col, _ = st.columns([1.3, 1.2, 1.3])
    with submit_col:
        submitted = st.button("Predict My Risk", use_container_width=True, type="primary", disabled=bool(missing_field_labels))

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

                    function visibleFields() {
                        return Array.from(formContainer.querySelectorAll(selector)).filter((el) => el.offsetParent !== null);
                    }

                    visibleFields().forEach((el) => {
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

    if submitted and not missing_field_labels:
        derived_widget_values = apply_dietary_derived_totals(widget_values, feature_names)
        input_values = build_input_values_from_widgets(feature_names, derived_widget_values)
        input_frame = make_input_frame(feature_names, input_values)
        result = predict_with_venn_abers(model, input_frame, calibrator)
        st.session_state.scored = True
        st.session_state._result = result
        st.session_state._input_frame = input_frame
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
        tier_lower = mapped_risk_label.lower()
        tier_css = "risk-low" if "low" in tier_lower else ("risk-medium" if "medium" in tier_lower else "risk-high")

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
                f"<div class='output-hero'><div class='oh-label'>Risk Tier</div><div class='oh-value' style='font-size:1.4rem;'><span class='risk-badge {tier_css}'>{mapped_risk_label}</span></div><div class='oh-sub'>0-33% Low, 34-66% Medium, 67%+ High</div></div>",
                unsafe_allow_html=True,
            )

        chart_left, chart_right = st.columns([1, 1.5])
        with chart_left:
            st.markdown("**Risk Gauge**")
            st.plotly_chart(make_gauge_chart(result.calibrated_probability), use_container_width=True)
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
            st.dataframe(summary_frame, use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='section-header' style='margin-top:1.8rem;'>"
            "<h2>Explainability</h2>"
            "<p>Local explanations for this prediction.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Computing SHAP and LIME explanations..."):
            shap_local_df, shap_global_df, shap_error = _try_compute_shap(model, feature_names, input_frame)
            lime_df, lime_error = _try_compute_lime(model, feature_names, input_frame)

        exp_a, exp_b, exp_c = st.columns(3)

        with exp_a:
            st.markdown("##### SHAP Local")
            if shap_error:
                st.warning(shap_error)
            elif shap_local_df is not None and not shap_local_df.empty:
                local_top = shap_local_df.head(12)
                st.dataframe(local_top[["feature", "shap_value"]], use_container_width=True, hide_index=True)
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
                )
            else:
                st.info("No SHAP local output produced.")

        with exp_b:
            st.markdown("##### SHAP Global")
            if shap_error:
                st.warning(shap_error)
            elif shap_global_df is not None and not shap_global_df.empty:
                global_top = shap_global_df.head(12)
                st.dataframe(global_top, use_container_width=True, hide_index=True)
                chart_df = global_top.head(10).iloc[::-1]
                st.plotly_chart(
                    go.Figure(
                        go.Bar(
                            x=chart_df["mean_abs_shap"],
                            y=chart_df["feature"],
                            orientation="h",
                            marker_color="#7c3aed",
                        )
                    ).update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10)),
                    use_container_width=True,
                )
            else:
                st.info("No SHAP global output produced.")

        with exp_c:
            st.markdown("##### LIME Local")
            if lime_error:
                st.warning(lime_error)
            elif lime_df is not None and not lime_df.empty:
                lime_top = lime_df.head(12)
                st.dataframe(lime_top, use_container_width=True, hide_index=True)
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

