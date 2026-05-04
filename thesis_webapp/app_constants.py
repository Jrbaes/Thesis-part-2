from __future__ import annotations

from pathlib import Path


def get_dictionary_paths(project_root: Path) -> list[Path]:
    return [project_root / "Datasets2015" / "Clinical" / "Jonathan Ralph_Baes_2026-03-26141903_data-dictionary_clinical.csv", project_root / "Datasets2015" / "Dietary" / "Jonathan Ralph_Baes_2026-03-26141801_data-dictionary_dietary.csv", project_root / "Datasets2015" / "Anthropometric" / "Jonathan Ralph_Baes_2026-03-26141834_data-dictionary_anthrop.csv"]


FEATURE_DICTIONARY_ALIASES = {"Total_Ener": ["Total_Energy"], "Total_Prot": ["Total_Protein"], "Total_Calc": ["Total_Calcium"], "Total_VitA": ["Total_VitaminA"], "Total_VitC": ["Total_VitaminC"], "Total_Thia": ["Total_Thiamin"], "Total_Ribo": ["Total_Riboflavin"], "Total_Nia": ["Total_Niacin"], "Total_Food_epwt": ["Total_FoodIntake"]}


VARIABLE_DEFINITION_OVERRIDES = {
    "age": "Age of Respondent: exact age as of last birthday.",
    "sex": "Sex of Respondent: sex of household member.",
    "ethnicity": "Ethnicity code: 0 Not IP/without foreign blood, 1 Indigenous People, 2 2/3 Filipino, 3 with 1/2 foreign blood.",
    "current_smoking": "Presently smoke cigarettes/cigars/pipes/tobacco products: current smoking frequency code.",
    "ever_smk": "Ever smoked in the past: former smoking behavior code.",
    "alcohol": "Ever consumed alcoholic drink such as beer, wine, or spirits.",
    "con_alcohol": "Consumed alcoholic drink within the past 12 months (current drinkers).",
    "drnk_30days": "Consumed alcoholic drink within the past 30 days.",
    "smoke_status": "Smoking Status (Generated): 0 Never, 1 Current, 2 Former, 9 Not Applicable.",
    "alcohol_status": "Alcohol Status (Generated): 0 Never, 1 Current, 2 Former, 9 Not Applicable.",
    "binge_drink": "Binge Drinking Status (Generated): female >=4 standard drinks in a row; male >=5, among those who drank in past 30 days.",
    "weight": "Ave Weight (kg): body heaviness from muscle, fat, bone, organs and related conditions.",
    "height": "Ave Height (cm): standing height (or recumbent length for very young children in source survey).",
    "waist": "Ave Waist Circumference (cm): perimeter around natural waist/abdomen.",
    "hip": "Ave Hip Circumference (cm): distance around largest hip/buttocks area.",
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
    "Total_FoodIntake": "Total Food Intake (g): total intake across 27 food groups.",
    "Total_Energy": "Total Energy (kcal): total energy intake.",
    "Total_Protein": "Total Protein (g): total protein intake.",
    "Total_Calcium": "Total Calcium (mg): total calcium intake.",
    "Total_Iron": "Total Iron (mg): total iron intake.",
    "Total_VitaminA": "Total Vitamin A (mcg RE).",
    "Total_Thiamin": "Total Thiamin (mg).",
    "Total_Riboflavin": "Total Riboflavin (mg).",
    "Total_Niacin": "Total Niacin (mg).",
    "Total_VitaminC": "Total Vitamin C (mg).",
    "Total_CHO": "Total Carbohydrates (g): total carbohydrate intake.",
    "Total_Fat": "Total Fats (g): total fat intake.",
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
}


DISPLAY_LABEL_OVERRIDES = {
    "age": "Age",
    "sex": "Sex",
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
    "Total_FoodIntake", "Total_Food_epwt", "Total_Energy", "Total_Ener", "Total_Protein", "Total_Prot"
}


FOOD_GROUP_COMPONENT_TOTALS = {
    1: [2, 3, 4],
    8: [9, 10],
    11: [12, 13],
    14: [15, 16],
    19: [20, 21],
    24: [25, 26, 27],
}


MISSING_INPUT_CODES = {9.0, 99.0, 888888.0, 999999.0}

CONDITIONALLY_ALLOWED_NA_CODES: dict[str, set[float]] = {
    "ever_smk": {888888.0},
    "current_smoking": {888888.0},
    "con_alcohol": {999999.0},
    "drnk_30days": {999999.0},
    "binge_drink": {99.0},
}


NO_HELP_FEATURES = {"Total_VitA", "Total_VitC", "Total_Thia", "Total_Ribo", "Total_Nia", "Total_VitaminA", "Total_VitaminC", "Total_Thiamin", "Total_Riboflavin", "Total_Niacin"}
