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
    "fg1": "Cereals and Cereal Products (in grams): include rice and rice products, corn and corn products, and other cereal products.",
    "fg2": "Rice and Rice Products (in grams): include rice (ordinary, special, glutinous) and rice products like bihon, puto, biko, suman, arroz caldo, champorado, and others.",
    "fg3": "Corn and Corn Products (in grams): refer to milled corn, corn on cob, and products like cornstarch, maja blanca, popcorn, corn chips, and others.",
    "fg4": "Other Cereal Products (in grams): refer to pandesal, bread, cookies/biscuits, cakes/pastries, noodles, flour, and others.",
    "fg5": "Starchy Roots and Tubers (in grams): consist of sweet potatoes and products, potatoes and products, cassava and products, and other roots/tubers such as yam, taro, arrowroot, and yakon.",
    "fg6": "Sugar and Syrups (in grams): consist of refined/second class/brown/crude sugars, jams, candies, honey, sweetened soda, sherbet, ice candy, chocolates, and others.",
    "fg7": "Dried Beans, Nuts and Seeds (in grams): include mungbeans, soybeans, nuts, and other dried beans/seeds and products like almond, peas, garbanzos, sesame seed, patani, mani, taho/tofu/tokwa, and others.",
    "fg8": "Vegetables (in grams): refer to green leafy and yellow vegetables and other vegetables.",
    "fg9": "Green Leafy and Yellow Vegetables (in grams): include camote tops, kangkong, malunggay, alugbati, pechay, squash fruit/flower, carrot, and other yellow vegetables.",
    "fg10": "Other Vegetables (in grams): include eggplant, stringbeans, abitsuelas, ampalaya, wild vegetables, and canned/processed vegetables like canned mushroom and pickled cucumber.",
    "fg11": "Fruits (in grams): include vitamin C-rich fruits and other fruits.",
    "fg12": "Vitamin C-Rich Fruits (in grams): include mangoes, papaya, citrus fruits, strawberry, guava, and others.",
    "fg13": "Other Fruits (in grams): include bananas, watermelon, melon, jackfruit, pineapple, young coconut, kaimito, and others.",
    "fg14": "Fish, Meat and Poultry (in grams): refer to fresh fish, dried fish, processed fish, crustaceans and mollusks, fresh meat, organ meat, processed meat, poultry, and others.",
    "fg15": "Fish and Fish Products (in grams): refer to fresh fish (tulingan, bangus, galunggong, tilapia, and others), dried fish, processed fish (bagoong isda, patis, canned/smoked fish), and crustaceans/mollusks.",
    "fg16": "Meat and Meat Products (in grams): refer to fresh meat (pork, beef, carabeef, and others), organ meat, and processed meat (hotdog, longganisa, tocino, ham, meat loaf, and others).",
    "fg17": "Poultry (in grams): refer to chicken and other fowls like duck, goose, pigeon, turkey, and others.",
    "fg18": "Eggs (in grams): include hen egg, duck egg, and other eggs like ant egg, quail egg, turkey egg, and others.",
    "fg19": "Milk and Milk Products (in grams): consist of fresh whole milk, evaporated milk, recombined milk, powdered milk, condensed milk, and other milk products.",
    "fg20": "Whole Milk (in grams): consist of fresh whole milk, evaporated milk, recombined milk, powdered milk (infant formula, whole/full cream, filled, skimmed), and condensed milk.",
    "fg21": "Milk Products (in grams): refer to cheese and other milk products like ice cream, yogurt, cultured milk, and others.",
    "fg23": "Fats and Oils (in grams): refer to cooking oil, coconut meat, coconut cream, pork drippings/lard, butter, margarine, peanut butter, and others.",
    "fg24": "Miscellaneous (in grams): include beverages, condiments/spices, and other miscellaneous items.",
    "fg25": "Beverages (in grams): consist of coffee, tea, alcoholic beverages, cacao/chocolate-based beverages, fruit flavored drinks, and others.",
    "fg26": "Condiments and Spices (in grams): consist of salt, vinegar, catsup, and other seasonings.",
    "fg27": "Other Miscellaneous (in grams): include lemon grass, laurel leaves, oregano, turmeric, food coloring, and others.",
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
    "fg14": "Fish, Meat and Poultry (in grams)",
    "epwt_fg14": "Fish, Meat and Poultry (in grams)",
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
    14: [15, 16, 17, 18],
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


FEATURE_UNITS = {
    "age": "years",
    "weight": "kg",
    "height": "cm",
    "waist": "cm",
    "hip": "cm",
    "BMI": "kg/m^2",
    "bmi": "kg/m^2",
    "whr": "ratio",
    "hemoglobin": "g/dL",
    "Total_FoodIntake": "g/day",
    "Total_Food_epwt": "g/day",
    "Total_Energy": "kcal/day",
    "Total_Ener": "kcal/day",
    "Total_Protein": "g/day",
    "Total_Prot": "g/day",
    "Total_CHO": "g/day",
    "Total_Fat": "g/day",
    "Total_Calcium": "mg",
    "Total_Calc": "mg",
    "Total_Iron": "mg",
    "Total_VitaminA": "mcg RE/day",
    "Total_VitA": "mcg RE/day",
    "Total_VitaminC": "mg",
    "Total_VitC": "mg",
    "Total_Thiamin": "mg",
    "Total_Thia": "mg",
    "Total_Riboflavin": "mg",
    "Total_Ribo": "mg",
    "Total_Niacin": "mg",
    "Total_Nia": "mg",
}


# Do not append a redundant unit to food-group labels because
# the dictionary names already include "(in grams)".
