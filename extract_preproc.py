import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

keywords = ['ColumnTransformer', 'StandardScaler', 'OneHotEncoder', 'BMI', 'whr',
            'alcohol_level', 'smoking_level', 'feature_engineer', 'preprocess',
            'Pipeline', 'SimpleImputer', 'KNNImputer', 'drop_duplicates',
            'fillna', 'FEATURES', 'feature_cols', 'selected_features', 'X_train',
            'make_column', 'numeric_features', 'categorical_features',
            'build_smoking', 'build_alcohol', 'build_bmi', 'build_whr',
            'fe_smoking', 'fe_alcohol', 'OHE', 'ohe', 'scaler', 'transformer']

for nb_file in ['EXP3_A_KNN_RF.ipynb', 'EXP3_B_XGB_AdaBoost.ipynb', 'EXP3_C_LogReg_CatBoost.ipynb']:
    print(f'\n\n{"="*60}')
    print(f'NOTEBOOK: {nb_file}')
    print('='*60)
    with open(nb_file, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        if any(k.lower() in src.lower() for k in keywords):
            print(f'\n--- Cell {i} ---')
            print(src[:5000])
