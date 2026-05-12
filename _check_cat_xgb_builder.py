import json

nb_c = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for i, c in enumerate(nb_c['cells']):
    s = ''.join(c.get('source', []))
    if 'e3_build_model_for_search' in s and 'def ' in s:
        # Find catboost section
        cat_idx = s.find("if model_name == 'catboost'")
        if cat_idx < 0:
            cat_idx = s.find("catboost")
        end_idx = s.find("\n    if model_name", cat_idx + 1)
        if end_idx < 0:
            end_idx = cat_idx + 800
        print("CATBOOST BUILDER SECTION:")
        print(s[cat_idx:end_idx])
        
        # Find xgboost section
        xgb_idx = s.find("if model_name == 'xgboost'")
        end_xgb = s.find("\n    if model_name", xgb_idx + 1)
        if end_xgb < 0:
            end_xgb = xgb_idx + 800
        print("\nXGBOOST BUILDER SECTION:")
        print(s[xgb_idx:end_xgb])
        break
