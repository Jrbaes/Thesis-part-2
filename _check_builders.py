import json, re

# Check EXP3_A xgboost full block
nb_a = json.loads(open('/workspace/Thesis-part-2/EXP3_A_KNN_RF copy.ipynb').read())
for c in nb_a['cells']:
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES =' in s and "'xgboost'" in s:
        idx = s.find("    'xgboost': {")
        end = s.find("    },", idx) + 6
        print("EXP3_A xgboost block:")
        print(s[idx:end])
        break

# Check EXP3_C builder function (e3_build_model_for_search)
print("\n" + "="*60)
print("EXP3_C builder function (cell with e3_build_model_for_search):")
nb_c = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for i, c in enumerate(nb_c['cells']):
    s = ''.join(c.get('source', []))
    if 'e3_build_model_for_search' in s and 'def ' in s:
        # Show catboost section of builder
        cat_idx = s.lower().find('catboost')
        lgb_idx = s.lower().find('lightgbm')
        xgb_idx = s.lower().find('xgboost')
        print(f"Cell #{i}")
        if cat_idx >= 0:
            print(f"\n  CatBoost builder section (context):")
            print(s[max(0,cat_idx-50):cat_idx+300])
        break

# Check LightGBM builder (_build_lgb)
print("\n" + "="*60)
print("EXP3_C _build_lgb function:")
for i, c in enumerate(nb_c['cells']):
    s = ''.join(c.get('source', []))
    if '_build_lgb' in s and 'def ' in s:
        idx = s.find('def _build_lgb')
        end_idx = s.find('\ndef ', idx + 1)
        print(s[idx:end_idx if end_idx > 0 else idx+600])
        break
