import json, re

# EXP3_B check
nb = json.loads(open('/workspace/Thesis-part-2/EXP3_B_XGB_AdaBoost copy.ipynb').read())
for c in nb['cells']:
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES' in s and "'xgboost'" in s:
        idx = s.find('E3_MODEL_SPACES')
        print("EXP3_B Cell ID:", c.get('id','N/A'))
        print(repr(s[idx:idx+1000]))
        print("---")
        break

# EXP3_C check
nb = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for c in nb['cells']:
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES' in s and "'catboost'" in s and "'logreg'" in s:
        idx = s.find('E3_MODEL_SPACES')
        print("EXP3_C Cell ID:", c.get('id','N/A'))
        print(repr(s[idx:idx+1500]))
        print("---")
        break

# EXP3_C LightGBM check
for c in nb['cells']:
    s = ''.join(c.get('source', []))
    if '_LGB_SPACES' in s and 'num_leaves' in s:
        idx = s.find('_LGB_SPACES')
        print("EXP3_C LGB Cell ID:", c.get('id','N/A'))
        print(repr(s[idx:idx+600]))
        print("---")
        break
