import json, re

# List ALL cells in EXP3_C that mention catboost or E3_MODEL_SPACES
nb = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
print("EXP3_C all cells with catboost or E3_MODEL_SPACES:")
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if 'catboost' in s.lower() and ('E3_MODEL_SPACES' in s or 'MODEL_SPACE' in s or "catboost'" in s):
        print(f"\nCell #{i} ID:{c.get('id','N/A')} len={len(s)}")
        # Show context around catboost
        idx = s.lower().find("'catboost'")
        if idx >= 0:
            print("  catboost context:", repr(s[max(0,idx-20):idx+400]))
