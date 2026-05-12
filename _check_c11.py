import json, re

# Check EXP3_C Cell 11 specifically (by finding the cell with ONLY logreg+catboost)
nb = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
print("EXP3_C E3_MODEL_SPACES cells:")
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if "'logreg'" in s and "'catboost'" in s and 'E3_MODEL_SPACES =' in s:
        idx = s.find('E3_MODEL_SPACES =')
        end_idx = s.find('\n}\n', idx) + 4
        print(f"  Cell #{i} ID:{c.get('id','N/A')}")
        print(repr(s[idx:min(end_idx, idx+2000)]))
        print("---")
