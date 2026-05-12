import json, re

# EXP3_C check
nb = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
print("EXP3_C cells with E3_MODEL_SPACES:")
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES' in s and 'E3_MODEL_SPACES =' in s:
        idx = s.find('E3_MODEL_SPACES =')
        end_idx = s.find('\n}\n', idx) + 4
        if end_idx < 4:
            end_idx = idx + 2000
        print(f"  Cell #{i} ID:{c.get('id','N/A')}")
        print(repr(s[idx:end_idx]))
        print("---")

print("\nEXP3_A cells with E3_MODEL_SPACES:")
nb = json.loads(open('/workspace/Thesis-part-2/EXP3_A_KNN_RF copy.ipynb').read())
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES' in s and 'E3_MODEL_SPACES =' in s:
        idx = s.find('E3_MODEL_SPACES =')
        end_idx = s.find('\n}\n', idx) + 4
        if end_idx < 4:
            end_idx = idx + 2000
        print(f"  Cell #{i} ID:{c.get('id','N/A')}")
        print(repr(s[idx:end_idx]))
        print("---")
