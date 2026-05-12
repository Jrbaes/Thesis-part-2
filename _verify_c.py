import json, re

nb = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES =' in s:
        idx = s.find('E3_MODEL_SPACES =')
        end_idx = s.find('\n}\n', idx) + 4
        print(f"EXP3_C Cell #{i} E3_MODEL_SPACES:")
        print(s[idx:end_idx if end_idx > 4 else idx+3000])
        break

for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if '_LGB_SPACES =' in s:
        idx = s.find('_LGB_SPACES =')
        end_idx = s.find('\n}\n', idx) + 4
        print(f"\nEXP3_C Cell #{i} _LGB_SPACES:")
        print(s[idx:end_idx if end_idx > 4 else idx+400])
        break
