import json

nb_c = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())

for i, c in enumerate(nb_c['cells']):
    s = ''.join(c.get('source', []))
    if 'e3_build_model_for_search' not in s or 'def ' not in s:
        continue
    # Show adaboost and catboost sections
    for mname in ['adaboost', 'catboost']:
        idx = s.find(f"if model_name == '{mname}'")
        if idx >= 0:
            end_idx = s.find("\n    if model_name", idx + 1)
            if end_idx < 0:
                end_idx = idx + 600
            print(f"\n--- {mname} builder (Cell #{i}) ---")
            print(s[idx:end_idx])
    break

print("\n\n--- _build_lgb ---")
for i, c in enumerate(nb_c['cells']):
    s = ''.join(c.get('source', []))
    if 'def _build_lgb' in s:
        idx = s.find('def _build_lgb')
        end_idx = s.find('\ndef ', idx + 1)
        print(s[idx:end_idx if end_idx > 0 else idx+600])
        break
