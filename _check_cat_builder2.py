import json

nb_c = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for i, c in enumerate(nb_c['cells']):
    s = ''.join(c.get('source', []))
    if 'e3_build_model_for_search' in s and 'def ' in s:
        print(f"Cell #{i} total length: {len(s)}")
        # Search for catboost
        idx = 0
        while True:
            cat_idx = s.find('catboost', idx)
            if cat_idx < 0:
                break
            print(f"  catboost at pos {cat_idx}: ...{s[max(0,cat_idx-30):cat_idx+80]}...")
            idx = cat_idx + 1
        break
