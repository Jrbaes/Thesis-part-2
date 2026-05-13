import json

nb_c = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for i, c in enumerate(nb_c['cells']):
    s = ''.join(c.get('source', []))
    if 'CatBoostClassifier' in s:
        print(f"Cell #{i} has CatBoostClassifier:")
        idx = 0
        while True:
            pos = s.find('CatBoostClassifier', idx)
            if pos < 0: break
            print(f"  pos {pos}: ...{s[max(0,pos-100):pos+200]}...")
            idx = pos + 1
        print()
