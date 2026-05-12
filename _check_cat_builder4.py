import json

nb_c = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for i, c in enumerate(nb_c['cells']):
    s = ''.join(c.get('source', []))
    if i != 9:
        continue
    # Find the catboostclassifier return and show context
    pos = s.find('CatBoostClassifier(**kwargs)')
    if pos >= 0:
        print(f"Context around CatBoostClassifier(**kwargs):")
        print(s[max(0,pos-600):pos+100])
    break
