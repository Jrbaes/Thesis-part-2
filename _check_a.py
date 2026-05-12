import json
nb = json.loads(open('/workspace/Thesis-part-2/EXP3_A_KNN_RF copy.ipynb').read())
for c in nb['cells']:
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES' in s and "'knn'" in s:
        idx = s.find('E3_MODEL_SPACES')
        print("Cell ID:", c.get('id','N/A'))
        print(repr(s[idx:idx+1500]))
        print("---")
