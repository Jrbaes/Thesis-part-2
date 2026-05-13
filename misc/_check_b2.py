import json, re

def check_nb(path, label):
    nb = json.loads(open(path).read())
    cells_found = []
    for i, c in enumerate(nb['cells']):
        s = ''.join(c.get('source', []))
        if 'E3_MODEL_SPACES' in s and 'E3_MODEL_SPACES =' in s:
            idx = s.find('E3_MODEL_SPACES =')
            end = s.find('\n}\n', idx) + 4
            print(f"\n{label} Cell #{i} ID:{c.get('id','N/A')}")
            print(repr(s[idx:end if end > 4 else idx+1200]))

check_nb('/workspace/Thesis-part-2/EXP3_B_XGB_AdaBoost copy.ipynb', 'EXP3_B')
