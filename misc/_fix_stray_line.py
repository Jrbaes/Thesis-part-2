import json

nb_path = '/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb'
nb = json.loads(open(nb_path).read())

OLD = """            else:
                raise
            model_name, params, use_cw, yf_tr_s, epoch_budget, seed)
        p_val"""

NEW = """            else:
                raise
        p_val"""

changed = False
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if OLD in s:
        s = s.replace(OLD, NEW)
        c['source'] = s.splitlines(keepends=True)
        changed = True
        print(f"Fixed stray line in Cell #{i}")
        break

if not changed:
    print("ERROR: Pattern not found!")
else:
    open(nb_path, 'w').write(json.dumps(nb, ensure_ascii=False, indent=1))
    print("SAVED")
