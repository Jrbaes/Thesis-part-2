import json

nb_path = '/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb'
nb = json.loads(open(nb_path).read())

OLD_LGB = """        reg_lambda       = max(0.0, float(p.get('reg_lambda', 1.0))),
        scale_pos_weight = pos_w if use_cw else 1.0,"""

NEW_LGB = """        reg_lambda       = max(0.0, float(p.get('reg_lambda', 1.0))),
        reg_alpha        = max(0.0, float(p.get('reg_alpha', 0.0))),
        min_child_samples= max(1, int(round(p.get('min_child_samples', 20)))),
        scale_pos_weight = pos_w if use_cw else 1.0,"""

changed = False
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if OLD_LGB in s:
        s = s.replace(OLD_LGB, NEW_LGB)
        c['source'] = s.splitlines(keepends=True)
        changed = True
        print(f"Fixed _build_lgb in Cell #{i}")
        break

if not changed:
    print("ERROR: Could not find pattern!")
else:
    open(nb_path, 'w').write(json.dumps(nb, ensure_ascii=False, indent=1))
    print("SAVED")
