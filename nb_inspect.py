import json
keywords = ['calibration_df', 'calibrated_model_artifacts', 'calibrated_test_probs',
            'metric_pack', 'apply_with_calibrator', 'safe_predict_proba',
            'expected_calibration_error', 'X_test_final', 'y_test']
with open(r'c:\Jon\College\Thesis\V2.2.1.1\Main_2015_GPU_RF_XGB_CAT_RIGOROUS_OPT_exp2.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
print('Total cells:', len(cells))
for i, c in enumerate(cells, 1):
    src = ''.join(c['source'])
    hits = [l.strip() for l in src.splitlines() if any(k in l for k in keywords)]
    if hits:
        print(f'\n--- Cell {i} (id={c["id"]}) ---')
        for h in hits[:8]:
            print(' ', h)
