import json, uuid

new_cell = {
    'cell_type': 'code',
    'id': uuid.uuid4().hex[:8],
    'metadata': {},
    'source': [
        '# All Calibrated Models - Full Test-Set Evaluation\n',
        '_ok_df = calibration_df[calibration_df["error"].isna()] if "error" in calibration_df.columns else calibration_df\n',
        '\n',
        '_rows = []\n',
        'for _, row in _ok_df.iterrows():\n',
        '    mdl, mth, thr = row["model"], row["method"], float(row["selected_threshold"])\n',
        '    key = (mdl, mth)\n',
        '\n',
        '    if key in calibrated_test_probs:\n',
        '        p_test = calibrated_test_probs[key]\n',
        '    elif key in calibrated_model_artifacts:\n',
        '        art    = calibrated_model_artifacts[key]\n',
        '        p_base = safe_predict_proba(art["base_model"], X_test_final[art["feature_names"]])\n',
        '        p_test = apply_with_calibrator(mth, art["calibrator"], p_base)\n',
        '    else:\n',
        '        print(f"skipping {mdl}/{mth} - artifact not in memory")\n',
        '        continue\n',
        '\n',
        '    m = metric_pack(y_test, p_test, threshold=thr)\n',
        '    _rows.append({\n',
        '        "model"      : mdl,\n',
        '        "calibration": mth,\n',
        '        "threshold"  : thr,\n',
        '        "accuracy"   : round(m["accuracy"],  4),\n',
        '        "recall"     : round(m["recall"],    4),\n',
        '        "precision"  : round(m["precision"], 4),\n',
        '        "f1"         : round(m["f1"],        4),\n',
        '        "auc"        : round(m["auc"],       4),\n',
        '        "logloss"    : round(m["logloss"],   4),\n',
        '        "ece"        : round(expected_calibration_error(np.asarray(y_test), p_test), 4),\n',
        '    })\n',
        '\n',
        'all_test_df = pd.DataFrame(_rows).sort_values("auc", ascending=False).reset_index(drop=True)\n',
        'all_test_df.index += 1\n',
        '\n',
        'print(f"Evaluated {len(all_test_df)} calibrated model/method combinations on test set\\n")\n',
        'display(all_test_df)\n',
        '\n',
        '_out = ARTIFACT_DIR / "all_calibrated_test_results.csv"\n',
        'all_test_df.to_csv(_out, index=True, index_label="test_rank")\n',
        'print(f"Saved: {_out}")\n',
    ],
    'outputs': [],
    'execution_count': None
}

with open(r'c:\Jon\College\Thesis\V2.2.1.1\Main_2015_GPU_RF_XGB_CAT_RIGOROUS_OPT_exp2.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].append(new_cell)

with open(r'c:\Jon\College\Thesis\V2.2.1.1\Main_2015_GPU_RF_XGB_CAT_RIGOROUS_OPT_exp2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Done. Total cells:', len(nb['cells']), ' new id:', new_cell['id'])
