import json

nb_path = '/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb'
nb = json.loads(open(nb_path).read())

# The corrupted adaboost block (which incorrectly builds CatBoostClassifier)
OLD_ADABOOST_BLOCK = """    if model_name == 'adaboost':
        base_d = max(1, int(round(p.get('base_depth', 1))))
        base   = DecisionTreeClassifier(
            max_depth=base_d, random_state=seed,
            class_weight='balanced' if use_cw else None)
        kwargs = dict(
            n_estimators=max(10, int(round(p.get('n_estimators', epoch_budget)))),
            learning_rate=max(1e-4, float(p.get('learning_rate', 0.1))),
            min_data_in_leaf=max(1, int(p.get('min_data_in_leaf', 1))),
            bagging_temperature=max(0.0, float(p.get('bagging_temperature', 1.0))),
            loss_function='Logloss', eval_metric='Logloss',
            random_seed=seed, verbose=0,
        )
        if use_cw:  kwargs['auto_class_weights'] = 'Balanced'
        if use_gpu: kwargs['task_type'] = 'GPU'; kwargs['devices'] = '0'
        return CatBoostClassifier(**kwargs)

    if model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=int(epoch_budget),
            learning_rate=max(1e-4, float(p.get('learning_rate', 0.1))),
            max_depth=max(1, int(round(p.get('max_depth', 5)))),
            subsample=float(np.clip(p.get('subsample', 0.8), 0.1, 1.0)),
            colsample_bytree=float(np.clip(p.get('colsample_bytree', 0.8), 0.1, 1.0)),
            min_child_weight=max(1, int(round(p.get('min_child_weight', 1)))),
            gamma=max(0.0, float(p.get('gamma', 0.0))),
            reg_lambda=max(1e-4, float(p.get('reg_lambda', 1.0))),
            objective='binary:logistic', eval_metric='logloss',
            random_state=seed, tree_method='hist',
            device='cuda' if use_gpu else 'cpu',
            verbosity=0,
            scale_pos_weight=pos_w if use_cw else 1.0,
        )"""

NEW_BLOCKS = """    if model_name == 'adaboost':
        base_d = max(1, int(round(p.get('base_depth', 1))))
        base   = DecisionTreeClassifier(
            max_depth=base_d, random_state=seed,
            class_weight='balanced' if use_cw else None)
        return AdaBoostClassifier(
            estimator=base,
            n_estimators=max(10, int(round(p.get('n_estimators', epoch_budget)))),
            learning_rate=max(1e-4, float(p.get('learning_rate', 0.1))),
            random_state=seed,
        )

    if model_name == 'catboost':
        kwargs = dict(
            iterations=max(10, int(epoch_budget)),
            learning_rate=max(1e-4, float(p.get('learning_rate', 0.1))),
            depth=max(1, int(round(p.get('depth', 6)))),
            l2_leaf_reg=max(1e-4, float(p.get('l2_leaf_reg', 3.0))),
            random_strength=max(0.0, float(p.get('random_strength', 1.0))),
            bagging_temperature=max(0.0, float(p.get('bagging_temperature', 1.0))),
            loss_function='Logloss', eval_metric='Logloss',
            random_seed=seed, verbose=0,
        )
        if use_cw:  kwargs['auto_class_weights'] = 'Balanced'
        if use_gpu: kwargs['task_type'] = 'GPU'; kwargs['devices'] = '0'
        return CatBoostClassifier(**kwargs)

    if model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=int(epoch_budget),
            learning_rate=max(1e-4, float(p.get('learning_rate', 0.1))),
            max_depth=max(1, int(round(p.get('max_depth', 5)))),
            subsample=float(np.clip(p.get('subsample', 0.8), 0.1, 1.0)),
            colsample_bytree=float(np.clip(p.get('colsample_bytree', 0.8), 0.1, 1.0)),
            min_child_weight=max(1, int(round(p.get('min_child_weight', 1)))),
            gamma=max(0.0, float(p.get('gamma', 0.0))),
            reg_lambda=max(1e-4, float(p.get('reg_lambda', 1.0))),
            reg_alpha=max(0.0, float(p.get('reg_alpha', 0.0))),
            objective='binary:logistic', eval_metric='logloss',
            random_state=seed, tree_method='hist',
            device='cuda' if use_gpu else 'cpu',
            verbosity=0,
            scale_pos_weight=pos_w if use_cw else 1.0,
        )"""

changed = False
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if OLD_ADABOOST_BLOCK in s:
        s = s.replace(OLD_ADABOOST_BLOCK, NEW_BLOCKS)
        c['source'] = s.splitlines(keepends=True)
        changed = True
        print(f"Fixed builder in Cell #{i}")
        break

if not changed:
    print("ERROR: Could not find the old adaboost block pattern!")
    # Try partial search
    for i, c in enumerate(nb['cells']):
        s = ''.join(c.get('source', []))
        if 'min_data_in_leaf' in s and 'adaboost' in s:
            print(f"  Found 'min_data_in_leaf' + 'adaboost' in Cell #{i}")
else:
    open(nb_path, 'w').write(json.dumps(nb, ensure_ascii=False, indent=1))
    print("SAVED")
