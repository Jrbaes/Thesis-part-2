import json, re

def fix_notebook(nb_path, label):
    nb = json.loads(open(nb_path).read())
    changed = False

    for i, c in enumerate(nb['cells']):
        s = ''.join(c.get('source', []))
        if 'E3_MODEL_SPACES =' not in s:
            continue

        print(f"\n{label} Cell #{i} (ID: {c.get('id','?')}): processing E3_MODEL_SPACES")

        # ── Fix 1: Fix corrupted adaboost (catboost params injected inside it) ──
        # Pattern: adaboost block contains catboost-like keys (learning_rate duplicate,
        # depth, l2_leaf_reg, random_strength, min_data_in_leaf, bagging_temperature)
        corruption_pattern = re.compile(
            r"('adaboost'\s*:\s*\{[^}]*?'base_depth'\s*:[^,\n]+,\s*)\n"
            r"(\s*'learning_rate'\s*:\s*loguniform.*?\n"
            r"\s*'depth'\s*:.*?\n"
            r"(?:\s*'l2_leaf_reg'\s*:.*?\n)?"
            r"(?:\s*'random_strength'\s*:.*?\n)?"
            r"(?:\s*'min_data_in_leaf'\s*:.*?\n)?"
            r"(?:\s*'bagging_temperature'\s*:.*?\n)?"
            r"\s*\},)",
            re.DOTALL
        )
        m = corruption_pattern.search(s)
        if m:
            print(f"  Found adaboost corruption at pos {m.start()}-{m.end()}")
            # Replace with clean adaboost block closing
            clean_ada_end = m.group(1).rstrip() + "\n    },"
            s = s[:m.start()] + clean_ada_end + s[m.end():]
            print(f"  Removed catboost params from adaboost block")
            changed = True

        # ── Fix 2: Ensure 'catboost' key exists in E3_MODEL_SPACES ──
        # Check if catboost key exists
        if "'catboost'" not in s or "E3_MODEL_SPACES" not in s:
            pass
        else:
            # Check if catboost is INSIDE adaboost (corrupted) vs proper key
            em_idx = s.find("E3_MODEL_SPACES =")
            end_em = s.find("\n}\n", em_idx)
            em_section = s[em_idx:end_em+3] if end_em > 0 else s[em_idx:]
            # Count 'catboost' occurrences as dict KEY (not in string/comment)
            catboost_keys = re.findall(r"^\s{4}'catboost'\s*:", em_section, re.MULTILINE)
            print(f"  catboost as top-level key count: {len(catboost_keys)}")

        # ── Fix 3: Add proper catboost key if missing ──
        # Find E3_MODEL_SPACES section
        em_idx = s.find("E3_MODEL_SPACES =")
        if em_idx >= 0:
            end_em = s.find("\n}\n", em_idx)
            if end_em > 0:
                em_section = s[em_idx:end_em+3]
                # Check if catboost is a top-level key in E3_MODEL_SPACES
                has_catboost_key = bool(re.search(r"^\s{4}'catboost'\s*:", em_section, re.MULTILINE))
                print(f"  Has catboost as proper key: {has_catboost_key}")

                if not has_catboost_key:
                    # Insert catboost before xgboost or before the closing }
                    catboost_entry = """    'catboost': {
        'learning_rate':       loguniform(0.001, 0.5),
        'depth':               randint(4, 12),
        'l2_leaf_reg':         loguniform(1e-2, 100),
        'random_strength':     uniform(0.0, 2.5),
        'bagging_temperature': uniform(0.0, 2.0),
    },\n"""
                    # Insert before xgboost if it exists, else before randomforest, else before }
                    for anchor in ["    'xgboost'", "    'randomforest'", "\n}"]:
                        anchor_pos = s.find(anchor, em_idx)
                        if anchor_pos > em_idx:
                            s = s[:anchor_pos] + catboost_entry + s[anchor_pos:]
                            print(f"  Inserted catboost entry before '{anchor.strip()}'")
                            changed = True
                            break

        # ── Fix 4: Update logreg in ALL notebooks ──
        # Improve LogReg hyperparams if still original
        if "'logreg'" in s:
            # EXP3_C should have wider C and more solvers - check and update
            s2 = re.sub(
                r"'C':\s+loguniform\(1e-4, 1e2\)",
                "'C':       loguniform(1e-5, 1e3)",
                s
            )
            if s2 != s:
                s = s2
                print("  Updated LogReg C range to 1e-5..1e3")
                changed = True

        # ── Fix 5: Update CatBoost depth if still randint(4, 10) ──
        s2 = re.sub(r"('catboost'[^}]*?'depth':\s+randint\(4,\s*10\))", 
                    lambda m: m.group(0).replace('randint(4, 10)', 'randint(4, 12)'), s, flags=re.DOTALL)
        if s2 != s:
            s = s2
            print("  Updated CatBoost depth to randint(4, 12)")
            changed = True

        # ── Fix 6: Update XGBoost in non-EXP3_B notebooks (EXP3_A, EXP3_C) ──
        if "'xgboost'" in s:
            old_xgb_lr = r"'learning_rate':\s+loguniform\(0\.01, 0\.30\)"
            if re.search(r"'xgboost'[^}]*?" + old_xgb_lr, s, re.DOTALL):
                s2 = re.sub(r"'learning_rate':\s+loguniform\(0\.01, 0\.30\)",
                            "'learning_rate':    loguniform(0.005, 0.40)", s)
                s2 = re.sub(r"'max_depth':\s+randint\(3, 9\)",
                            "'max_depth':        randint(3, 12)", s2)
                s2 = re.sub(r"'subsample':\s+uniform\(0\.6, 0\.4\)",
                            "'subsample':        uniform(0.5, 0.5)", s2)
                s2 = re.sub(r"'colsample_bytree': uniform\(0\.6, 0\.4\)",
                            "'colsample_bytree': uniform(0.5, 0.5)", s2)
                s2 = re.sub(r"'min_child_weight': randint\(1, 7\)",
                            "'min_child_weight': randint(1, 10)", s2)
                s2 = re.sub(r"'gamma':\s+uniform\(0\.0, 0\.5\)",
                            "'gamma':            uniform(0.0, 1.0)", s2)
                s2 = re.sub(r"'reg_lambda':\s+loguniform\(1e-2, 10\)",
                            "'reg_lambda':       loguniform(1e-2, 100)", s2)
                if s2 != s:
                    # Add reg_alpha after reg_lambda if not present
                    if 'reg_alpha' not in s2:
                        s2 = re.sub(
                            r"('reg_lambda':\s+loguniform\(1e-2, 100\),)(\s*\n\s*\},)",
                            r"\1\n        'reg_alpha':        loguniform(1e-3, 10),\2",
                            s2
                        )
                    s = s2
                    print("  Updated XGBoost hyperparams")
                    changed = True

        # ── Fix 7: Update RandomForest if still original ──
        if "'randomforest'" in s:
            s2 = re.sub(r"'max_features':\s+uniform\(0\.2, 0\.8\)",
                        "'max_features':      uniform(0.1, 0.9)", s)
            s2 = re.sub(r"'min_samples_split': randint\(2, 20\)",
                        "'min_samples_split': randint(2, 30)", s2)
            s2 = re.sub(r"'min_samples_leaf':\s+randint\(1, 10\)",
                        "'min_samples_leaf':  randint(1, 15)", s2)
            s2 = re.sub(r"'max_depth':\s+\[None, 10, 20, 30\]",
                        "'max_depth':         [None, 10, 20, 30, 50]", s2)
            if s2 != s:
                s = s2
                print("  Updated RandomForest hyperparams")
                changed = True

        if changed:
            c['source'] = s.splitlines(keepends=True)

    # ── Fix LightGBM _LGB_SPACES in EXP3_C ──────────────────────────────────────
    if 'EXP3_C' in label:
        for i, c in enumerate(nb['cells']):
            s = ''.join(c.get('source', []))
            if '_LGB_SPACES =' in s and 'num_leaves' in s:
                print(f"\n{label} LightGBM Cell #{i} (ID: {c.get('id','?')}): updating _LGB_SPACES")
                old_lgb = re.compile(
                    r"_LGB_SPACES\s*=\s*\{[^}]+\}", re.DOTALL
                )
                new_lgb = """_LGB_SPACES = {
    'learning_rate':     loguniform(0.005, 0.40),
    'max_depth':         randint(3, 12),
    'num_leaves':        randint(20, 200),
    'min_child_samples': randint(5, 50),
    'subsample':         sp_uniform(0.5, 0.5),
    'colsample_bytree':  sp_uniform(0.5, 0.5),
    'reg_lambda':        loguniform(1e-3, 100),
    'reg_alpha':         loguniform(1e-3, 10),
}"""
                s2 = old_lgb.sub(new_lgb, s)
                if s2 != s:
                    c['source'] = s2.splitlines(keepends=True)
                    print(f"  Updated _LGB_SPACES with wider ranges + min_child_samples + reg_alpha")
                    changed = True
                else:
                    print(f"  WARNING: Could not find _LGB_SPACES pattern to update")

    if changed:
        open(nb_path, 'w').write(json.dumps(nb, ensure_ascii=False, indent=1))
        print(f"\n{label}: SAVED successfully")
    else:
        print(f"\n{label}: No changes needed")

    return changed

# Apply to all notebooks
fix_notebook('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb', 'EXP3_C')
fix_notebook('/workspace/Thesis-part-2/EXP3_A_KNN_RF copy.ipynb', 'EXP3_A')
fix_notebook('/workspace/Thesis-part-2/EXP3_B_XGB_AdaBoost copy.ipynb', 'EXP3_B')
