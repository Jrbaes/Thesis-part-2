import json, re

def fix_adaboost(nb_path, label):
    nb = json.loads(open(nb_path).read())
    changed = False

    for i, c in enumerate(nb['cells']):
        s = ''.join(c.get('source', []))
        if 'E3_MODEL_SPACES =' not in s or "'adaboost'" not in s:
            continue

        ada_start = s.find("    'adaboost': {")
        if ada_start < 0:
            continue

        # Find matching closing brace
        brace_count = 0
        end_pos = -1
        for j, ch in enumerate(s[ada_start:], ada_start):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = j + 1
                    if j + 1 < len(s) and s[j + 1] == ',':
                        end_pos = j + 2
                    break

        if end_pos < 0:
            continue

        old_ada_block = s[ada_start:end_pos]
        print(f"\n{label} Cell #{i}: adaboost block:")
        print(repr(old_ada_block[:200]))

        # Check for corruption (catboost params inside)
        is_corrupted = "'depth'" in old_ada_block or "'l2_leaf_reg'" in old_ada_block
        # Check for wrong indentation (4-space instead of 8-space for inner keys)
        has_wrong_indent = bool(re.search(r"\n    '[a-z]", old_ada_block))

        if is_corrupted:
            print(f"  -> Corruption detected! Fixing...")
            clean_ada = "    'adaboost': {\n        'n_estimators':  randint(20, 80),\n        'learning_rate': loguniform(0.01, 1.0),\n        'base_depth':    randint(1, 5),\n    },"
            s = s[:ada_start] + clean_ada + s[end_pos:]
            c['source'] = s.splitlines(keepends=True)
            changed = True
        elif has_wrong_indent:
            print(f"  -> Wrong indentation! Fixing...")
            clean_ada = "    'adaboost': {\n        'n_estimators':  randint(20, 80),\n        'learning_rate': loguniform(0.01, 1.0),\n        'base_depth':    randint(1, 5),\n    },"
            s = s[:ada_start] + clean_ada + s[end_pos:]
            c['source'] = s.splitlines(keepends=True)
            changed = True
        else:
            print(f"  -> OK, no fix needed")
        break

    if changed:
        open(nb_path, 'w').write(json.dumps(nb, ensure_ascii=False, indent=1))
        print(f"  SAVED: {nb_path}")
    return changed

fix_adaboost('/workspace/Thesis-part-2/EXP3_B_XGB_AdaBoost copy.ipynb', 'EXP3_B')
fix_adaboost('/workspace/Thesis-part-2/EXP3_A_KNN_RF copy.ipynb', 'EXP3_A')
fix_adaboost('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb', 'EXP3_C')
fix_adaboost('/workspace/Thesis-part-2/EXP3_D_LightGBM copy.ipynb', 'EXP3_D')
