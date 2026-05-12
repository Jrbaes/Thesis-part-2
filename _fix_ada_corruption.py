import json, re

nb_path = '/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb'
nb = json.loads(open(nb_path).read())

for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES =' not in s:
        continue

    # Find and fix the corrupted adaboost block
    # The pattern: adaboost dict with catboost params injected after 'base_depth'
    
    # Use a simpler approach: find 'adaboost': { ... } block and extract/replace it
    ada_start = s.find("    'adaboost': {")
    if ada_start < 0:
        print("No adaboost key found")
        continue
    
    # Find the matching closing },
    # Count braces from ada_start
    brace_count = 0
    pos = ada_start + len("    'adaboost': {")
    end_pos = -1
    for j, ch in enumerate(s[ada_start:], ada_start):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                # Check if followed by ,
                end_pos = j + 1
                if j + 1 < len(s) and s[j + 1] == ',':
                    end_pos = j + 2
                break
    
    if end_pos < 0:
        print("Could not find end of adaboost block")
        continue
    
    old_ada_block = s[ada_start:end_pos]
    print(f"Found adaboost block ({len(old_ada_block)} chars):")
    print(repr(old_ada_block))
    
    # Check if corrupted (has 'depth' key which is catboost-specific)
    if "'depth'" in old_ada_block or "'l2_leaf_reg'" in old_ada_block:
        print("\nCorruption detected! Replacing with clean adaboost block...")
        clean_ada_block = "    'adaboost': {\n        'n_estimators': randint(50, 300),\n        'learning_rate': uniform(0.01, 0.49),\n        'base_depth':    randint(1, 5),\n    },"
        s = s[:ada_start] + clean_ada_block + s[end_pos:]
        c['source'] = s.splitlines(keepends=True)
        
        nb_path_out = nb_path
        open(nb_path_out, 'w').write(json.dumps(nb, ensure_ascii=False, indent=1))
        print(f"Saved to {nb_path_out}")
    else:
        print("adaboost block is clean, no fix needed")
    
    break
