import json, re

nb = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES =' in s:
        idx = s.find('E3_MODEL_SPACES =')
        end_idx = s.find('\n}\n', idx) + 4
        em = s[idx:end_idx if end_idx > 4 else idx+3000]
        print(f"EXP3_C Cell #{i} E3_MODEL_SPACES (first 1500 chars):")
        print(em[:1500])
        print("... (continues) ...")
        # Check for corruption
        ada_idx = em.find("'adaboost'")
        if ada_idx >= 0:
            ada_end = em.find("\n    },", ada_idx) + 7
            print("\nadaboost block:")
            print(em[ada_idx:ada_end])
        cat_idx = em.find("'catboost'")
        if cat_idx >= 0:
            cat_end = em.find("\n    },", cat_idx) + 7
            print("\ncatboost block:")
            print(em[cat_idx:cat_end])
        else:
            print("\nWARNING: No catboost key found!")
        break
