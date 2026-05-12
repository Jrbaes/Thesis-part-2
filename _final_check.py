import json, re

def check_hyperparams(nb_path, label, key_models):
    nb = json.loads(open(nb_path).read())
    print(f"\n{'='*55}")
    print(f"FINAL STATE: {label}")
    print('='*55)
    
    for i, c in enumerate(nb['cells']):
        s = ''.join(c.get('source', []))
        if 'E3_MODEL_SPACES =' not in s:
            continue
        idx = s.find('E3_MODEL_SPACES =')
        end_idx = s.find('\n}\n', idx) + 4
        em = s[idx:end_idx if end_idx > 4 else idx+3000]
        
        for model in key_models:
            midx = em.find(f"'{model}'")
            if midx >= 0:
                # Get key params
                mblock_end = em.find("    },", midx) + 6
                mblock = em[midx:mblock_end]
                print(f"\n  [{model}]: {mblock[:300].strip()}")
            else:
                print(f"\n  [{model}]: NOT FOUND in E3_MODEL_SPACES")
        break
    
    # Check LightGBM for EXP3_C
    if '_LGB_SPACES' in ['_LGB_SPACES']:
        for c in nb['cells']:
            s = ''.join(c.get('source', []))
            if '_LGB_SPACES =' in s:
                idx = s.find('_LGB_SPACES =')
                end_idx = s.find('\n}\n', idx) + 4
                print(f"\n  [LightGBM _LGB_SPACES]:")
                print("  " + s[idx:end_idx if end_idx > 4 else idx+400].replace('\n', '\n  '))
                break

check_hyperparams(
    '/workspace/Thesis-part-2/EXP3_A_KNN_RF copy.ipynb', 
    'EXP3_A (KNN+RF)', 
    ['knn', 'randomforest']
)
check_hyperparams(
    '/workspace/Thesis-part-2/EXP3_B_XGB_AdaBoost copy.ipynb', 
    'EXP3_B (XGB+AdaBoost)', 
    ['xgboost', 'adaboost']
)
check_hyperparams(
    '/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb', 
    'EXP3_C (LogReg+CatBoost+LGB)', 
    ['logreg', 'catboost']
)
# Check LightGBM specifically
nb_c = json.loads(open('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb').read())
for c in nb_c['cells']:
    s = ''.join(c.get('source', []))
    if '_LGB_SPACES =' in s:
        idx = s.find('_LGB_SPACES =')
        end_idx = s.find('\n}\n', idx) + 4
        print(f"\n  EXP3_C [LightGBM _LGB_SPACES]:")
        print(s[idx:end_idx if end_idx > 4 else idx+400])
        break
