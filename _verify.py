import json, re

def verify_nb(path, label):
    nb = json.loads(open(path).read())
    print(f"\n{'='*60}")
    print(f"VERIFICATION: {label}")
    print('='*60)
    for i, c in enumerate(nb['cells']):
        s = ''.join(c.get('source', []))
        if 'E3_MODEL_SPACES =' in s:
            idx = s.find('E3_MODEL_SPACES =')
            end_idx = s.find('\n}\n', idx) + 4
            em = s[idx:end_idx if end_idx > 4 else idx+3000]
            print(f"\nCell #{i} E3_MODEL_SPACES (first 3000 chars):")
            print(em[:3000])
            break
    
    # Check _LGB_SPACES for EXP3_C
    if 'EXP3_C' in label:
        for i, c in enumerate(nb['cells']):
            s = ''.join(c.get('source', []))
            if '_LGB_SPACES =' in s:
                idx = s.find('_LGB_SPACES =')
                end_idx = s.find('\n}\n', idx) + 4
                print(f"\nCell #{i} _LGB_SPACES:")
                print(s[idx:end_idx if end_idx > 4 else idx+400])
                break

verify_nb('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb', 'EXP3_C')
verify_nb('/workspace/Thesis-part-2/EXP3_A_KNN_RF copy.ipynb', 'EXP3_A')
verify_nb('/workspace/Thesis-part-2/EXP3_B_XGB_AdaBoost copy.ipynb', 'EXP3_B')
