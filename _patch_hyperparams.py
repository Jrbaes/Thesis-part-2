#!/usr/bin/env python3
"""Patch hyperparameters in copy notebooks using JSON manipulation."""
import json, sys, re
from pathlib import Path

def patch_cell_source(source_lines, old_fragments, new_fragments):
    """Replace old text fragments in cell source with new ones."""
    source = ''.join(source_lines)
    for old, new in zip(old_fragments, new_fragments):
        if old not in source:
            print(f"  WARNING: Could not find: {old!r}", file=sys.stderr)
            return source_lines
        source = source.replace(old, new, 1)
    return source.splitlines(keepends=True)

def find_cell_by_id(nb, cell_id):
    for cell in nb['cells']:
        if cell.get('id') == cell_id or cell.get('metadata', {}).get('id') == cell_id:
            return cell
    # Try VSC- prefix stripped
    for cell in nb['cells']:
        cid = cell.get('id', '')
        if cid == cell_id or f'#VSC-{cid}' == cell_id or cid == cell_id.lstrip('#VSC-'):
            return cell
    return None

# ── EXP3_A: KNN + RandomForest ──────────────────────────────────────────────
nb_path = Path('/workspace/Thesis-part-2/EXP3_A_KNN_RF copy.ipynb')
nb = json.loads(nb_path.read_text())

cell = find_cell_by_id(nb, 'VSC-1c97e701')
if cell is None:
    # Try finding by E3_MODEL_SPACES content
    for c in nb['cells']:
        src = ''.join(c.get('source', []))
        if 'E3_MODEL_SPACES' in src and 'n_neighbors' in src and 'randomforest' in src:
            cell = c
            print("Found EXP3_A Cell 11 by content scan")
            break

if cell is None:
    print("ERROR: Could not find Cell 11 in EXP3_A", file=sys.stderr)
    sys.exit(1)

src = ''.join(cell['source'])
print("EXP3_A Cell 11 found, length:", len(src))

# Patch KNN
old_knn = """    'knn': {
        'n_neighbors': randint(3, 31),
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan'],
    },"""
new_knn = """    'knn': {
        'n_neighbors': randint(3, 51),
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan', 'chebyshev'],
    },"""

# Patch RandomForest
old_rf = """    'randomforest': {
        'max_features':      uniform(0.2, 0.8),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf':  randint(1, 10),
        'max_depth':         [None, 10, 20, 30],
    },"""
new_rf = """    'randomforest': {
        'max_features':      uniform(0.1, 0.9),
        'min_samples_split': randint(2, 30),
        'min_samples_leaf':  randint(1, 15),
        'max_depth':         [None, 10, 20, 30, 50],
    },"""

changed = False
if old_knn in src:
    src = src.replace(old_knn, new_knn, 1)
    print("  Patched KNN n_neighbors + chebyshev metric")
    changed = True
else:
    print("  WARNING: KNN old pattern not found, trying flexible match")
    # Try flexible matching
    src2 = re.sub(r"'n_neighbors': randint\(3, 31\)", "'n_neighbors': randint(3, 51)", src)
    src2 = re.sub(r"'metric':\s+\['euclidean', 'manhattan'\]", "'metric':      ['euclidean', 'manhattan', 'chebyshev']", src2)
    if src2 != src:
        src = src2
        print("  Patched KNN via flexible regex")
        changed = True

if old_rf in src:
    src = src.replace(old_rf, new_rf, 1)
    print("  Patched RandomForest hyperparams")
    changed = True
else:
    print("  WARNING: RF old pattern not found, trying flexible match")
    src2 = re.sub(r"'max_features':\s+uniform\(0\.2, 0\.8\)", "'max_features':      uniform(0.1, 0.9)", src)
    src2 = re.sub(r"'min_samples_split': randint\(2, 20\)", "'min_samples_split': randint(2, 30)", src2)
    src2 = re.sub(r"'min_samples_leaf':\s+randint\(1, 10\)", "'min_samples_leaf':  randint(1, 15)", src2)
    src2 = re.sub(r"'max_depth':\s+\[None, 10, 20, 30\]", "'max_depth':         [None, 10, 20, 30, 50]", src2)
    if src2 != src:
        src = src2
        print("  Patched RF via flexible regex")
        changed = True

if changed:
    cell['source'] = src.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print("EXP3_A saved.")
else:
    print("ERROR: No changes made to EXP3_A")


# ── EXP3_B: XGBoost + AdaBoost ──────────────────────────────────────────────
nb_path = Path('/workspace/Thesis-part-2/EXP3_B_XGB_AdaBoost copy.ipynb')
nb = json.loads(nb_path.read_text())

cell = None
for c in nb['cells']:
    src_check = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES' in src_check and 'xgboost' in src_check and 'adaboost' in src_check:
        cell = c
        print("Found EXP3_B Cell 11 by content scan")
        break

if cell is None:
    print("ERROR: Could not find Cell 11 in EXP3_B", file=sys.stderr)
    sys.exit(1)

src = ''.join(cell['source'])

old_xgb = """    'xgboost': {
        'learning_rate':    loguniform(0.01, 0.30),
        'max_depth':        randint(3, 9),
        'subsample':        uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 7),
        'gamma':            uniform(0.0, 0.5),
        'reg_lambda':       loguniform(1e-2, 10),
    },"""
new_xgb = """    'xgboost': {
        'learning_rate':    loguniform(0.005, 0.40),
        'max_depth':        randint(3, 12),
        'subsample':        uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'min_child_weight': randint(1, 10),
        'gamma':            uniform(0.0, 1.0),
        'reg_lambda':       loguniform(1e-2, 100),
        'reg_alpha':        loguniform(1e-3, 10),
    },"""

old_ada = """    'adaboost': {
        'n_estimators':  randint(20, 60),   # capped — CPU-only
        'learning_rate': uniform(0.05, 0.45),
        'base_depth':    randint(1, 3),
    },"""
new_ada = """    'adaboost': {
        'n_estimators':  randint(20, 80),
        'learning_rate': loguniform(0.01, 1.0),
        'base_depth':    randint(1, 5),
    },"""

changed = False
if old_xgb in src:
    src = src.replace(old_xgb, new_xgb, 1)
    print("  Patched XGBoost hyperparams")
    changed = True
else:
    print("  WARNING: XGBoost old pattern not found, trying flexible match")
    src2 = re.sub(r"'learning_rate':\s+loguniform\(0\.01, 0\.30\)", "'learning_rate':    loguniform(0.005, 0.40)", src)
    src2 = re.sub(r"'max_depth':\s+randint\(3, 9\)", "'max_depth':        randint(3, 12)", src2)
    src2 = re.sub(r"'subsample':\s+uniform\(0\.6, 0\.4\)", "'subsample':        uniform(0.5, 0.5)", src2)
    src2 = re.sub(r"'colsample_bytree': uniform\(0\.6, 0\.4\)", "'colsample_bytree': uniform(0.5, 0.5)", src2)
    src2 = re.sub(r"'min_child_weight': randint\(1, 7\)", "'min_child_weight': randint(1, 10)", src2)
    src2 = re.sub(r"'gamma':\s+uniform\(0\.0, 0\.5\)", "'gamma':            uniform(0.0, 1.0)", src2)
    src2 = re.sub(r"'reg_lambda':\s+loguniform\(1e-2, 10\)", "'reg_lambda':       loguniform(1e-2, 100)", src2)
    if src2 != src:
        src = src2
        # Add reg_alpha after reg_lambda in xgboost block
        src = re.sub(
            r"('reg_lambda':\s+loguniform\(1e-2, 100\),)(\s*\},\s*\n)",
            r"\1\n        'reg_alpha':        loguniform(1e-3, 10),\2",
            src
        )
        print("  Patched XGBoost via flexible regex")
        changed = True

if old_ada in src:
    src = src.replace(old_ada, new_ada, 1)
    print("  Patched AdaBoost hyperparams")
    changed = True
else:
    print("  WARNING: AdaBoost old pattern not found, trying flexible match")
    src2 = re.sub(r"'n_estimators':\s+randint\(20, 60\).*?# capped.*?\n", 
                  "'n_estimators':  randint(20, 80),\n", src)
    src2 = re.sub(r"'learning_rate': uniform\(0\.05, 0\.45\)", 
                  "'learning_rate': loguniform(0.01, 1.0)", src2)
    src2 = re.sub(r"'base_depth':\s+randint\(1, 3\)", 
                  "'base_depth':    randint(1, 5)", src2)
    if src2 != src:
        src = src2
        print("  Patched AdaBoost via flexible regex")
        changed = True

if changed:
    cell['source'] = src.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print("EXP3_B saved.")
else:
    print("ERROR: No changes made to EXP3_B")


# ── EXP3_C: LogReg + CatBoost additions ─────────────────────────────────────
nb_path = Path('/workspace/Thesis-part-2/EXP3_C_LogReg_CatBoost copy.ipynb')
nb = json.loads(nb_path.read_text())

cell = None
for c in nb['cells']:
    src_check = ''.join(c.get('source', []))
    if 'E3_MODEL_SPACES' in src_check and 'logreg' in src_check and 'catboost' in src_check:
        cell = c
        print("Found EXP3_C Cell 11 by content scan")
        break

if cell is None:
    print("ERROR: Could not find Cell 11 in EXP3_C", file=sys.stderr)
    sys.exit(1)

src = ''.join(cell['source'])

# Add 'saga' solver to LogReg
changed = False
src2 = re.sub(
    r"'solver': \['lbfgs', 'liblinear'\]",
    "'solver': ['lbfgs', 'liblinear', 'saga']",
    src
)
if src2 != src:
    src = src2
    print("  Added 'saga' solver to LogReg")
    changed = True
else:
    print("  LogReg solver already includes saga or pattern not found")

# Extend CatBoost depth to randint(4, 12)
src2 = re.sub(r"'depth':\s+randint\(4, 10\)", "'depth':               randint(4, 12)", src)
if src2 != src:
    src = src2
    print("  Extended CatBoost depth to randint(4, 12)")
    changed = True
else:
    # Maybe already 12 or pattern different
    if 'randint(4, 12)' in src:
        print("  CatBoost depth already extended to 12")
    else:
        print("  WARNING: CatBoost depth pattern not found")

# Add bagging_temperature to CatBoost if not present
if 'bagging_temperature' not in src:
    # Insert after random_strength line
    src2 = re.sub(
        r"('random_strength':\s+uniform\(0\.0, 2\.5\),)",
        r"\1\n        'bagging_temperature': uniform(0.0, 2.0),",
        src
    )
    if src2 != src:
        src = src2
        print("  Added bagging_temperature to CatBoost")
        changed = True
    else:
        print("  WARNING: Could not add bagging_temperature to CatBoost")
else:
    print("  bagging_temperature already present in CatBoost")

if changed:
    cell['source'] = src.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print("EXP3_C Cell 11 saved.")
else:
    print("No changes needed in EXP3_C Cell 11")

print("\nAll done!")
