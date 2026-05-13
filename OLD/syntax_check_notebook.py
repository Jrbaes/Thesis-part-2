import json
import ast
import sys

notebook_path = '/workspace/Thesis-part-2/GPU/GPU_Only_RF_XGBoost_CatBoost_LightGBM.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error loading notebook: {e}")
    sys.exit(1)

cells = nb.get('cells', [])
failed_cells = []

for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        # Handle IPython magic commands which are not valid Python syntax
        # We can comment them out or filter them if they cause parsing errors
        # for a basic check, let's try parsing and if it fails, we report.
        # Note: ast.parse doesn't like magic commands like %matplotlib inline or !pip install
        # So we'll do a simple filtering of lines starting with % or !
        cleaned_source = []
        for line in source.splitlines():
            if line.strip().startswith('%') or line.strip().startswith('!'):
                cleaned_source.append("# " + line)
            else:
                cleaned_source.append(line)
        cleaned_source = "\n".join(cleaned_source)

        try:
            ast.parse(cleaned_source)
        except SyntaxError as e:
            failed_cells.append({
                'cell_index': i,
                'error': str(e),
                'lineno': e.lineno,
                'offset': e.offset,
                'text': e.text
            })

if failed_cells:
    for fail in failed_cells:
        print(f"Cell {fail['cell_index']} failed syntax check:")
        print(f"  Error: {fail['error']}")
else:
    print("All python code cells passed syntax check.")
