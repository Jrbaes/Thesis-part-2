import json
import ast
import sys

def parse_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    cells = nb.get('cells', [])
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                source = "".join(source)
            
            lines = source.splitlines()
            processed_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('%') or stripped.startswith('!'):
                    processed_lines.append(f"# {line}")
                else:
                    processed_lines.append(line)
            
            processed_source = "\n".join(processed_lines)
            
            try:
                ast.parse(processed_source)
            except SyntaxError as e:
                print(f"Cell {i}: Syntax Error: {e}")
                return

    print("Success: All code cells parsed correctly.")

if __name__ == "__main__":
    parse_notebook("/workspace/Thesis-part-2/GPU/GPU_Only_RF_XGBoost_CatBoost_LightGBM.ipynb")
