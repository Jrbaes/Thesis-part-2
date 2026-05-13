import json
import ast
import sys

def check_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    errors = []
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                source = "".join(source)
            
            # Process lines: comment out magics
            lines = source.splitlines()
            processed_lines = []
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith('%') or stripped.startswith('!'):
                    processed_lines.append(f"# {line}")
                else:
                    processed_lines.append(line)
            
            processed_source = "\n".join(processed_lines)
            
            try:
                ast.parse(processed_source)
            except SyntaxError as e:
                errors.append(f"Cell {i} SyntaxError: {e}")

    if errors:
        for err in errors:
            print(err)
    else:
        print("All good")

if __name__ == "__main__":
    check_notebook(sys.argv[1])
