import json
import ast
import os

path = "/workspace/Thesis-part-2/GPU/GPU_Only_RF_XGBoost_CatBoost_LightGBM.ipynb"
if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

errors = []
for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        
        # Comment out magic commands
        lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!")):
                lines.append("# " + line)
            else:
                lines.append(line)
        processed_source = "\n".join(lines)
        
        try:
            ast.parse(processed_source)
        except SyntaxError as e:
            errors.append(f"Cell {i} SyntaxError at line {e.lineno}: {e.msg}\nSource snippet:\n{line}")

if errors:
    for err in errors:
        print(err)
else:
    print("Success: All code cells parsed correctly.")
