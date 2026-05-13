import json, ast, sys

file_path = "/workspace/Thesis-part-2/GPU/GPU_Only_RF_XGBoost_CatBoost_LightGBM.ipynb"
try:
    with open(file_path, "r") as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

failures = []
for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        lines = []
        for line in source.splitlines():
            if line.strip().startswith(("%", "!")):
                lines.append("# " + line)
            else:
                lines.append(line)
        modified_source = "\n".join(lines)
        try:
            ast.parse(modified_source)
        except SyntaxError as e:
            failures.append((i, e))

if failures:
    for idx, err in failures:
        print(f"Cell {idx} SyntaxError: {err}")
else:
    print("all code cells are syntactically valid")
