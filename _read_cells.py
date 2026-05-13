import json
path = r"c:\Jon\College\Thesis\V2.2.1.1\EXP3_B_XGB_AdaBoost copy.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]
for idx in [4, 7, 8, 10, 11]:
    print(f"\n=== CELL {idx} ===")
    print("".join(cells[idx]["source"]))
