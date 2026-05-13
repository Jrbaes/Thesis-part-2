import json
path = r"c:\Jon\College\Thesis\V2.2.1.1\EXP3_B_XGB_AdaBoost copy.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = "".join(c["source"])[:150].replace("\n", " ").encode("ascii", "replace").decode()
    print(f"[{i}] {c['cell_type']}: {src}")
