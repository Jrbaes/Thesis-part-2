import json

path = '/workspace/Thesis-part-2/Main_2015_GPU_RF_XGB_CAT_RIGOROUS_OPT_exp2.ipynb'

with open(path) as f:
    nb = json.load(f)

print('JSON valid')

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = cell['source']
        for i, line in enumerate(src):
            if 'pip install' in line and 'cuml' in line:
                print('Found:', repr(line))
                src[i] = '%pip install -q cuml-cu12 "cuda-python<13" "cuda-toolkit<13" --extra-index-url=https://pypi.nvidia.com'
                print('Fixed:', repr(src[i]))

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

print('Saved')
