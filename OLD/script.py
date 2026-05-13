import inspect
import numpy as np
try:
    import venn_abers
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "venn-abers"])
    import venn_abers

print('module:', venn_abers)
print('version:', getattr(venn_abers, '__version__', 'unknown'))
print('symbols:', [s for s in dir(venn_abers) if 'Venn' in s or 'Abers' in s or 'venn' in s.lower()][:50])

for name in dir(venn_abers):
    if 'Venn' in name or 'Abers' in name:
        obj = getattr(venn_abers, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            print('\nNAME', name)
            try:
                print('SIGNATURE', inspect.signature(obj))
            except Exception as e:
                print('SIGNATURE_ERR', e)
            doc = inspect.getdoc(obj)
            if doc:
                print('DOC', doc.split('\n')[0][:200])

# Try calibrator usage variants
rng = np.random.default_rng(0)
y = rng.integers(0,2, size=100)
p = np.clip(rng.random(100), 1e-6, 1-1e-6)

if hasattr(venn_abers, 'VennAbersCalibrator'):
    C = venn_abers.VennAbersCalibrator
    print('\nTesting VennAbersCalibrator methods:', [m for m in dir(C) if not m.startswith('_')])
    for mode in ['reshape', 'flat']:
        try:
            c = C()
            if mode=='reshape':
                c.fit(p.reshape(-1,1), y)
                out = c.predict_proba(p.reshape(-1,1))
            else:
                c.fit(p, y)
                out = c.predict_proba(p)
            print('SUCCESS', mode, type(out), np.asarray(out).shape)
        except Exception as e:
            print('FAIL', mode, type(e).__name__, e)
