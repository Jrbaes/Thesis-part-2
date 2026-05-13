import venn_abers
import numpy as np
import inspect

print("Version:", getattr(venn_abers, "__version__", "unknown"))
print("Symbols:", [s for s in dir(venn_abers) if not s.startswith("_")])

for name in dir(venn_abers):
    if name.startswith("_"): continue
    obj = getattr(venn_abers, name)
    print(f"\n--- {name} ---")
    if callable(obj):
        try:
            print("Signature:", inspect.signature(obj))
        except:
            print("Signature: unknown")
    print("Doc:", obj.__doc__[:200] if obj.__doc__ else "No doc")

# Synthetic data
np.random.seed(42)
p = np.random.rand(10)
y = (p > 0.5).astype(int)

# 1. VennAbersCalibrator.fit(p.reshape(-1,1), y)
if hasattr(venn_abers, 'VennAbersCalibrator'):
    print("\nTesting VennAbersCalibrator fit patterns:")
    try:
        va = venn_abers.VennAbersCalibrator()
        va.fit(p.reshape(-1,1), y)
        print("Fit (p.reshape(-1,1), y) success")
        res = va.predict_proba(p.reshape(-1,1))
        print("Predict proba shape:", res.shape)
    except Exception as e:
        print("Fit (p.reshape(-1,1), y) failed:", e)

    try:
        va = venn_abers.VennAbersCalibrator()
        va.fit(p, y)
        print("Fit (p, y) success")
        res = va.predict_proba(p)
        print("Predict proba shape:", res.shape)
    except Exception as e:
        print("Fit (p, y) failed:", e)

# 2. venn_abers(p, y) or similar?
if hasattr(venn_abers, 'venn_abers'):
    print("\nTesting venn_abers function:")
    try:
        # Some implementations take (ds_train, y_train, ds_test)
        # Try a simple guess or look at signature
        pass
    except Exception as e:
        print("venn_abers function test failed:", e)

