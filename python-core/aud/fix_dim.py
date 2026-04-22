"""Fix the dimensional model's remaining old key names."""
import sys, os, shutil
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from safetensors.torch import load_file, save_file

_EMO_DIR = os.path.dirname(os.path.abspath(__file__))
DIM_DIR = os.path.join(_EMO_DIR, "local_emotion_models", "dimensional_model")
DIM_PATH = os.path.join(DIM_DIR, "model.safetensors")
TMP_PATH = os.path.join(DIM_DIR, "model_fixed.safetensors")

tensors = load_file(DIM_PATH)

DIM_MAP = {
    "classifier.out_proj.weight": "classifier.weight",
    "classifier.out_proj.bias":   "classifier.bias",
}

new_tensors = {}
for key, val in tensors.items():
    if key in DIM_MAP:
        new_key = DIM_MAP[key]
        new_tensors[new_key] = val
        print(f"  REMAP: {key} -> {new_key}  shape={list(val.shape)}")
    else:
        new_tensors[key] = val

# Save to temp, then replace
save_file(new_tensors, TMP_PATH)
os.remove(DIM_PATH)
shutil.move(TMP_PATH, DIM_PATH)

# Verify
verify = load_file(DIM_PATH)
cls_keys = sorted([k for k in verify if 'classifier' in k or 'projector' in k])
print(f"\nVerified keys: {cls_keys}")
for k in cls_keys:
    t = verify[k]
    print(f"  {k}: shape={list(t.shape)}, mean={t.float().mean():.6f}, std={t.float().std():.6f}")

print("\nDimensional model fixed!")
