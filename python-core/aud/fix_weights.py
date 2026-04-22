"""
Restore original model weights from backup, then remap key names.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import shutil
from safetensors.torch import load_file, save_file

_EMO_DIR = os.path.dirname(os.path.abspath(__file__))

KEY_MAP = {
    "classifier.dense.weight":  "projector.weight",
    "classifier.dense.bias":    "projector.bias",
    "classifier.output.weight": "classifier.weight",
    "classifier.output.bias":   "classifier.bias",
}

def fix_model(model_dir, label):
    sf_path = os.path.join(model_dir, "model.safetensors")
    backup = sf_path + ".bak"

    # Restore from backup first (the originals before download_models.py overwrote them)
    if os.path.exists(backup):
        print(f"  [{label}] Restoring original from backup...")
        shutil.copy2(backup, sf_path)
    
    tensors = load_file(sf_path)
    print(f"  [{label}] Loaded {len(tensors)} tensors")

    # Show classifier-related keys
    cls_keys = sorted([k for k in tensors if 'classifier' in k or 'projector' in k])
    print(f"  [{label}] Classifier/projector keys: {cls_keys}")

    # Check what we have
    has_old = any(k in tensors for k in KEY_MAP)
    has_new = all(k in tensors for k in KEY_MAP.values())

    if has_new and not has_old:
        print(f"  [{label}] Already has new-style keys only")
        # Print shapes to verify they're not random
        for k in KEY_MAP.values():
            if k in tensors:
                t = tensors[k]
                print(f"    {k}: shape={list(t.shape)}, mean={t.float().mean():.6f}, std={t.float().std():.6f}")
        return

    if has_old:
        print(f"  [{label}] Found old-style keys, remapping...")
        new_tensors = {}
        for key, val in tensors.items():
            if key in KEY_MAP:
                new_key = KEY_MAP[key]
                new_tensors[new_key] = val
                print(f"    {key} -> {new_key}  shape={list(val.shape)} mean={val.float().mean():.4f}")
            else:
                new_tensors[key] = val
        save_file(new_tensors, sf_path)
        print(f"  [{label}] Saved fixed weights")
    else:
        print(f"  [{label}] WARNING: No recognizable classifier keys found!")
        # Try to see all keys
        for k in cls_keys:
            t = tensors[k]
            print(f"    {k}: shape={list(t.shape)}")


print("Fixing categorical model...")
fix_model(os.path.join(_EMO_DIR, "local_emotion_models", "categorical_model"), "CAT")

print("\nFixing dimensional model...")
fix_model(os.path.join(_EMO_DIR, "local_emotion_models", "dimensional_model"), "DIM")

print("\nDone!")
