"""
Download models from HuggingFace Hub and properly remap old key names
to the new transformers 5.x format before saving.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import torch

_EMO_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings('ignore')

from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, AutoConfig

KEY_MAP = {
    "classifier.dense.weight":  "projector.weight",
    "classifier.dense.bias":    "projector.bias",
    "classifier.output.weight": "classifier.weight",
    "classifier.output.bias":   "classifier.bias",
}

def download_and_fix(hub_id, local_dir, label):
    print(f"\n{'='*60}")
    print(f"  [{label}] Downloading from: {hub_id}")
    print(f"{'='*60}")

    # Download raw safetensors file
    os.makedirs(local_dir, exist_ok=True)

    # Try safetensors first, fall back to pytorch_model.bin
    try:
        sf_path = hf_hub_download(repo_id=hub_id, filename="model.safetensors")
        from safetensors.torch import load_file
        tensors = load_file(sf_path)
        print(f"  [{label}] Loaded safetensors: {len(tensors)} keys")
    except Exception:
        print(f"  [{label}] No safetensors found, trying pytorch_model.bin...")
        bin_path = hf_hub_download(repo_id=hub_id, filename="pytorch_model.bin")
        tensors = torch.load(bin_path, map_location="cpu", weights_only=True)
        print(f"  [{label}] Loaded pytorch_model.bin: {len(tensors)} keys")

    # Show classifier keys before remap
    cls_keys = sorted([k for k in tensors if 'classifier' in k or 'projector' in k])
    print(f"  [{label}] Classifier keys from Hub: {cls_keys}")
    for k in cls_keys:
        t = tensors[k]
        print(f"    {k}: shape={list(t.shape)}, mean={t.float().mean():.6f}, std={t.float().std():.6f}")

    # Remap keys
    new_tensors = {}
    remapped_count = 0
    for key, val in tensors.items():
        if key in KEY_MAP:
            new_key = KEY_MAP[key]
            new_tensors[new_key] = val
            print(f"  [{label}] REMAP: {key} -> {new_key}")
            remapped_count += 1
        else:
            new_tensors[key] = val

    if remapped_count == 0:
        print(f"  [{label}] No remapping needed (keys already in new format)")

    # Save fixed safetensors
    out_path = os.path.join(local_dir, "model.safetensors")
    save_file(new_tensors, out_path)
    print(f"  [{label}] Saved to {out_path}")

    # Also download config and preprocessor
    feat_ext = AutoFeatureExtractor.from_pretrained(hub_id)
    feat_ext.save_pretrained(local_dir)

    config = AutoConfig.from_pretrained(hub_id)
    config.save_pretrained(local_dir)
    print(f"  [{label}] Config and preprocessor saved")

    # Verify the saved weights
    from safetensors.torch import load_file as lf
    verify = lf(out_path)
    v_cls = sorted([k for k in verify if 'classifier' in k or 'projector' in k])
    print(f"  [{label}] Verification - classifier keys: {v_cls}")
    for k in v_cls:
        t = verify[k]
        print(f"    {k}: shape={list(t.shape)}, mean={t.float().mean():.6f}, std={t.float().std():.6f}")


# Download and fix both models
download_and_fix(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    os.path.join(_EMO_DIR, "local_emotion_models", "categorical_model"),
    "CAT"
)

download_and_fix(
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    os.path.join(_EMO_DIR, "local_emotion_models", "dimensional_model"),
    "DIM"
)

print("\n\nAll models downloaded and fixed!")
print("Run: python test_model.py  to verify")
