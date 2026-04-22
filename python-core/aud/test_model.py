"""Test multiple chunks from different parts of the audio files."""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import soundfile as sf
import torch
import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline

_EMO_DIR = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(_EMO_DIR, "local_emotion_models", "categorical_model")
DIM = os.path.join(_EMO_DIR, "local_emotion_models", "dimensional_model")

print("Loading models...")
cat_clf = pipeline("audio-classification", model=CAT, device=0, torch_dtype=torch.float32)
dim_clf = pipeline("audio-classification", model=DIM, device=0, torch_dtype=torch.float32)
print("Models loaded.\n")

for fname in ["test.wav", "test1.wav", "test_fixed.wav", "test_fixed.wav"]:
    try:
        audio, sr = sf.read(fname, dtype='float32')
    except Exception:
        continue
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    total_dur = len(audio) / sr
    print(f"{'='*60}")
    print(f"FILE: {fname} | Duration: {total_dur:.1f}s | Samples: {len(audio)}")
    print(f"{'='*60}")

    # Test chunks at different offsets
    offsets = [0, int(total_dur/4), int(total_dur/2), int(3*total_dur/4)]
    for off_sec in offsets:
        start = off_sec * sr
        end = start + 3 * sr
        if end > len(audio):
            continue
        chunk = audio[start:end]

        rms = float(np.sqrt(np.mean(chunk**2)))
        peak = float(np.max(np.abs(chunk)))

        # Normalize
        chunk_norm = chunk / (peak + 1e-9)

        inp = {"array": np.array(chunk_norm, dtype=np.float32), "sampling_rate": 16000}
        cat_res = cat_clf(inp, top_k=None)
        dim_res = dim_clf(inp)

        top = cat_res[0]
        top2 = cat_res[1] if len(cat_res) > 1 else None
        dims = {r['label'].lower(): r['score'] for r in dim_res}

        spread = cat_res[0]['score'] - cat_res[-1]['score']

        print(f"  [{off_sec:.0f}s-{off_sec+3:.0f}s] RMS={rms:.4f} peak={peak:.4f} | "
              f"{top['label']:<10} {top['score']:.1%} | "
              f"spread={spread:.3f} | "
              f"V={dims.get('valence',0):.3f} A={dims.get('arousal',0):.3f}")
    print()

print("DONE")
