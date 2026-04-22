"""
audio_emotion_bridge.py — Audio Emotion Bridge for Advisor-AI
═══════════════════════════════════════════════════════════════
Runs the acoustic emotion engine as a background service and
writes results to the shared signal file (ta_face_signals.json)
so the vision pipeline, backend, and dashboard can consume them.

Usage:
    conda activate aud
    python audio_emotion_bridge.py
    python audio_emotion_bridge.py --no-canary
    python audio_emotion_bridge.py --device 2
"""

import argparse
import collections
import json
import os
import queue
import sys
import tempfile
import threading
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

# ── Import from the emotions package (now inside python-core/aud) ────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTHON_CORE = os.path.join(_BASE_DIR, "python-core")
if _PYTHON_CORE not in sys.path:
    sys.path.insert(0, _PYTHON_CORE)

from aud.emo import (
    EmotionSmoother,
    SharedState,
    analyse_text_emotion,
    fuse_emotions,
    preprocess_audio,
    compute_rms,
    sharpen_scores,
    parse_dims,
    load_emotion_models,
    load_canary_model,
    check_gpu,
    inference_worker,
    asr_worker,
    save_temp_wav,
    SAMPLE_RATE,
    WINDOW_SEC,
    STEP_SEC,
    ASR_CHUNK_SEC,
    NOISE_GATE_RMS,
    EMA_ALPHA,
    ACOUSTIC_WEIGHT,
    TEXT_WEIGHT,
)


# ═══════════════════════════════════════════════
#  SHARED SIGNAL FILE (same as vision pipeline)
# ═══════════════════════════════════════════════
SIGNAL_FILE = os.path.join(_BASE_DIR, "ta_face_signals.json")
BRIDGE_SIGNAL_FILE = os.path.join(_BASE_DIR, "ta_audio_signals.json")

# Colour codes
G  = "\033[92m"; Y  = "\033[93m"; C  = "\033[96m"
W  = "\033[97m"; DIM = "\033[2m"; RESET = "\033[0m"; BOLD = "\033[1m"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 4)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_audio_signals(audio_data):
    """Write audio emotion data to its own signal file."""
    audio_data["timestamp"] = int(time.time())
    try:
        tmp = BRIDGE_SIGNAL_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(audio_data, f, indent=2, cls=NumpyEncoder)
        os.replace(tmp, BRIDGE_SIGNAL_FILE)
    except OSError:
        try:
            with open(BRIDGE_SIGNAL_FILE, "w") as f:
                json.dump(audio_data, f, indent=2, cls=NumpyEncoder)
        except Exception:
            pass


def merge_into_vision_signals(audio_data):
    """Read the vision signal file, inject audio fields, write back."""
    try:
        if os.path.exists(SIGNAL_FILE):
            with open(SIGNAL_FILE, "r") as f:
                vision = json.load(f)
        else:
            vision = {}
    except (json.JSONDecodeError, IOError):
        vision = {}

    # Inject audio emotion data under a dedicated key
    vision["audio_emotion"] = {
        "label": audio_data.get("label", "unknown"),
        "confidence": audio_data.get("confidence", 0),
        "all_scores": audio_data.get("all_scores", {}),
        "valence": audio_data.get("valence", 0),
        "arousal": audio_data.get("arousal", 0),
        "dominance": audio_data.get("dominance", 0),
        "vad_quadrant": audio_data.get("vad_quadrant", ""),
        "source": "wav2vec2",
    }

    # Inject fusion data if available
    if audio_data.get("fused_scores"):
        fused = audio_data["fused_scores"]
        fused_label = max(fused, key=fused.get) if fused else "unknown"
        vision["multimodal_emotion"] = {
            "label": fused_label,
            "confidence": fused.get(fused_label, 0),
            "all_scores": fused,
            "source": "acoustic+text_fusion",
            "acoustic_weight": ACOUSTIC_WEIGHT,
            "text_weight": TEXT_WEIGHT,
        }

    # Inject transcript
    if audio_data.get("transcript"):
        vision["live_transcript"] = audio_data["transcript"]

    try:
        tmp = SIGNAL_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(vision, f, indent=2, cls=NumpyEncoder)
        os.replace(tmp, SIGNAL_FILE)
    except OSError:
        try:
            with open(SIGNAL_FILE, "w") as f:
                json.dump(vision, f, indent=2, cls=NumpyEncoder)
        except Exception:
            pass


def signal_publisher(state, stop_event):
    """
    Periodically reads SharedState and publishes audio emotion
    data to both the standalone and merged signal files.
    """
    while not stop_event.is_set():
        with state.lock:
            acoustic = state.latest_acoustic
            fused = state.latest_fused
            text_emo = state.latest_text_emotion
            transcripts = list(state.transcript_lines) if hasattr(state, 'transcript_lines') else []

        if acoustic and acoustic.get("type") == "emotion":
            from aud.emo import vad_quadrant

            audio_data = {
                "label": acoustic.get("label", "unknown"),
                "confidence": round(acoustic.get("conf", 0), 4),
                "all_scores": acoustic.get("all", {}),
                "valence": round(acoustic.get("valence", 0), 4),
                "arousal": round(acoustic.get("arousal", 0), 4),
                "dominance": round(acoustic.get("dominance", 0), 4),
                "vad_quadrant": vad_quadrant(
                    acoustic.get("valence", 0),
                    acoustic.get("arousal", 0),
                ),
                "rms": round(acoustic.get("rms", 0), 6),
                "fused_scores": dict(fused) if fused else None,
                "text_emotion": dict(text_emo) if text_emo else None,
                "transcript": [
                    {"ts": ts, "text": txt} for ts, txt in transcripts[-8:]
                ] if transcripts else None,
            }

            write_audio_signals(audio_data)
            merge_into_vision_signals(audio_data)

        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Audio Emotion Bridge — feeds acoustic emotion into Advisor-AI pipeline"
    )
    parser.add_argument("--device", type=int, default=None,
                        help="Microphone device index")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print audio devices and exit")
    parser.add_argument("--no-canary", action="store_true",
                        help="Disable Canary-Qwen (acoustic-only mode)")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        sys.exit(f"✘  sounddevice not installed. Run: pip install sounddevice")

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    # GPU
    gpu_device = check_gpu()

    # Load models
    dim_clf, cat_clf = load_emotion_models(gpu_device)

    canary_model = None
    if not args.no_canary:
        canary_model = load_canary_model()

    canary_available = canary_model is not None

    # Shared state and queues
    state    = SharedState()
    stop_evt = threading.Event()

    acoustic_q = queue.Queue(maxsize=4)
    asr_q      = queue.Queue(maxsize=2)

    # Start workers
    threads = []

    t_acoustic = threading.Thread(
        target=inference_worker,
        args=(dim_clf, cat_clf, acoustic_q, state, stop_evt),
        daemon=True, name="acoustic",
    )
    threads.append(t_acoustic)

    if canary_available:
        t_asr = threading.Thread(
            target=asr_worker,
            args=(canary_model, asr_q, state, stop_evt),
            daemon=True, name="asr",
        )
        threads.append(t_asr)

    # Signal publisher — writes to ta_face_signals.json
    t_publisher = threading.Thread(
        target=signal_publisher,
        args=(state, stop_evt),
        daemon=True, name="publisher",
    )
    threads.append(t_publisher)

    for t in threads:
        t.start()

    # Audio ring buffers
    buf_samples  = int(WINDOW_SEC * SAMPLE_RATE)
    step_samples = int(STEP_SEC * SAMPLE_RATE)
    ring = collections.deque(maxlen=buf_samples)
    samples_since_acoustic = 0

    asr_buf_samples = int(ASR_CHUNK_SEC * SAMPLE_RATE)
    asr_ring = collections.deque(maxlen=asr_buf_samples)
    samples_since_asr = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal samples_since_acoustic, samples_since_asr
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        mono_list = mono.tolist()

        ring.extend(mono_list)
        samples_since_acoustic += frames
        if samples_since_acoustic >= step_samples and len(ring) == buf_samples:
            chunk = list(ring)
            samples_since_acoustic = 0
            if not acoustic_q.full():
                acoustic_q.put_nowait(chunk)

        if canary_available:
            asr_ring.extend(mono_list)
            samples_since_asr += frames
            if samples_since_asr >= asr_buf_samples and len(asr_ring) == asr_buf_samples:
                asr_chunk = list(asr_ring)
                samples_since_asr = 0
                asr_ring.clear()
                if not asr_q.full():
                    asr_q.put_nowait(asr_chunk)

    # Banner
    print(f"\n{BOLD}═══════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{C}  ◈  AUDIO EMOTION BRIDGE — Advisor-AI  ◈{RESET}")
    print(f"{BOLD}═══════════════════════════════════════════════{RESET}")
    print(f"{DIM}Writing audio emotion → {BRIDGE_SIGNAL_FILE}{RESET}")
    print(f"{DIM}Merging into vision  → {SIGNAL_FILE}{RESET}")
    print(f"{DIM}Acoustic: Window={WINDOW_SEC}s Step={STEP_SEC}s{RESET}")
    if canary_available:
        print(f"{DIM}Canary ASR: Chunk={ASR_CHUNK_SEC}s | Fusion={ACOUSTIC_WEIGHT:.0%}A/{TEXT_WEIGHT:.0%}T{RESET}")
    print(f"{DIM}Press Ctrl+C to stop{RESET}\n")

    t_start = time.time()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(0.1 * SAMPLE_RATE),
            device=args.device,
            callback=audio_callback,
        ):
            while True:
                # Periodic status print
                with state.lock:
                    result = state.latest_acoustic

                elapsed = time.time() - t_start
                if result and result.get("type") == "emotion":
                    label = result["label"]
                    conf = result["conf"]
                    print(f"\r  {G}●{RESET} {label.upper():12} "
                          f"{conf:.0%} conf  "
                          f"V={result.get('valence', 0):+.2f} "
                          f"A={result.get('arousal', 0):+.2f}  "
                          f"{DIM}[{elapsed:.0f}s]{RESET}   ",
                          end="", flush=True)
                elif result and result.get("type") == "silence":
                    print(f"\r  {DIM}● Listening... "
                          f"RMS={result.get('rms', 0):.4f}  "
                          f"[{elapsed:.0f}s]{RESET}   ",
                          end="", flush=True)

                time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n\n{G}Audio Emotion Bridge stopped.{RESET}")
    finally:
        stop_evt.set()
        for t in threads:
            t.join(timeout=3)

        # Cleanup temp files
        tmp_dir = os.path.join(tempfile.gettempdir(), "emo_canary")
        if os.path.isdir(tmp_dir):
            for f in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except OSError:
                    pass


if __name__ == "__main__":
    main()
