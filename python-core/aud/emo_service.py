"""
emo_service.py — Headless Audio Emotion Service for Advisor-AI
═══════════════════════════════════════════════════════════════
Runs the acoustic emotion engine as a background service and
outputs results via TWO channels simultaneously:

  1. JSON file  → ta_face_signals.json  (for Streamlit dashboard)
  2. HTTP POST  → /audio_emotion        (for Node.js web dashboard)

Usage:
    python -m aud.emo_service
    python -m aud.emo_service --no-canary
    python -m aud.emo_service --device 2
    python -m aud.emo_service --backend-url http://localhost:3000
"""

import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
import urllib.request
import urllib.error

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

try:
    from .emo import (
        EmotionSmoother, SharedState, analyse_text_emotion, fuse_emotions,
        preprocess_audio, compute_rms, sharpen_scores, parse_dims, vad_quadrant,
        load_emotion_models, load_canary_model, check_gpu,
        inference_worker, asr_worker,
        SAMPLE_RATE, WINDOW_SEC, STEP_SEC, ASR_CHUNK_SEC,
        NOISE_GATE_RMS, NOISE_GATE_MIN, AUTO_GAIN_TARGET,
        EMA_ALPHA, ACOUSTIC_WEIGHT, TEXT_WEIGHT,
    )
except ImportError:
    from emo import (
        EmotionSmoother, SharedState, analyse_text_emotion, fuse_emotions,
        preprocess_audio, compute_rms, sharpen_scores, parse_dims, vad_quadrant,
        load_emotion_models, load_canary_model, check_gpu,
        inference_worker, asr_worker,
        SAMPLE_RATE, WINDOW_SEC, STEP_SEC, ASR_CHUNK_SEC,
        NOISE_GATE_RMS, NOISE_GATE_MIN, AUTO_GAIN_TARGET,
        EMA_ALPHA, ACOUSTIC_WEIGHT, TEXT_WEIGHT,
    )

# Colour codes for terminal output
G  = "\033[92m"; Y  = "\033[93m"; C  = "\033[96m"
W  = "\033[97m"; DIM = "\033[2m"; RESET = "\033[0m"; BOLD = "\033[1m"


# ── Signal file paths (for Streamlit dashboard) ──────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIGNAL_FILE = os.path.join(_PROJECT_ROOT, "ta_face_signals.json")
AUDIO_SIGNAL_FILE = os.path.join(_PROJECT_ROOT, "ta_audio_signals.json")


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


def _atomic_json_write(filepath, data):
    """Atomically write JSON data to a file (temp + rename)."""
    try:
        tmp = filepath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        os.replace(tmp, filepath)
    except OSError:
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
        except Exception:
            pass


class BackendAudioPoster:
    """
    Periodically reads SharedState and outputs audio emotion data
    via two channels:
      1. Writes to ta_face_signals.json (for Streamlit)
      2. HTTP-POSTs to Node.js backend (for web dashboard)
    """

    def __init__(self, backend_url, state, stop_event):
        self._url = f"{backend_url.rstrip('/')}/audio_emotion"
        self._state = state
        self._stop = stop_event
        self._running = True

    def run(self):
        while not self._stop.is_set() and self._running:
            payload = self._build_payload()
            if payload:
                # Channel 1: Write to JSON file (Streamlit)
                self._write_to_signals(payload)
                # Channel 2: HTTP POST to Node.js backend
                self._post(payload)
            time.sleep(0.5)

    def _build_payload(self):
        with self._state.lock:
            acoustic = self._state.latest_acoustic
            fused = self._state.latest_fused
            text_emo = self._state.latest_text_emotion
            transcripts = (
                list(self._state.transcript_lines)
                if hasattr(self._state, "transcript_lines")
                else []
            )

        if not acoustic or acoustic.get("type") != "emotion":
            # Still report silence so dashboard knows mic is active
            if acoustic and acoustic.get("type") == "silence":
                return {
                    "audio_emotion": {
                        "status": "silence",
                        "rms": round(acoustic.get("rms", 0), 6),
                        "timestamp": acoustic.get("ts", ""),
                    }
                }
            return None

        audio_emotion = {
            "status": "active",
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
            "timestamp": acoustic.get("ts", ""),
        }

        result = {"audio_emotion": audio_emotion}

        # Fused emotion (acoustic + text)
        if fused:
            fused_label = max(fused, key=fused.get)
            result["fused_emotion"] = {
                "label": fused_label,
                "confidence": round(fused.get(fused_label, 0), 4),
                "all_scores": dict(fused),
                "acoustic_weight": ACOUSTIC_WEIGHT,
                "text_weight": TEXT_WEIGHT,
            }

        # Text emotion
        if text_emo:
            text_label = max(text_emo, key=text_emo.get)
            result["text_emotion"] = {
                "label": text_label,
                "all_scores": dict(text_emo),
            }

        # Transcript
        if transcripts:
            result["transcript"] = [
                {"ts": ts, "text": txt}
                for ts, txt in transcripts[-8:]
            ]

        return result

    def _write_to_signals(self, payload):
        """Write audio emotion into ta_face_signals.json for Streamlit."""
        ae = payload.get("audio_emotion", {})
        if ae.get("status") == "silence":
            return  # Don't write silence to Streamlit

        # Write standalone audio signal file
        audio_data = dict(ae)
        audio_data["timestamp"] = int(time.time())
        if payload.get("fused_emotion"):
            audio_data["fused_scores"] = payload["fused_emotion"].get("all_scores")
        if payload.get("transcript"):
            audio_data["transcript"] = payload["transcript"]
        _atomic_json_write(AUDIO_SIGNAL_FILE, audio_data)

        # Merge into vision signal file (so Streamlit sees it)
        try:
            if os.path.exists(SIGNAL_FILE):
                with open(SIGNAL_FILE, "r") as f:
                    vision = json.load(f)
            else:
                vision = {}
        except (json.JSONDecodeError, IOError):
            vision = {}

        vision["audio_emotion"] = {
            "label": ae.get("label", "unknown"),
            "confidence": ae.get("confidence", 0),
            "all_scores": ae.get("all_scores", {}),
            "valence": ae.get("valence", 0),
            "arousal": ae.get("arousal", 0),
            "dominance": ae.get("dominance", 0),
            "vad_quadrant": ae.get("vad_quadrant", ""),
            "source": "wav2vec2",
        }

        if payload.get("fused_emotion"):
            fused = payload["fused_emotion"]
            vision["multimodal_emotion"] = {
                "label": fused.get("label", "unknown"),
                "confidence": fused.get("confidence", 0),
                "all_scores": fused.get("all_scores", {}),
                "source": "acoustic+text_fusion",
            }

        if payload.get("transcript"):
            vision["live_transcript"] = payload["transcript"]

        _atomic_json_write(SIGNAL_FILE, vision)

    def _post(self, payload):
        """HTTP POST to Node.js backend."""
        try:
            data = json.dumps(payload, cls=NumpyEncoder).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except (urllib.error.URLError, OSError):
            pass
        except Exception as e:
            print(f"  {Y}⚠ POST error: {e}{RESET}")

    def stop(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(
        description="Audio Emotion Service — posts to Advisor-AI backend"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Microphone device index (omit = system default)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print available audio devices and exit",
    )
    parser.add_argument(
        "--no-canary", action="store_true",
        help="Disable Canary-Qwen (acoustic-only mode)",
    )
    parser.add_argument(
        "--backend-url", type=str,
        default=os.environ.get("ADVISOR_BACKEND_URL", "http://localhost:3000"),
        help="Node.js backend base URL (default http://localhost:3000)",
    )
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        sys.exit(f"✘  sounddevice not installed. Run: pip install sounddevice")

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    # GPU check
    gpu_device = check_gpu()

    # Load models
    dim_clf, cat_clf = load_emotion_models(gpu_device)

    canary_model = None
    if not args.no_canary:
        canary_model = load_canary_model()

    canary_available = canary_model is not None

    # Shared state and queues
    state = SharedState()
    stop_evt = threading.Event()

    acoustic_q = queue.Queue(maxsize=4)
    asr_q = queue.Queue(maxsize=2)

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

    # Backend poster
    poster = BackendAudioPoster(args.backend_url, state, stop_evt)
    t_poster = threading.Thread(
        target=poster.run,
        daemon=True, name="poster",
    )
    threads.append(t_poster)

    for t in threads:
        t.start()

    # ── Mic gain calibration ──────────────────
    print(f"\n{BOLD}Calibrating microphone gain...{RESET}")
    print(f"{DIM}  Speak normally for 1 second...{RESET}")
    try:
        cal_audio = sd.rec(
            int(1.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
            channels=1, dtype="float32", device=args.device,
        )
        sd.wait()
        cal_audio = cal_audio.flatten()
        cal_peak = float(np.max(np.abs(cal_audio)))
        cal_rms  = float(np.sqrt(np.mean(cal_audio ** 2)))

        if cal_peak > 1e-7:
            mic_gain = min(AUTO_GAIN_TARGET / cal_peak, 500.0)
        else:
            mic_gain = 100.0

        effective_gate = max(cal_rms * mic_gain * 1.5, NOISE_GATE_MIN)
        effective_gate = min(effective_gate, NOISE_GATE_RMS)

        print(f"{G}\u2714  Mic calibrated:{RESET} peak={cal_peak:.6f}  "
              f"gain={mic_gain:.1f}x  gate={effective_gate:.5f}")
    except Exception as exc:
        print(f"{Y}\u26a0  Calibration failed ({exc}), using defaults{RESET}")
        mic_gain = 100.0
        effective_gate = NOISE_GATE_MIN

    with state.lock:
        state.active_noise_gate = effective_gate

    # Audio ring buffers
    buf_samples = int(WINDOW_SEC * SAMPLE_RATE)
    step_samples = int(STEP_SEC * SAMPLE_RATE)
    ring = collections.deque(maxlen=buf_samples)
    samples_since_acoustic = 0

    asr_buf_samples = int(ASR_CHUNK_SEC * SAMPLE_RATE)
    asr_ring = collections.deque(maxlen=asr_buf_samples)
    samples_since_asr = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal samples_since_acoustic, samples_since_asr
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        # Apply gain to bring low-level mic signals into usable range
        mono = mono * mic_gain
        mono = np.clip(mono, -1.0, 1.0)

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
    print(f"{BOLD}{C}  ◈  AUDIO EMOTION SERVICE — Advisor-AI  ◈{RESET}")
    print(f"{BOLD}═══════════════════════════════════════════════{RESET}")
    print(f"{DIM}Signal file:  {SIGNAL_FILE}{RESET}")
    print(f"{DIM}Audio file:   {AUDIO_SIGNAL_FILE}{RESET}")
    print(f"{DIM}Backend URL:  {args.backend_url}{RESET}")
    print(f"{DIM}POST endpoint: {args.backend_url.rstrip('/')}/audio_emotion{RESET}")
    print(f"{DIM}Acoustic: Window={WINDOW_SEC}s Step={STEP_SEC}s{RESET}")
    if canary_available:
        print(f"{DIM}Canary ASR: Chunk={ASR_CHUNK_SEC}s | "
              f"Fusion={ACOUSTIC_WEIGHT:.0%}A/{TEXT_WEIGHT:.0%}T{RESET}")
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
                with state.lock:
                    result = state.latest_acoustic

                elapsed = time.time() - t_start
                if result and result.get("type") == "emotion":
                    label = result["label"]
                    conf = result["conf"]
                    print(
                        f"\r  {G}●{RESET} {label.upper():12} "
                        f"{conf:.0%} conf  "
                        f"V={result.get('valence', 0):+.2f} "
                        f"A={result.get('arousal', 0):+.2f}  "
                        f"{DIM}→ backend  [{elapsed:.0f}s]{RESET}   ",
                        end="", flush=True,
                    )
                elif result and result.get("type") == "silence":
                    print(
                        f"\r  {DIM}● Listening... "
                        f"RMS={result.get('rms', 0):.4f}  "
                        f"[{elapsed:.0f}s]{RESET}   ",
                        end="", flush=True,
                    )

                time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n\n{G}Audio Emotion Service stopped.{RESET}")
    finally:
        stop_evt.set()
        poster.stop()
        for t in threads:
            t.join(timeout=3)


if __name__ == "__main__":
    main()
