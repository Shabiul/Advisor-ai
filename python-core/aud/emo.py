"""
emo.py — Multimodal Real-Time Emotion Intelligence Engine
══════════════════════════════════════════════════════════════
Combines two inference tiers into a single terminal dashboard:

  TIER 1 — Acoustic Emotion  (fast, every ~1 s)
      Wav2Vec2 models detect vocal emotion + VAD dimensions.

  TIER 2 — Speech Understanding  (slower, every ~5 s)
      NVIDIA Canary-Qwen-2.5B transcribes speech and provides
      LLM-powered summarization & interactive Q&A.

Install deps (once):
    pip install sounddevice torch transformers safetensors numpy
    pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"

Run:
    python emo.py
    python emo.py --device 2          # pick a specific mic index
    python emo.py --list-devices
    python emo.py --no-canary         # run without Canary (acoustic only)

Keyboard shortcuts while running:
    S  — generate a session summary (Canary LLM)
    Q  — ask a question about the conversation (Canary LLM)
   Ctrl-C — stop
"""

import argparse
import collections
import os
import queue
import struct
import sys
import tempfile
import threading
import time
import wave

# Fix for Windows console UnicodeEncodeError with symbols
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════

# ── Base directory (this file's location) ────
_EMO_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Wav2Vec2 emotion models ──────────────────
DIMENSIONAL_MODEL = os.path.join(_EMO_DIR, "local_emotion_models", "dimensional_model")
CATEGORICAL_MODEL = os.path.join(_EMO_DIR, "local_emotion_models", "categorical_model")

# ── NVIDIA Canary-Qwen ───────────────────────
CANARY_MODEL_NAME = "nvidia/canary-qwen-2.5b"
CANARY_LOCAL_DIR  = os.path.join(_EMO_DIR, "canary_model")

# ── Acoustic loop timing ─────────────────────
WINDOW_SEC   = 3.0      # analysis window length (seconds)
STEP_SEC     = 1.0      # how often to run acoustic inference
MIN_CONF     = 0.25     # ignore results below this confidence
SAMPLE_RATE  = 16000
HISTORY_LEN  = 8        # past emotion results in timeline

# ── ASR loop timing ──────────────────────────
ASR_CHUNK_SEC        = 5.0    # transcription chunk length
TRANSCRIPT_LINES     = 8      # lines of transcript in the dashboard
SESSION_BUFFER_LINES = 200    # rolling transcript for LLM context

# ── Audio preprocessing ──────────────────
NOISE_GATE_RMS  = 0.005
NOISE_GATE_MIN  = 0.0003   # absolute floor — below this is true silence
EMA_ALPHA       = 0.35
TEMPERATURE     = 2.0
AUTO_GAIN_TARGET = 0.25    # target peak level after gain

# ── Multimodal fusion weights ────────────────
ACOUSTIC_WEIGHT = 0.6
TEXT_WEIGHT      = 0.4


# ═══════════════════════════════════════════════
#  COLOUR CODES
# ═══════════════════════════════════════════════
R  = "\033[91m";  G  = "\033[92m";  Y  = "\033[93m"
B  = "\033[94m";  M  = "\033[95m";  C  = "\033[96m"
W  = "\033[97m";  DIM = "\033[2m";  RESET = "\033[0m"; BOLD = "\033[1m"

EMOTION_COLOURS = {
    "happy":     G,  "neutral":   W,  "sad":       B,
    "angry":     R,  "fear":      M,  "disgust":   Y,
    "fearful":   M,  "surprised": C,  "calm":      C,
}


# ═══════════════════════════════════════════════
#  TEXT EMOTION KEYWORD ANALYSER
#  Fast, zero-cost sentiment from transcript text.
# ═══════════════════════════════════════════════
TEXT_EMOTION_KEYWORDS = {
    "happy":     ["happy", "great", "awesome", "love", "excited", "wonderful",
                  "amazing", "fantastic", "good", "nice", "joy", "glad",
                  "pleased", "delighted", "thrilled", "brilliant", "excellent"],
    "sad":       ["sad", "sorry", "miss", "lonely", "depressed", "unfortunately",
                  "loss", "grief", "cry", "tears", "upset", "heartbroken",
                  "painful", "miserable", "hopeless", "regret"],
    "angry":     ["angry", "hate", "furious", "annoyed", "frustrated", "mad",
                  "terrible", "worst", "stupid", "ridiculous", "unacceptable",
                  "outrageous", "pissed", "rage", "hostile"],
    "fear":      ["scared", "afraid", "worry", "anxious", "nervous", "terrified",
                  "panic", "danger", "threat", "risk", "dread", "horror",
                  "frightened", "uneasy"],
    "surprised": ["surprised", "shocked", "wow", "unbelievable", "unexpected",
                  "incredible", "really", "seriously", "astonishing", "stunned"],
    "disgust":   ["disgusting", "gross", "awful", "horrible", "nasty", "sick",
                  "revolting", "repulsive", "vile", "appalling"],
    "calm":      ["calm", "relaxed", "peaceful", "quiet", "steady", "fine",
                  "okay", "alright", "serene", "composed"],
    "neutral":   [],
}


def analyse_text_emotion(text):
    """
    Score transcript text against keyword dictionaries.
    Returns a dict of {emotion: score} that sums to 1.
    """
    if not text or not text.strip():
        return {e: (1.0 if e == "neutral" else 0.0) for e in TEXT_EMOTION_KEYWORDS}

    words = text.lower().split()
    scores = {}
    for emotion, keywords in TEXT_EMOTION_KEYWORDS.items():
        score = sum(1 for w in words if any(kw in w for kw in keywords))
        scores[emotion] = float(score)

    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    else:
        # No keyword hits — default to neutral
        scores = {e: (1.0 if e == "neutral" else 0.0) for e in TEXT_EMOTION_KEYWORDS}
    return scores


# ═══════════════════════════════════════════════
#  MULTIMODAL FUSION
# ═══════════════════════════════════════════════
def fuse_emotions(acoustic_scores, text_scores):
    """
    Weighted blend of acoustic emotion (Wav2Vec2) and
    text emotion (keyword analysis) into a single distribution.
    """
    all_labels = set(acoustic_scores.keys()) | set(text_scores.keys())
    fused = {}
    for label in all_labels:
        a = acoustic_scores.get(label, 0.0)
        t = text_scores.get(label, 0.0)
        fused[label] = ACOUSTIC_WEIGHT * a + TEXT_WEIGHT * t

    total = sum(fused.values())
    if total > 0:
        fused = {k: v / total for k, v in fused.items()}
    return fused


# ═══════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════
def check_gpu():
    if not torch.cuda.is_available():
        print(f"{Y}⚠  No CUDA GPU — falling back to CPU (may lag).{RESET}")
        return -1
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{G}✔  GPU:{RESET} {name}  ({mem:.1f} GB VRAM)")
    return 0


def load_emotion_models(device):
    from transformers import pipeline as hf_pipeline
    print(f"\n{BOLD}Loading Wav2Vec2 emotion models...{RESET}")
    t0 = time.time()
    dim_clf = hf_pipeline(
        "audio-classification", model=DIMENSIONAL_MODEL,
        device=device, torch_dtype=torch.float32,
    )
    cat_clf = hf_pipeline(
        "audio-classification", model=CATEGORICAL_MODEL,
        device=device, torch_dtype=torch.float32,
    )
    print(f"{G}✔  Emotion models loaded in {time.time()-t0:.1f}s{RESET}")
    return dim_clf, cat_clf


def load_canary_model():
    """
    Attempt to load NVIDIA Canary-Qwen-2.5B via NeMo.
    Returns the model object or None on failure.
    """
    try:
        from nemo.collections.speechlm2.models import SALM
    except ImportError:
        print(f"{Y}⚠  NeMo not installed — Canary features disabled.{RESET}")
        print(f"{DIM}   Install: pip install \"nemo_toolkit[asr] @ "
              f"git+https://github.com/NVIDIA/NeMo.git\"{RESET}")
        return None

    print(f"\n{BOLD}Loading Canary-Qwen-2.5B  (this may take a minute)...{RESET}")
    t0 = time.time()

    # Try local directory first, then fall back to remote
    if os.path.isdir(CANARY_LOCAL_DIR) and os.path.exists(
        os.path.join(CANARY_LOCAL_DIR, "model.safetensors")
    ):
        print(f"{DIM}   Loading from local: {CANARY_LOCAL_DIR}{RESET}")
        source = CANARY_LOCAL_DIR
    else:
        print(f"{DIM}   Downloading from HuggingFace: {CANARY_MODEL_NAME}{RESET}")
        source = CANARY_MODEL_NAME

    try:
        model = SALM.from_pretrained(source)
        model.eval()
        print(f"{G}✔  Canary-Qwen-2.5B loaded in {time.time()-t0:.1f}s{RESET}")
        return model
    except Exception as exc:
        print(f"{Y}⚠  Failed to load Canary: {exc}{RESET}")
        print(f"{DIM}   Continuing in acoustic-only mode.{RESET}")
        return None


def parse_dims(raw):
    d = {item["label"].lower(): item["score"] for item in raw}
    return {
        "valence":   d.get("valence",   d.get("v", 0.0)),
        "arousal":   d.get("arousal",   d.get("a", 0.0)),
        "dominance": d.get("dominance", d.get("d", 0.0)),
    }


def vad_quadrant(v, a):
    if v >= 0 and a >= 0: return "Excited / Happy"
    if v <  0 and a >= 0: return "Angry / Fearful"
    if v >= 0 and a <  0: return "Calm / Content"
    return                       "Sad / Depressed"


def bar(value, width=18):
    norm   = max(0.0, min(1.0, (value + 1) / 2))
    filled = int(norm * width)
    return "█" * filled + "░" * (width - filled)


def conf_bar(score, width=14):
    filled = int(max(0.0, min(1.0, score)) * width)
    return "█" * filled + "░" * (width - filled)


# ═══════════════════════════════════════════════
#  AUDIO PREPROCESSING
# ═══════════════════════════════════════════════
def preprocess_audio(chunk):
    """Peak-normalize the audio chunk to [-1, 1]."""
    audio = np.array(chunk, dtype=np.float32)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
    return audio


def compute_rms(chunk):
    """Compute RMS energy of an audio chunk."""
    audio = np.array(chunk, dtype=np.float32)
    return float(np.sqrt(np.mean(audio ** 2)))


def save_temp_wav(audio_np, sample_rate=16000):
    """
    Save a numpy float32 audio array to a temporary WAV file.
    Returns the absolute path to the file.
    """
    tmp_dir = os.path.join(tempfile.gettempdir(), "emo_canary")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"chunk_{int(time.time()*1000)}.wav")

    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return path


# ═══════════════════════════════════════════════
#  SOFTMAX WITH TEMPERATURE
# ═══════════════════════════════════════════════
def sharpen_scores(scores_dict, temperature=TEMPERATURE):
    labels = list(scores_dict.keys())
    values = np.array([scores_dict[l] for l in labels], dtype=np.float64)
    values_scaled = values * temperature
    values_scaled -= values_scaled.max()
    exp_vals = np.exp(values_scaled)
    probs = exp_vals / exp_vals.sum()
    return {l: float(p) for l, p in zip(labels, probs)}


# ═══════════════════════════════════════════════
#  TEMPORAL SMOOTHER (EMA)
# ═══════════════════════════════════════════════
class EmotionSmoother:
    """
    Exponential moving average across category scores.
    """
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.ema_scores = None
        self.ema_dims   = None

    def update(self, raw_scores, raw_dims):
        if self.ema_scores is None:
            self.ema_scores = dict(raw_scores)
            self.ema_dims   = dict(raw_dims)
        else:
            for k in raw_scores:
                prev = self.ema_scores.get(k, 0.0)
                self.ema_scores[k] = self.alpha * raw_scores[k] + (1 - self.alpha) * prev
            for k in raw_dims:
                prev = self.ema_dims.get(k, 0.0)
                self.ema_dims[k] = self.alpha * raw_dims[k] + (1 - self.alpha) * prev

        total = sum(self.ema_scores.values())
        if total > 0:
            normed = {k: v / total for k, v in self.ema_scores.items()}
        else:
            normed = dict(self.ema_scores)
        return normed, dict(self.ema_dims)


# ═══════════════════════════════════════════════
#  SHARED STATE  (thread-safe via locks)
# ═══════════════════════════════════════════════
class SharedState:
    """
    Central store for all cross-thread data.
    Every field is guarded by the lock.
    """
    def __init__(self):
        self.lock = threading.Lock()

        # Acoustic emotion (Tier 1)
        self.latest_acoustic  = None          # dict from inference_worker
        self.emotion_history  = collections.deque(maxlen=HISTORY_LEN)

        # Transcription (Tier 2)
        self.transcript_lines = collections.deque(maxlen=TRANSCRIPT_LINES)
        self.session_buffer   = collections.deque(maxlen=SESSION_BUFFER_LINES)

        # Text emotion + fusion
        self.latest_text_emotion = None       # dict from text keyword analyser
        self.latest_fused        = None       # fused scores

        # LLM output
        self.llm_output      = ""             # latest summary / answer
        self.llm_busy        = False          # True while LLM is generating
        self.llm_label       = ""             # "SUMMARY" or "ANSWER"

        # Misc
        self.canary_ready    = False
        self.question_mode   = False          # True when waiting for user input
        self.active_noise_gate = NOISE_GATE_RMS   # overridden after calibration


# ═══════════════════════════════════════════════
#  WORKER 1 — ACOUSTIC INFERENCE  (Tier 1)
# ═══════════════════════════════════════════════
def inference_worker(dim_clf, cat_clf, in_q, state, stop_event):
    smoother = EmotionSmoother(alpha=EMA_ALPHA)

    while not stop_event.is_set():
        try:
            audio_chunk = in_q.get(timeout=0.5)
        except queue.Empty:
            continue

        rms = compute_rms(audio_chunk)
        with state.lock:
            gate = state.active_noise_gate
        if rms < gate:
            with state.lock:
                state.latest_acoustic = {
                    "type": "silence", "ts": time.strftime("%H:%M:%S"), "rms": rms
                }
            continue

        audio_norm = preprocess_audio(audio_chunk)
        inp = {"array": audio_norm, "sampling_rate": SAMPLE_RATE}
        try:
            cat_raw = cat_clf(inp)
            dim_raw = dim_clf(inp)
        except Exception as exc:
            print(f"\n{Y}⚠  Acoustic error: {exc}{RESET}")
            continue

        raw_scores = {x["label"].lower(): x["score"] for x in cat_raw}
        raw_dims   = parse_dims(dim_raw)
        sharpened  = sharpen_scores(raw_scores)
        smoothed_scores, smoothed_dims = smoother.update(sharpened, raw_dims)

        label = max(smoothed_scores, key=smoothed_scores.get)
        conf  = smoothed_scores[label]

        result = {
            "type": "emotion", "ts": time.strftime("%H:%M:%S"),
            "label": label, "conf": conf, "all": smoothed_scores,
            "rms": rms, **smoothed_dims,
        }

        with state.lock:
            state.latest_acoustic = result
            if result["type"] == "emotion":
                state.emotion_history.append(result)

                # Multimodal fusion if we have text scores
                if state.latest_text_emotion:
                    state.latest_fused = fuse_emotions(
                        smoothed_scores, state.latest_text_emotion
                    )


# ═══════════════════════════════════════════════
#  WORKER 2 — ASR TRANSCRIPTION  (Tier 2)
# ═══════════════════════════════════════════════
def asr_worker(canary_model, in_q, state, stop_event):
    """
    Receives non-overlapping audio chunks, transcribes them
    via Canary-Qwen ASR mode, and updates the shared state.
    """
    if canary_model is None:
        return

    with state.lock:
        state.canary_ready = True

    while not stop_event.is_set():
        try:
            audio_chunk = in_q.get(timeout=0.5)
        except queue.Empty:
            continue

        rms = compute_rms(audio_chunk)
        with state.lock:
            gate = state.active_noise_gate
        if rms < gate:
            continue  # skip silence chunks

        # Normalize & save to temp WAV
        audio_norm = preprocess_audio(audio_chunk)
        wav_path = save_temp_wav(audio_norm, SAMPLE_RATE)

        try:
            answer_ids = canary_model.generate(
                prompts=[[{
                    "role": "user",
                    "content": f"Transcribe the following: {canary_model.audio_locator_tag}",
                    "audio": [wav_path],
                }]],
                max_new_tokens=128,
            )
            transcript = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        except Exception as exc:
            print(f"\n{Y}⚠  ASR error: {exc}{RESET}")
            transcript = None
        finally:
            # Clean up temp file
            try:
                os.remove(wav_path)
            except OSError:
                pass

        if transcript and transcript.strip():
            ts = time.strftime("%H:%M:%S")
            text_emotion = analyse_text_emotion(transcript)

            with state.lock:
                state.transcript_lines.append((ts, transcript))
                state.session_buffer.append((ts, transcript))
                state.latest_text_emotion = text_emotion

                # Update fusion immediately with latest acoustic
                if state.latest_acoustic and state.latest_acoustic.get("all"):
                    state.latest_fused = fuse_emotions(
                        state.latest_acoustic["all"], text_emotion
                    )


# ═══════════════════════════════════════════════
#  WORKER 3 — LLM  (Summary / Q&A)
# ═══════════════════════════════════════════════
def llm_worker(canary_model, request_q, state, stop_event):
    """
    Processes user-triggered LLM requests (summary or question).
    Uses Canary-Qwen in LLM mode (text-only, adapter disabled).
    """
    if canary_model is None:
        return

    while not stop_event.is_set():
        try:
            req = request_q.get(timeout=0.5)
        except queue.Empty:
            continue

        with state.lock:
            state.llm_busy = True
            # Build the full transcript context
            lines = list(state.session_buffer)

        if not lines:
            with state.lock:
                state.llm_output = "(No transcript available yet — keep talking!)"
                state.llm_label = req.get("label", "INFO")
                state.llm_busy = False
            continue

        transcript_text = "\n".join(f"[{ts}] {txt}" for ts, txt in lines)

        if req["type"] == "summary":
            prompt = (
                "Below is a timestamped transcript of a live conversation. "
                "Provide a concise summary (3-5 sentences) of the key topics discussed, "
                "the overall emotional tone, and any notable shifts in sentiment.\n\n"
                f"{transcript_text}"
            )
            label = "SESSION SUMMARY"
        elif req["type"] == "question":
            question = req.get("question", "What happened?")
            prompt = (
                "Below is a timestamped transcript of a live conversation.\n\n"
                f"{transcript_text}\n\n"
                f"Answer this question about the conversation: {question}"
            )
            label = f"Q: {question}"
        else:
            continue

        try:
            with canary_model.llm.disable_adapter():
                answer_ids = canary_model.generate(
                    prompts=[[{"role": "user", "content": prompt}]],
                    max_new_tokens=512,
                )
            answer = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        except Exception as exc:
            answer = f"(LLM error: {exc})"

        with state.lock:
            state.llm_output = answer
            state.llm_label  = label
            state.llm_busy   = False


# ═══════════════════════════════════════════════
#  WORKER 4 — KEYBOARD INPUT
# ═══════════════════════════════════════════════
def input_worker(llm_request_q, state, stop_event, canary_available):
    """
    Listens for keyboard input on Windows using msvcrt.
    S = summary, Q = interactive question, Ctrl-C = stop.
    """
    if sys.platform != "win32":
        # On non-Windows, fall back to a simple approach
        return

    import msvcrt

    while not stop_event.is_set():
        if msvcrt.kbhit():
            try:
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
            except Exception:
                continue

            if not canary_available:
                continue

            if key == "s":
                with state.lock:
                    if state.llm_busy:
                        continue
                llm_request_q.put({"type": "summary", "label": "SESSION SUMMARY"})

            elif key == "q":
                # Pause dashboard, get question from user
                with state.lock:
                    state.question_mode = True

                # Flush any buffered keys
                while msvcrt.kbhit():
                    msvcrt.getch()

                sys.stdout.write(f"\n  {BOLD}{C}╔══ ASK A QUESTION ══╗{RESET}\n")
                sys.stdout.write(f"  {C}║{RESET} Type your question and press Enter:\n")
                sys.stdout.write(f"  {C}║{RESET} > ")
                sys.stdout.flush()

                try:
                    question = input().strip()
                except (EOFError, KeyboardInterrupt):
                    question = ""

                with state.lock:
                    state.question_mode = False

                if question:
                    llm_request_q.put({
                        "type": "question",
                        "question": question,
                        "label": f"Q: {question}",
                    })

        time.sleep(0.1)


# ═══════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════
DASHBOARD_LINES = 0


def clear_dashboard():
    global DASHBOARD_LINES
    if DASHBOARD_LINES > 0:
        sys.stdout.write(f"\033[{DASHBOARD_LINES}A\033[J")
        sys.stdout.flush()


def draw_dashboard(state, elapsed_total, canary_available):
    global DASHBOARD_LINES

    with state.lock:
        result       = state.latest_acoustic
        history      = list(state.emotion_history)
        transcripts  = list(state.transcript_lines)
        fused        = state.latest_fused
        text_emo     = state.latest_text_emotion
        llm_output   = state.llm_output
        llm_label    = state.llm_label
        llm_busy     = state.llm_busy
        canary_ready = state.canary_ready
        q_mode       = state.question_mode

    if result is None or q_mode:
        return

    is_listening = result.get("type") == "silence"
    W_ = 60
    lines = []

    def ln(s=""):
        lines.append(f"  {s}")

    # ── Header ────────────────────────────────
    lines.append(f"  {BOLD}{'━' * W_}{RESET}")
    lines.append(f"  {BOLD}{C}  ◈  MULTIMODAL EMOTION INTELLIGENCE  ◈{RESET}")
    tier_status = (
        f"{G}●{RESET} Acoustic  "
        + (f"{G}●{RESET} Canary ASR  {G}●{RESET} LLM" if canary_ready
           else f"{Y}○{RESET} Canary (off)")
    )
    ln(f"{DIM}{tier_status}{RESET}")
    lines.append(f"  {BOLD}{'━' * W_}{RESET}")

    # ── Listening / Emotion ───────────────────
    if is_listening:
        ln(f"{DIM}Listening...  (waiting for speech){RESET}  "
           f"{DIM}[{result['ts']}]{RESET}")
        ln(f"Mic level    {DIM}RMS {result.get('rms', 0):.4f}{RESET}")
        ln(f"Running      {elapsed_total:.0f}s")
    else:
        label = result["label"]
        conf  = result["conf"]
        v, a, d = result["valence"], result["arousal"], result["dominance"]
        ec    = EMOTION_COLOURS.get(label, W)
        quad  = vad_quadrant(v, a)

        # Show fused emotion if available, else acoustic
        if fused:
            fused_label = max(fused, key=fused.get)
            fused_conf  = fused[fused_label]
            fused_ec    = EMOTION_COLOURS.get(fused_label, W)
            ln(f"{BOLD}Multimodal   {fused_ec}{fused_label.upper()}{RESET}  "
               f"({fused_conf:.0%} conf)  {DIM}[{result['ts']}]{RESET}")
            ln(f"{DIM}  acoustic: {ec}{label}{RESET}  "
               f"{DIM}| text: {EMOTION_COLOURS.get(max(text_emo, key=text_emo.get) if text_emo else 'neutral', W)}"
               f"{max(text_emo, key=text_emo.get) if text_emo else 'n/a'}{RESET}")
        else:
            ln(f"{BOLD}Acoustic     {ec}{label.upper()}{RESET}  "
               f"({conf:.0%} conf)  {DIM}[{result['ts']}]{RESET}")

        if conf < MIN_CONF:
            ln(f"{Y}⚠  Low confidence — result may be unreliable{RESET}")
        ln(f"VAD Quadrant   {BOLD}{quad}{RESET}")
        ln(f"Mic level      {DIM}RMS {result.get('rms', 0):.4f}{RESET}")
        ln(f"Running        {elapsed_total:.0f}s")

        # ── Dimensional bars ──────────────────
        lines.append(f"\n  {'─' * W_}")
        ln(f"{BOLD}DIMENSIONAL{RESET}   [-1 ← neutral → +1]")
        lines.append(f"  {'─' * W_}")

        def dim_row(name, val, lo, hi):
            sign = G if val > 0.1 else (R if val < -0.1 else DIM)
            ln(f"{name:<11} {sign}{bar(val)}{RESET}  {sign}{val:+.3f}{RESET}"
               f"  {DIM}[{lo}↔{hi}]{RESET}")

        dim_row("Valence",   v, "neg", "pos")
        dim_row("Arousal",   a, "calm", "exc")
        dim_row("Dominance", d, "sub", "dom")

        # ── All category scores ───────────────
        display_scores = fused if fused else result["all"]
        display_label  = "FUSED" if fused else "SMOOTHED"
        lines.append(f"\n  {'─' * W_}")
        ln(f"{BOLD}ALL CATEGORIES{RESET}  ({display_label.lower()})")
        lines.append(f"  {'─' * W_}")
        top_label = max(display_scores, key=display_scores.get)
        for emo, score in sorted(display_scores.items(), key=lambda x: -x[1]):
            ec2 = EMOTION_COLOURS.get(emo, W)
            marker = f" {BOLD}◄{RESET}" if emo == top_label else ""
            ln(f"{emo:<12} {ec2}{conf_bar(score)}{RESET}  "
               f"{ec2}{score:.0%}{RESET}{marker}")

    # ── Live transcript ───────────────────────
    lines.append(f"\n  {'─' * W_}")
    ln(f"{BOLD}LIVE TRANSCRIPT{RESET}  "
       + (f"{G}●{RESET}" if canary_ready else f"{DIM}(Canary not loaded){RESET}"))
    lines.append(f"  {'─' * W_}")
    if transcripts:
        for ts, txt in transcripts:
            # Truncate long lines for display
            display_txt = txt if len(txt) <= 50 else txt[:47] + "..."
            ln(f"{DIM}{ts}{RESET}  {W}{display_txt}{RESET}")
    else:
        ln(f"{DIM}Waiting for speech...{RESET}")

    # ── Recent emotion timeline ───────────────
    emotion_history = [r for r in history if r.get("type") == "emotion"]
    lines.append(f"\n  {'─' * W_}")
    ln(f"{BOLD}RECENT TIMELINE{RESET}  (newest first)")
    lines.append(f"  {'─' * W_}")
    if emotion_history:
        ln(f"{'TIME':<10} {'EMOTION':<12} {'CONF':>5}  {'V':>5} {'A':>5}")
        for r in reversed(emotion_history):
            ec3  = EMOTION_COLOURS.get(r["label"], W)
            flag = f"  {Y}⚠{RESET}" if r["conf"] < MIN_CONF else ""
            ln(f"{r['ts']:<10} {ec3}{r['label']:<12}{RESET} "
               f"{r['conf']:.0%}   {r['valence']:>+.2f} {r['arousal']:>+.2f}{flag}")
    else:
        ln(f"{DIM}No speech detected yet...{RESET}")

    # ── LLM Insight ───────────────────────────
    if canary_available:
        lines.append(f"\n  {'─' * W_}")
        if llm_busy:
            ln(f"{BOLD}LLM INSIGHT{RESET}  {Y}⏳ generating...{RESET}")
        elif llm_output:
            ln(f"{BOLD}LLM INSIGHT{RESET}  {C}{llm_label}{RESET}")
            lines.append(f"  {'─' * W_}")
            # Word-wrap the LLM output
            for chunk in _wrap_text(llm_output, W_ - 4):
                ln(f"  {chunk}")
        else:
            ln(f"{BOLD}LLM INSIGHT{RESET}  {DIM}Press S for summary, Q for question{RESET}")

    # ── Footer ────────────────────────────────
    lines.append(f"  {BOLD}{'━' * W_}{RESET}")
    shortcuts = f"{DIM}Ctrl-C stop"
    if canary_available:
        shortcuts += f"  │  S summary  │  Q ask question"
    shortcuts += f"{RESET}"
    lines.append(f"  {shortcuts}")

    output = "\n".join(lines)
    clear_dashboard()
    print(output)
    DASHBOARD_LINES = len(lines) + output.count("\n")


def _wrap_text(text, width):
    """Simple word-wrap for LLM output."""
    words = text.split()
    result_lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            result_lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        result_lines.append(current)
    return result_lines if result_lines else [""]


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Real-Time Emotion Intelligence Engine"
    )
    parser.add_argument("--device", type=int, default=None,
                        help="Microphone device index (omit = system default)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio devices and exit")
    parser.add_argument("--no-canary", action="store_true",
                        help="Disable Canary-Qwen (run acoustic-only mode)")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        sys.exit(f"{R}✘  sounddevice not installed.\n"
                 f"   Run:  pip install sounddevice{RESET}")

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    # ── GPU check ─────────────────────────────
    gpu_device = check_gpu()

    # ── Load models ───────────────────────────
    dim_clf, cat_clf = load_emotion_models(gpu_device)

    canary_model = None
    if not args.no_canary:
        canary_model = load_canary_model()

    canary_available = canary_model is not None

    # ── Shared state & queues ─────────────────
    state    = SharedState()
    stop_evt = threading.Event()

    acoustic_q   = queue.Queue(maxsize=4)     # audio → acoustic inference
    asr_q        = queue.Queue(maxsize=2)     # audio → Canary ASR
    llm_req_q    = queue.Queue()              # user commands → LLM worker

    # ── Start worker threads ──────────────────
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

        t_llm = threading.Thread(
            target=llm_worker,
            args=(canary_model, llm_req_q, state, stop_evt),
            daemon=True, name="llm",
        )
        threads.append(t_llm)

    t_input = threading.Thread(
        target=input_worker,
        args=(llm_req_q, state, stop_evt, canary_available),
        daemon=True, name="input",
    )
    threads.append(t_input)

    for t in threads:
        t.start()

    # ── Mic gain calibration ──────────────────
    # Record a short burst to measure actual signal level and compute gain.
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
            mic_gain = min(AUTO_GAIN_TARGET / cal_peak, 500.0)   # cap at 500x
        else:
            mic_gain = 100.0   # fallback for truly silent calibration

        # Adapt noise gate to the actual mic level
        effective_gate = max(cal_rms * mic_gain * 1.5, NOISE_GATE_MIN)
        effective_gate = min(effective_gate, NOISE_GATE_RMS)  # never exceed original

        print(f"{G}✔  Mic calibrated:{RESET} peak={cal_peak:.6f}  "
              f"gain={mic_gain:.1f}x  gate={effective_gate:.5f}")
    except Exception as exc:
        print(f"{Y}⚠  Calibration failed ({exc}), using defaults{RESET}")
        mic_gain = 100.0
        effective_gate = NOISE_GATE_MIN

    # Override the noise gate with the calibrated value
    with state.lock:
        state.active_noise_gate = effective_gate

    # ── Rolling audio buffers ─────────────────
    # Buffer for acoustic inference (overlapping window)
    buf_samples  = int(WINDOW_SEC * SAMPLE_RATE)
    step_samples = int(STEP_SEC   * SAMPLE_RATE)
    ring         = collections.deque(maxlen=buf_samples)
    samples_since_acoustic = 0

    # Buffer for ASR (non-overlapping chunks)
    asr_buf_samples = int(ASR_CHUNK_SEC * SAMPLE_RATE)
    asr_ring        = collections.deque(maxlen=asr_buf_samples)
    samples_since_asr = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal samples_since_acoustic, samples_since_asr
        if status:
            pass
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        # Apply gain to bring low-level mic signals into usable range
        mono = mono * mic_gain
        mono = np.clip(mono, -1.0, 1.0)

        mono_list = mono.tolist()

        # Feed the acoustic ring buffer
        ring.extend(mono_list)
        samples_since_acoustic += frames
        if samples_since_acoustic >= step_samples and len(ring) == buf_samples:
            chunk = list(ring)
            samples_since_acoustic = 0
            if not acoustic_q.full():
                acoustic_q.put_nowait(chunk)

        # Feed the ASR ring buffer (separate, non-overlapping)
        if canary_available:
            asr_ring.extend(mono_list)
            samples_since_asr += frames
            if samples_since_asr >= asr_buf_samples and len(asr_ring) == asr_buf_samples:
                asr_chunk = list(asr_ring)
                samples_since_asr = 0
                asr_ring.clear()
                if not asr_q.full():
                    asr_q.put_nowait(asr_chunk)

    # ── Print startup banner ──────────────────
    print(f"\n{BOLD}Starting microphone capture...{RESET}")
    print(f"{DIM}Acoustic — Window: {WINDOW_SEC}s | Step: {STEP_SEC}s | "
          f"Min confidence: {MIN_CONF:.0%}{RESET}")
    print(f"{DIM}Smoothing: EMA α={EMA_ALPHA} | Temperature: {TEMPERATURE}{RESET}")
    if canary_available:
        print(f"{DIM}Canary ASR — Chunk: {ASR_CHUNK_SEC}s | "
              f"Multimodal fusion: {ACOUSTIC_WEIGHT:.0%} acoustic / "
              f"{TEXT_WEIGHT:.0%} text{RESET}")
        print(f"{DIM}Press S for session summary, Q to ask a question{RESET}")
    print(f"{DIM}Filling buffer ({WINDOW_SEC:.0f}s) before first result...{RESET}\n")

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
                    q_mode = state.question_mode

                if not q_mode:
                    draw_dashboard(state, time.time() - t_start, canary_available)

                time.sleep(0.25)

    except KeyboardInterrupt:
        print(f"\n\n{G}Stopped.{RESET}")
    finally:
        stop_evt.set()
        for t in threads:
            t.join(timeout=3)

        # Clean up temp directory
        tmp_dir = os.path.join(tempfile.gettempdir(), "emo_canary")
        if os.path.isdir(tmp_dir):
            for f in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except OSError:
                    pass


if __name__ == "__main__":
    main()
