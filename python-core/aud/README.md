# Multimodal Real-Time Emotion Intelligence Engine

A real-time, terminal-based speech emotion and behavioral intelligence system. Captures live audio from your microphone and runs a **two-tier inference pipeline**:

- **Tier 1 — Acoustic Emotion** (Wav2Vec2, ~1s latency): Detects vocal emotion and VAD dimensions from tone of voice.
- **Tier 2 — Speech Understanding** (NVIDIA Canary-Qwen-2.5B, ~5s latency): Transcribes speech, performs multimodal emotion fusion, and enables LLM-powered summarization & interactive Q&A.

Everything runs locally. No cloud APIs required.

---

## Features

- **Real-Time Acoustic Emotion Detection**
  Analyzes your voice every ~1 second using a sliding audio window for instant feedback.

- **Live Transcription (Real-Time Subtitles)**
  NVIDIA Canary-Qwen-2.5B transcribes your speech in real time, displayed as a rolling subtitle feed on the dashboard.

- **True Multimodal Emotion Detection**
  Fuses acoustic emotion (how you sound) with text emotion (what you say) into a single, more accurate prediction. Configurable weighting (default: 60% acoustic / 40% text).

- **Speech Summarization & Topic Tracking**
  Press `S` at any time to generate a concise LLM-powered summary of the session — covering key topics, emotional tone, and sentiment shifts.

- **Interactive Behavioral Assistant**
  Press `Q` to ask a free-form question about the conversation (e.g., *"Did the speaker sound defensive when discussing the budget?"*). Canary-Qwen's built-in LLM reasons over the full transcript to answer.

- **Beautiful Terminal Dashboard**
  Color-coded emotions, confidence bars, VAD dimensional bars, live transcript feed, emotion timeline, and LLM insight panel — all in one view.

- **Graceful Degradation**
  If NeMo/Canary is not installed, the system automatically falls back to acoustic-only mode with all Tier 1 features intact.

---

## Models Used

| Model | Source | Purpose |
|-------|--------|---------|
| `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | HuggingFace | Categorical emotion (Happy, Sad, Angry, Neutral, Fear, Disgust, Surprised, Calm) |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | HuggingFace | Dimensional emotion (Valence, Arousal, Dominance) |
| `nvidia/canary-qwen-2.5b` | HuggingFace / NVIDIA NeMo | Live transcription (ASR mode) + Summarization & Q&A (LLM mode) |

---

## Project Structure

| File | Description |
|------|-------------|
| `emo.py` | Main script — runs the real-time multimodal dashboard |
| `download_models.py` | Downloads and formats Wav2Vec2 emotion models from HuggingFace |
| `canary_model/` | Local copy of the `nvidia/canary-qwen-2.5b` model weights |
| `local_emotion_models/` | Local Wav2Vec2 categorical and dimensional model weights |

---

## Prerequisites

- **Python 3.8+**
- **A functioning microphone**
- **CUDA-compatible GPU** with **12 GB+ VRAM** recommended (for running all three models simultaneously). Falls back to CPU if no GPU is available.

---

## Installation

### 1. Core dependencies (Acoustic Emotion)

```bash
pip install torch transformers safetensors numpy sounddevice
```

> To enable GPU acceleration, install the CUDA version of PyTorch from the [official PyTorch website](https://pytorch.org/).

### 2. Download Wav2Vec2 emotion models

```bash
python download_models.py
```

### 3. Canary-Qwen dependencies (optional but recommended)

```bash
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
```

> Requires PyTorch 2.6+ for FSDP2 support.

### 4. Download Canary-Qwen-2.5B model

The model weights should be placed in `canary_model/`. If not already downloaded:

```bash
python canary_model/down.py
```

---

## Usage

### Start the full multimodal engine

```bash
python emo.py
```

### Acoustic-only mode (skip Canary)

```bash
python emo.py --no-canary
```

### List audio devices / select a specific mic

```bash
python emo.py --list-devices
python emo.py --device <ID>
```

### Keyboard Shortcuts (while running)

| Key | Action |
|-----|--------|
| `S` | Generate a session summary (Canary LLM) |
| `Q` | Ask a question about the conversation (Canary LLM) |
| `Ctrl+C` | Stop |

---

## Architecture

```
Microphone
    │
    ├──▶ [Acoustic Ring Buffer — 3s window, 1s step]
    │        └──▶ Wav2Vec2 Categorical → emotion label + confidence
    │        └──▶ Wav2Vec2 Dimensional → valence / arousal / dominance
    │                  │
    │                  ▼
    │          ┌──────────────┐
    │          │  EMA Smoother │──── acoustic scores
    │          └──────────────┘            │
    │                                      │
    ├──▶ [ASR Ring Buffer — 5s chunks]     │       ┌──────────────────┐
    │        └──▶ Canary-Qwen ASR ──▶ transcript ──▶│  Text Keyword    │
    │                                      │       │  Emotion Analyzer │
    │                                      │       └──────────────────┘
    │                                      │                │
    │                                      ▼                ▼
    │                               ┌────────────────────────────┐
    │                               │   Multimodal Fusion        │
    │                               │   (60% acoustic + 40% text)│
    │                               └────────────────────────────┘
    │                                           │
    │                                           ▼
    │                                  Terminal Dashboard
    │
    └──▶ [On-Demand: S / Q keys]
             └──▶ Canary-Qwen LLM Mode ──▶ Summary / Answer
```

---

## How It Works

1. **Audio capture** runs continuously via `sounddevice`, feeding two separate ring buffers.
2. **Acoustic inference** (Tier 1) runs every ~1 second on a 3-second sliding window. Results are temporally smoothed with EMA and sharpened with temperature-scaled softmax.
3. **ASR inference** (Tier 2) runs every ~5 seconds on non-overlapping chunks. Each transcript is analyzed for keyword-based text emotion and fused with the acoustic scores.
4. **LLM inference** is triggered on-demand by the user pressing `S` (summary) or `Q` (question). It operates in Canary-Qwen's LLM mode with adapter disabled, reasoning over the full session transcript.
5. All inference runs on **separate daemon threads** so the dashboard never freezes.
