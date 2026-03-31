# Trusted Advisor AI — Behavioral Intelligence Platform

A real-time behavioral intelligence platform that captures webcam frames, extracts 20+ behavioral signals using MediaPipe and DeepFace, and streams live analytics to a Twitch-inspired dark-mode dashboard. Built for proctoring, meeting analysis, and engagement monitoring.

---

## 🏗️ System Architecture

```
Advisor-ai/
├── python-core/          <- Vision & analytics engine (MediaPipe, DeepFace)
│   ├── main.py           <- Headless pipeline: camera -> signals -> HTTP POST
│   ├── session_recorder.py  <- Auto-records video + signals when camera starts
│   ├── training_exporter.py <- CLI tool: convert recordings to ML datasets
│   └── vision/           <- Pose, face, and attention analyzers
├── backend/              <- Express server (legacy, replaced by SDK)
│   ├── server.js
│   └── public/           <- Dashboard assets (HTML, CSS, JS)
├── sdk/                  <- * Node.js SDK -- the entire project as a package
│   ├── src/
│   │   ├── index.js      <- Entry point
│   │   ├── server.js     <- Full HTTP server (replaces backend/server.js)
│   │   ├── dashboard.js  <- Embedded dashboard (replaces backend/public/)
│   │   ├── session.js    <- Session orchestrator
│   │   ├── api.js        <- HTTP client (zero deps)
│   │   ├── signals.js    <- Reactive signal store
│   │   ├── attention.js  <- Attention timeline & focus streaks
│   │   ├── away.js       <- Away detection & interval logging
│   │   ├── gestures.js   <- Session gesture counter
│   │   ├── interpreter.js<- Behavioral interpretation engine
│   │   └── events.js     <- EventEmitter
│   ├── examples/
│   └── test/
├── recordings/           <- Auto-generated session recordings
│   └── session_YYYY-MM-DD_HH-MM-SS_mmm/
│       ├── video.mp4     <- Raw webcam recording (no overlays)
│       ├── signals.jsonl <- Timestamped behavioral signals (JSON Lines)
│       └── metadata.json <- Session info (duration, frames, timestamps)
├── tests/
│   └── test_recording.py <- 30 tests for recording + export pipeline
├── camera_pipeline.py    <- Standalone pipeline with DeepFace + recording
└── README.md
```

### Data Flow

```mermaid
graph LR
    subgraph Python["Python Vision Engine"]
        CAM[Webcam] --> MP[MediaPipe + DeepFace]
        MP --> SIG[Signal Extraction]
        SIG --> RPT[Cumulative Report]
        CAM --> REC[Session Recorder]
    end

    subgraph SDK["Node.js SDK"]
        SRV["createServer()"] --> ANALYZE[POST /analyze]
        SRV --> DATA[GET /data]
        SRV --> VF[POST /video_frame]
        SRV --> VS[GET /video_feed]
        SRV --> RECS[GET /recordings]
        SRV --> DASH[Dashboard UI]
    end

    subgraph Client["SDK Client / Browser"]
        SESS["createSession()"] --> ATT[Attention Tracker]
        SESS --> AWAY[Away Tracker]
        SESS --> GEST[Gesture Counter]
        SESS --> INTERP[Interpreter]
    end

    subgraph Training["Training Pipeline"]
        REC --> VID[video.mp4]
        REC --> JSONL[signals.jsonl]
        VID --> EXP[Training Exporter]
        JSONL --> EXP
        EXP --> CSV[CSV Dataset]
        EXP --> FRAMES[Frame + Label Pairs]
        EXP --> SEQ[Sequence Windows]
    end

    RPT -->|HTTP POST| ANALYZE
    RPT -->|JPEG frames| VF
    DATA -->|poll 400ms| SESS
    VS -->|MJPEG stream| DASH
```

---

## 🚀 Setup & Execution

### Prerequisites
- **Python 3.9+** with OpenCV, MediaPipe, DeepFace, NumPy
- **Node.js 18+**
- Webcam

### 1. Install Dependencies

```bash
# Python
cd python-core
pip install -r requirements.txt

# Node.js backend (legacy)
cd backend
npm install

# SDK has zero dependencies — nothing to install
```

### 2. Run the System

**Option A — Using the SDK (recommended):**
```bash
# Terminal 1: Start the SDK server
node sdk/examples/full-server.js

# Terminal 2: Start the Python pipeline
cd python-core
python main.py
```

**Option B — Using the legacy backend:**
```bash
# Terminal 1: Start Express backend
cd backend
node server.js

# Terminal 2: Start the Python pipeline
cd python-core
python main.py
```

### 3. View the Dashboard
Open **http://localhost:3000** in your browser.

---

## 📦 Node.js SDK

The SDK packages the **entire project** into a single importable module with zero external dependencies.

### Full Server (replaces `backend/server.js`)

```js
const { createServer } = require("./sdk");
createServer({ port: 3000 }).start();
// Dashboard: http://localhost:3000
// Python posts to: http://localhost:3000/analyze
```

### Client Session (for custom integrations)

```js
const { createSession } = require("./sdk");

const session = createSession({
  backendUrl: "http://localhost:3000",
  mode: "PROCTORING",
  pollInterval: 400,
});

session.on("update",       (data) => console.log(data));
session.on("focus_change", ({ from, to }) => console.log(`${from} → ${to}`));
session.on("away_end",     (log) => console.log(`Away ${log.durationSec}s`));
session.on("gesture",      (g) => console.log(`Gesture #${g.total}`));
session.on("alert",        (flags) => console.log(flags));

session.start();

// Get full summary at any time
const summary = session.getSummary();
```

### SDK Modules

| Module | Purpose |
|--------|---------|
| `createServer()` | Full HTTP backend + embedded dashboard |
| `createSession()` | Client orchestrator with real-time analytics |
| `ApiClient` | Native HTTP client for backend APIs |
| `AttentionTracker` | Rolling timeline, focus streaks, averages |
| `AwayTracker` | Timestamped away intervals (≥3s threshold) |
| `GestureCounter` | Cumulative session gesture count |
| `Interpreter` | PROCTORING / MEETING behavioral analysis |
| `SignalStore` | Reactive signal state with face transition events |

---

## 🎥 Session Recording & Training Data

When the camera pipeline starts, sessions are **automatically recorded** to the `recordings/` directory. No configuration needed.

### What Gets Recorded

| File | Contents |
|------|----------|
| `video.mp4` | Raw webcam frames (no overlays) -- clean data for model training |
| `signals.jsonl` | One JSON object per frame with all 20+ behavioral signals |
| `metadata.json` | Session start/end time, duration, frame count, FPS |

### Training Data Export

Convert recorded sessions into ML-ready datasets:

```bash
# List all recorded sessions
python python-core/training_exporter.py --list

# Export as labeled CSV (one row per frame, 29 signal columns)
python python-core/training_exporter.py --session recordings/session_2026-03-31_13-45-00_123 --format csv

# Extract video frames + per-frame label JSON files
python python-core/training_exporter.py --session <path> --format frames --sample-rate 5

# Export sliding-window sequences for temporal/RNN models
python python-core/training_exporter.py --session <path> --format sequences --window 30 --stride 10
```

### Export Formats

| Format | Use Case | Output |
|--------|----------|--------|
| `csv` | Tabular ML (sklearn, XGBoost) | Single CSV with 29 signal columns per frame |
| `frames` | Vision model fine-tuning | JPEG images + per-frame label JSON files |
| `sequences` | Temporal models (LSTM, Transformer) | Sliding-window JSON with aggregated labels |

### Programmatic Usage

```python
from session_recorder import SessionRecorder
from training_exporter import TrainingDataExporter

# Recording happens automatically, but you can also use it standalone:
recorder = SessionRecorder(base_dir="my_recordings", fps=30)
recorder.start(frame_width=1280, frame_height=720)
recorder.write_frame(frame)      # raw BGR numpy array
recorder.write_signals(sig)      # behavioral signal dict
recorder.stop()

# Export for training:
exporter = TrainingDataExporter()
exporter.load_session("recordings/session_2026-03-31_13-45-00_123")
exporter.export_labeled_csv("training_data.csv")
exporter.export_frame_dataset("frames/", sample_rate=5)
exporter.export_sequence_windows("sequences.json", window_size=30, stride=10)
```

---

## ✨ Key Features

### Vision Engine (Python)
- **20+ Behavioral Signals**: Gaze direction, eye contact score, blink rate, head pose, brow tension, lip state, smile detection (genuine/social/subtle), arm position, shoulder alignment, neck orientation, sitting posture
- **Emotion Detection**: Real-time facial emotion classification via DeepFace
- **Gesture Tracking**: Session-based cumulative hand gesture counting
- **Event Tracking**: Timestamped behavioral events with precise start/end times and durations
- **Auto-Recording**: Sessions automatically recorded (video + signals) when camera starts
- **Headless Pipeline**: Runs without GUI, streams JPEG frames to Node.js backend

### Training Pipeline
- **Automatic Capture**: Raw video + behavioral signal logs saved to disk every session
- **CSV Export**: Flat tabular datasets with 29 behavioral signal columns
- **Frame Datasets**: Extracted video frames paired with per-frame behavioral labels
- **Sequence Windows**: Sliding-window temporal datasets for RNN/Transformer training
- **CLI Tool**: `training_exporter.py` with session discovery and multi-format export

### Dashboard (Twitch-Inspired UI)
- **Live Video Feed**: MJPEG stream embedded directly in the dashboard
- **Attention Timeline**: Real-time Chart.js line graph of attention score
- **Live Timers**: Session uptime and away time chronometers
- **Away Interval Log**: Timestamped records of every absence >= 3 seconds
- **Focus Level Badge**: Dynamic gradient-text focus level indicator
- **8 Metric Cards**: Engagement, tension, eye contact, posture, gaze, gestures, head pose, blink rate
- **Signal Badges**: Real-time facial signal status (smile, brow, lip, nodding, head shake)
- **Emotion Breakdown**: Horizontal bar chart of emotion distribution
- **Behavior Analysis**: Contextual engagement and tension alerts

### SDK (Node.js)
- **Zero Dependencies**: Uses only native Node.js `http` module
- **Embedded Dashboard**: HTML + CSS + JS served from memory (no static files)
- **54 Unit Tests**: Full test suite with zero external test frameworks
- **Event-Driven**: 13 event types for reactive integrations
- **Dual Mode**: PROCTORING (suspicion levels) and MEETING (engagement levels)
- **Recording API**: `GET /recordings` endpoint to list all recorded sessions

---

## 🔌 API Reference

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/analyze` | Receive behavioral report, return interpretation |
| `GET`  | `/data` | Latest report + signals for dashboard polling |
| `POST` | `/video_frame` | Receive JPEG frame from Python pipeline |
| `GET`  | `/video_feed` | MJPEG live stream for browser embed |
| `GET`  | `/recordings` | List all recorded sessions with metadata |
| `GET`  | `/` | Twitch-style dashboard UI |

---

## 🧪 Testing

```bash
# SDK unit tests (54 tests)
cd sdk
npm test

# Recording + export pipeline tests (30 tests)
cd ..
python tests/test_recording.py

# Module demos (no backend needed)
node sdk/examples/standalone-modules.js
```

