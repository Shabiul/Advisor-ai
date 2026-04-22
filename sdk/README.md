# Trusted Advisor SDK

**The complete Trusted Advisor AI platform as a single Node.js package.**

This SDK contains the **entire backend** — HTTP server, Twitch-style live dashboard, video streaming, multimodal behavioral interpretation (vision + audio), LLM-powered insights, session analytics, attention tracking, away detection, and gesture counting. **Zero external dependencies.**

## Quick Start — Full Server

```js
const { createServer } = require("./sdk");

// One line — the entire backend + dashboard + video streaming + audio emotion
const server = createServer({ port: 3000 });
server.start();
```

Then:
1. Open **http://localhost:3000** → full Twitch-style dashboard
2. Start vision: `conda activate face && cd python-core && python main.py`
3. Start audio: `conda activate aud && cd python-core && python -m aud.emo_service`

## Quick Start — Client Session

```js
const { createSession } = require("./sdk");

const session = createSession({
  backendUrl: "http://localhost:3000",
  mode: "PROCTORING",   // or "MEETING"
  pollInterval: 400,
});

session.on("update",       (data) => console.log(data));
session.on("away_end",     (log)  => console.log("Away:", log.durationSec + "s"));
session.on("focus_change", (chg)  => console.log(chg.from, "→", chg.to));
session.on("gesture",      (g)    => console.log("Gesture #" + g.total));
session.on("alert",        (flags)=> console.log("Alerts:", flags));

session.start();
```

## What's Inside

| Module | What It Does |
|--------|-------------|
| **`createServer()`** | Full HTTP backend: 8 routes + embedded dashboard + audio emotion endpoint |
| **`createSession()`** | Client-side orchestrator: polls backend, tracks attention/away/gestures |
| **`Interpreter`** | Multimodal behavioral analysis (vision + audio fusion, inconsistency detection) |
| **`LLMEngine`** | Ollama-powered behavioral insights with rule-based fallback |
| **`ApiClient`** | Native HTTP client (zero deps) for backend communication |
| **`SignalStore`** | Reactive signal store with face transition events |
| **`AttentionTracker`** | Rolling attention timeline, focus streaks, averages |
| **`AwayTracker`** | Presence tracking, timestamped away interval logs |
| **`GestureCounter`** | Session-based cumulative gesture counter |
| **`Dashboard`** | Full HTML + CSS + JS embedded in memory (no static files needed) |

## Architecture

```
sdk/
├── src/
│   ├── index.js           Entry point — exports everything
│   ├── server.js          ★ Full HTTP server (8 routes + dashboard)
│   ├── dashboard.js       ★ Embedded HTML + CSS + JS (Twitch-style UI)
│   ├── interpreter.js     ★ Multimodal behavioral interpreter v3
│   ├── llm_engine.js      ★ Ollama LLM integration (rule-based fallback)
│   ├── session.js         Session orchestrator
│   ├── api.js             HTTP client
│   ├── signals.js         Signal state store
│   ├── attention.js       Attention timeline
│   ├── away.js            Away detection
│   ├── gestures.js        Gesture counter
│   └── events.js          EventEmitter
├── examples/
│   ├── full-server.js     ★ One-file full server launch
│   ├── basic-session.js   Live session monitoring
│   └── standalone-modules.js  Individual module demos
├── test/
│   └── run.js             54 unit tests
├── package.json
└── README.md
```

## API Routes (served by `createServer`)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/analyze` | Receive behavioral report from vision pipeline, return interpretation |
| POST | `/audio_emotion` | Receive audio emotion payload from `emo_service.py` |
| GET | `/data` | Return latest report + audio emotion for dashboard |
| GET | `/audio_data` | Return latest audio emotion data only |
| POST | `/video_frame` | Receive JPEG frames from Python pipeline |
| GET | `/video_feed` | MJPEG live stream for browser (relay fallback) |
| GET | `/recordings` | List all recorded sessions with metadata |
| GET | `/` | Twitch-style dashboard UI |

## Server Events

```js
const server = createServer({ port: 3000 });

server.on("analyze", ({ mode, sig, interpretation }) => {
  const modality = interpretation.modality;  // "vision_only" or "multimodal"
  console.log("Report:", mode, modality, interpretation.flags.length, "flags");
});

server.on("audio_emotion", (data) => {
  const ae = data.audio_emotion;
  console.log("Voice:", ae.label, Math.round(ae.confidence * 100) + "%");
});

server.on("started", ({ port }) => console.log("Running on port", port));
server.on("stopped", () => console.log("Server stopped"));
server.on("error", (err) => console.error(err));
```

## Session Events

| Event | Payload | Description |
|-------|---------|-------------|
| `update` | `{ sig, report, timestamp }` | New signal data |
| `focus_change` | `{ from, to, timestamp }` | Attention level changed |
| `away_start` | `{ startTime }` | User left frame |
| `away_end` | `{ start, end, durationSec }` | User returned |
| `gesture` | `{ total, timestamp }` | New gesture detected |
| `alert` | `[{ severity, message }]` | Behavioral flags |
| `emotion` | `{ emotion, confidence }` | Emotion detected |
| `tick` | `{ sessionSec, awaySec, isAway }` | 1-second timer |

## Multimodal Interpreter

```js
const { Interpreter } = require("./sdk");

const interp = new Interpreter("PROCTORING");

// Vision-only
const result = interp.interpret(report);

// Multimodal (vision + audio)
const multiResult = interp.interpretMultimodal(report, audioData);
// → { flags, fused_risk_score, emotional_state, inconsistencies, multimodal }

// Cross-modal inconsistency detection
const conflicts = interp.detectInconsistencies(report, audioData);
// → [{ label: "MASKED_ANGER", severity: "HIGH", message: "..." }]

// Combined emotional state
const state = interp.getEmotionalState(report, audioData);
// → { vision_emotion, audio_emotion, fused_emotion, congruence, insights }
```

## Running

```bash
# Run the full server
node examples/full-server.js

# Run tests (54 tests, zero dependencies)
npm test

# Run standalone module demos
node examples/standalone-modules.js
```
