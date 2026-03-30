# Trusted Advisor SDK

**The complete Trusted Advisor AI platform as a single Node.js package.**

This SDK contains the **entire project** — the HTTP backend server, the Twitch-style live dashboard, video streaming bridge, behavioral interpretation engine, session analytics, attention tracking, away detection, and gesture counting. **Zero external dependencies.**

## Quick Start — Full Server (replaces `backend/server.js`)

```js
const { createServer } = require("./sdk");

// One line — the entire backend + dashboard + video streaming
const server = createServer({ port: 3000 });
server.start();
```

Then:
1. Open **http://localhost:3000** → full Twitch-style dashboard
2. Start Python: `python python-core/main.py` → live data + video

## Quick Start — Client Session

```js
const { createSession } = require("./sdk");

const session = createSession({
  backendUrl: "http://localhost:3000",
  mode: "PROCTORING",
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
| **`createServer()`** | Full HTTP backend: `/analyze`, `/data`, `/video_frame`, `/video_feed` + embedded dashboard |
| **`createSession()`** | Client-side orchestrator: polls backend, tracks attention/away/gestures |
| **`ApiClient`** | Native HTTP client (zero deps) for backend communication |
| **`SignalStore`** | Reactive signal store with face transition events |
| **`AttentionTracker`** | Rolling attention timeline, focus streaks, averages |
| **`AwayTracker`** | Presence tracking, timestamped away interval logs |
| **`GestureCounter`** | Session-based cumulative gesture counter |
| **`Interpreter`** | Offline PROCTORING/MEETING behavioral analysis |
| **`Dashboard`** | Full HTML + CSS + JS embedded in memory (no static files needed) |

## Architecture

```
sdk/
├── src/
│   ├── index.js           Entry point — exports everything
│   ├── server.js          ★ Full HTTP server (replaces backend/server.js)
│   ├── dashboard.js       ★ Embedded HTML + CSS + JS (replaces backend/public/)
│   ├── session.js         Session orchestrator
│   ├── api.js             HTTP client
│   ├── signals.js         Signal state store
│   ├── attention.js       Attention timeline
│   ├── away.js            Away detection
│   ├── gestures.js        Gesture counter
│   ├── interpreter.js     Behavioral interpretation
│   └── events.js          EventEmitter
├── examples/
│   ├── full-server.js     ★ One-file replacement for entire backend
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
| POST | `/analyze` | Receive behavioral report from Python, return interpretation |
| GET | `/data` | Return latest report + signals for dashboard |
| POST | `/video_frame` | Receive JPEG frames from Python pipeline |
| GET | `/video_feed` | MJPEG live stream for browser |
| GET | `/` | Twitch-style dashboard UI |

## Server Events

```js
const server = createServer({ port: 3000 });

server.on("analyze", ({ mode, sig, interpretation }) => {
  console.log("Report received:", mode, interpretation.flags.length, "flags");
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

## Running

```bash
# Run the full server (replaces backend/server.js entirely)
node examples/full-server.js

# Run tests (54 tests, zero dependencies)
npm test

# Run standalone module demos
node examples/standalone-modules.js
```
