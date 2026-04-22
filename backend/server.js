/**
 * Trusted Advisor AI — Node.js Backend
 * =====================================
 * POST /analyze  — receive + store report, return interpretation
 * GET  /data     — return latest report for dashboard
 * Static files   — serve public/ directory for web dashboard
 */

const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// Serve static frontend files
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 3000;

// ── In-memory store ──────────────────────────────────────────────────

let latestReport = null;
let lastUpdated = null;

// ── Audio Emotion store ──────────────────────────────────────────────

let latestAudioEmotion = null;
let audioLastUpdated = null;

// ── Thresholds ───────────────────────────────────────────────────────

const THRESHOLDS = {
  PROCTORING: {
    attention_low: 0.50,
    off_screen_warn: 0.20,
    head_down_warn: 0.15,
    face_missing_warn: 0.10,
    alert_count_limit: 3,
  },
  MEETING: {
    attention_low: 0.35,
    off_screen_warn: 0.40,
    head_down_warn: 0.30,
    face_missing_warn: 0.25,
    alert_count_limit: 8,
  },
};

// ── POST /analyze ────────────────────────────────────────────────────

app.post("/analyze", (req, res) => {
  const { mode, data } = req.body;

  if (!mode || !data) {
    return res.status(400).json({ error: "Missing 'mode' or 'data' in request body." });
  }

  const upperMode = mode.toUpperCase();
  const thresholds = THRESHOLDS[upperMode];

  if (!thresholds) {
    return res.status(400).json({
      error: `Unknown mode '${mode}'. Supported: PROCTORING, MEETING.`,
    });
  }

  // Store the raw report for GET /data
  latestReport = req.body;
  lastUpdated = new Date().toISOString();

  const interpretation = interpret(data, thresholds, upperMode);
  return res.json({ mode: upperMode, interpretation });
});

// ── POST /audio_emotion ──────────────────────────────────────────────

app.post("/audio_emotion", (req, res) => {
  const body = req.body;

  if (!body || !body.audio_emotion) {
    return res.status(400).json({ error: "Missing 'audio_emotion' in request body." });
  }

  latestAudioEmotion = body;
  audioLastUpdated = new Date().toISOString();

  return res.json({ status: "ok", received: audioLastUpdated });
});

// ── GET /audio_data ──────────────────────────────────────────────────

app.get("/audio_data", (req, res) => {
  if (!latestAudioEmotion) {
    return res.json({ status: "waiting", message: "No audio emotion data yet." });
  }

  return res.json({
    status: "live",
    lastUpdated: audioLastUpdated,
    ...latestAudioEmotion,
  });
});

// ── Video Stream Handlers ─────────────────────────────────────────────

let videoClients = [];

app.post('/video_frame', express.raw({ type: 'image/jpeg', limit: '10mb' }), (req, res) => {
  if (!req.body || !req.body.length) {
    return res.status(400).send('Empty frame');
  }
  const frame = req.body;
  videoClients.forEach(c => {
    try {
      c.write(`--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: ${frame.length}\r\n\r\n`);
      c.write(frame);
      c.write('\r\n');
    } catch (e) {}
  });
  return res.sendStatus(200);
});

app.get('/video_feed', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'multipart/x-mixed-replace; boundary=FRAME',
    'Cache-Control': 'no-cache, private',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache'
  });
  videoClients.push(res);
  req.on('close', () => {
    videoClients = videoClients.filter(c => c !== res);
  });
});

// ── GET /data ────────────────────────────────────────────────────────

app.get("/data", (req, res) => {
  if (!latestReport && !latestAudioEmotion) {
    return res.json({ status: "waiting", message: "No data received yet." });
  }

  return res.json({
    status: "live",
    lastUpdated,
    report: latestReport,
    audioEmotion: latestAudioEmotion || null,
    audioLastUpdated: audioLastUpdated || null,
  });
});

// ── Interpretation logic ─────────────────────────────────────────────

function interpret(report, t, mode) {
  const summary = report.summary || {};
  const metrics = report.metrics || {};
  const alerts = report.alerts || [];
  const body = report.body_analysis || {};

  const flags = [];

  if ((summary.attention_score || 1) < t.attention_low) {
    flags.push({
      severity: mode === "PROCTORING" ? "HIGH" : "MEDIUM",
      message: `Attention score (${summary.attention_score}) is below threshold (${t.attention_low}).`,
    });
  }

  const offScreen = parsePercent(metrics.off_screen_time);
  if (offScreen > t.off_screen_warn) {
    flags.push({
      severity: mode === "PROCTORING" ? "HIGH" : "LOW",
      message: `Off-screen time (${metrics.off_screen_time}) exceeds limit (${(t.off_screen_warn * 100).toFixed(0)}%).`,
    });
  }

  const headDown = parsePercent(metrics.head_down_time);
  if (headDown > t.head_down_warn) {
    flags.push({
      severity: "MEDIUM",
      message: `Head-down time (${metrics.head_down_time}) exceeds limit (${(t.head_down_warn * 100).toFixed(0)}%).`,
    });
  }

  const faceMissing = parsePercent(metrics.face_missing_time);
  if (faceMissing > t.face_missing_warn) {
    flags.push({
      severity: mode === "PROCTORING" ? "HIGH" : "MEDIUM",
      message: `Face missing (${metrics.face_missing_time}) exceeds limit (${(t.face_missing_warn * 100).toFixed(0)}%).`,
    });
  }

  if (alerts.length >= t.alert_count_limit) {
    flags.push({
      severity: "HIGH",
      message: `${alerts.length} behavioral alerts detected (limit: ${t.alert_count_limit}).`,
    });
  }

  if (body.posture === "SLOUCHED") {
    flags.push({ severity: "LOW", message: "Subject appears slouched." });
  }
  if (body.neck === "FORWARD_HEAD") {
    flags.push({ severity: "LOW", message: "Forward head posture detected." });
  }

  let suspicion = null;
  if (mode === "PROCTORING") {
    const highCount = flags.filter((f) => f.severity === "HIGH").length;
    if (highCount >= 3) suspicion = "HIGH";
    else if (highCount >= 1) suspicion = "MODERATE";
    else suspicion = "LOW";
  }

  let engagement = null;
  if (mode === "MEETING") {
    const score = summary.attention_score || 0;
    if (score >= 0.75) engagement = "HIGHLY_ENGAGED";
    else if (score >= 0.50) engagement = "ENGAGED";
    else if (score >= 0.30) engagement = "PARTIALLY_ENGAGED";
    else engagement = "DISENGAGED";
  }

  return {
    flags,
    ...(suspicion && { suspicion_level: suspicion }),
    ...(engagement && { engagement_level: engagement }),
    recommendation: report.recommendation || "REVIEW_REQUIRED",
    original_report: report,
    // Audio emotion data (from bridge, if present)
    ...(report.audio_emotion && { audio_emotion: report.audio_emotion }),
    ...(report.multimodal_emotion && { multimodal_emotion: report.multimodal_emotion }),
    ...(report.live_transcript && { live_transcript: report.live_transcript }),
  };
}

// ── Helpers ──────────────────────────────────────────────────────────

function parsePercent(str) {
  if (!str) return 0;
  return parseInt(str.replace("%", ""), 10) / 100;
}

// ── Start ────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[Trusted Advisor AI] Backend running on http://localhost:${PORT}`);
  console.log(`[Trusted Advisor AI] Dashboard: http://localhost:${PORT}/`);
  console.log(`[Trusted Advisor AI] Waiting for Python pipeline data...`);
});
