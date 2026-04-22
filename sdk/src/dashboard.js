/**
 * Trusted Advisor SDK — Embedded Dashboard Assets
 * ==================================================
 * Entire Twitch-style dashboard (HTML + CSS + JS) embedded as
 * exportable string functions. Zero file I/O required.
 */

function getDashboardHTML() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Trusted Advisor AI — Real-time behavioral intelligence dashboard" />
  <title>Trusted Advisor AI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="style.css" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <div class="status-indicator" id="status-indicator">Connecting...</div>
  <header>
    <h1>🧠 Trusted Advisor AI</h1>
    <div class="sub-header">Real-time Behavioral Intelligence Dashboard</div>
  </header>
  <main class="dashboard-main">
    <div class="video-column">
      <div class="video-wrapper">
        <div class="video-header">LIVE STREAM</div>
        <img class="live-stream-img" id="live-stream" src="http://localhost:9090/video_feed" alt="Establishing connection..." onerror="if(this.src.includes('9090')){this.src='/video_feed';}else{this.onerror=null;this.alt='Video stream offline.';this.src='';this.style.backgroundColor='#1a1a2e';this.style.minHeight='360px';}" />
      </div>
      <div class="timer-container">
        <div class="timer-box">
          <div class="timer-label">UPTIME</div>
          <div class="timer-value neon-blue" id="session-time">00:00:00</div>
        </div>
        <div class="timer-box">
          <div class="timer-label">AWAY TIME</div>
          <div class="timer-value neon-red" id="away-time">00:00:00</div>
        </div>
      </div>
      <div class="chart-container">
        <canvas id="attentionChart"></canvas>
      </div>

    </div>
    <div class="metrics-column">
      <div class="confidence-box" id="confidence-box">
        <div class="conf-info">
          <div class="conf-label">FOCUS LEVEL</div>
          <div class="conf-level" id="conf-level">LOADING...</div>
        </div>
        <div class="conf-score" id="conf-score">--%</div>
      </div>
      <h3>📊 Key Metrics</h3>
      <div class="metrics-grid">
        <div class="metric-card"><span class="metric-label">⚡ Engagement</span><span class="metric-value" id="metric-eng">—/10</span></div>
        <div class="metric-card"><span class="metric-label">😰 Tension</span><span class="metric-value" id="metric-tension">—/10</span></div>
        <div class="metric-card"><span class="metric-label">👁️ Eye Contact</span><span class="metric-value" id="metric-eye">—%</span></div>
        <div class="metric-card"><span class="metric-label">🧍 Posture</span><span class="metric-value" id="metric-posture">—</span></div>
      </div>
      <div class="metrics-grid">
        <div class="metric-card"><span class="metric-label">👀 Gaze</span><span class="metric-value" id="metric-gaze">—</span></div>
        <div class="metric-card"><span class="metric-label">🤚 Gestures</span><span class="metric-value" id="metric-gestures">—</span></div>
        <div class="metric-card"><span class="metric-label">😊 Head Pose</span><span class="metric-value" id="metric-head">—</span></div>
        <div class="metric-card"><span class="metric-label">👁️ Blinks/min</span><span class="metric-value" id="metric-blinks">—</span></div>
      </div>
      <hr/>
      <h3>😊 Facial & Body Signals</h3>
      <div class="signals-container" id="signals-container"></div>
      <hr/>
      <h3>📝 Away Intervals</h3>
      <div id="away-log-container" style="display:flex; flex-direction:column; gap:8px;">
        <div style="color:var(--text-secondary); font-size:0.9rem;">No away intervals recorded.</div>
      </div>
      <hr/>
      <h3>🎭 Emotion Breakdown</h3>
      <div class="breakdown-container" id="breakdown-container"></div>
      <hr/>

      <h3>🎤 Voice Activity</h3>
      <div class="voice-activity-box" id="voice-activity-box">
        <div class="voice-header">
          <span class="voice-mic-icon" id="voice-mic-icon">🎙️</span>
          <span class="voice-status" id="voice-status">Waiting for audio service...</span>
        </div>
        <div class="voice-energy-track">
          <div class="voice-energy-fill" id="voice-energy-fill" style="width:0%"></div>
        </div>
        <div class="voice-details">
          <span>RMS: <strong id="voice-rms">0.000</strong></span>
          <span>Updated: <strong id="voice-ts">—</strong></span>
        </div>
      </div>

      <h3>🧠 Audio Emotion</h3>
      <div class="confidence-box" id="audio-emotion-box" style="margin-bottom:1.5rem">
        <div class="conf-info">
          <div class="conf-label">VOCAL EMOTION</div>
          <div class="conf-level" id="audio-emotion-label" style="font-size:2rem">WAITING...</div>
        </div>
        <div class="conf-score" id="audio-emotion-conf" style="font-size:3.2rem">—%</div>
      </div>
      <div class="breakdown-container" id="audio-breakdown-container">
        <div style="color:var(--text-secondary);font-size:0.9rem;">Waiting for audio emotion data...</div>
      </div>
      <div class="metrics-grid" style="margin-top:1rem">
        <div class="metric-card"><span class="metric-label">💓 Valence</span><span class="metric-value" id="metric-valence">—</span></div>
        <div class="metric-card"><span class="metric-label">⚡ Arousal</span><span class="metric-value" id="metric-arousal">—</span></div>
        <div class="metric-card"><span class="metric-label">👑 Dominance</span><span class="metric-value" id="metric-dominance">—</span></div>
        <div class="metric-card"><span class="metric-label">🧭 VAD Quadrant</span><span class="metric-value" id="metric-vad-quad" style="font-size:1rem">—</span></div>
      </div>
      <hr/>

      <h3>📝 Live Transcript</h3>
      <div class="transcript-panel" id="transcript-panel">
        <div class="transcript-header">🔴 LIVE TRANSCRIPT</div>
        <div class="transcript-body" id="transcript-body">
          <div class="transcript-placeholder">Waiting for speech...</div>
        </div>
      </div>
      <hr/>

      <h3>🔀 Multimodal Fusion</h3>
      <div class="multimodal-state-box" id="multimodal-state-box">
        <div class="emotion-fusion-row">
          <div class="fusion-source"><span class="fusion-label">👁️ Vision</span><span class="fusion-value" id="fusion-vision">—</span></div>
          <div class="fusion-source"><span class="fusion-label">🎤 Audio</span><span class="fusion-value" id="fusion-audio">—</span></div>
          <div class="fusion-source"><span class="fusion-label">🔀 Fused</span><span class="fusion-value" id="fusion-combined">—</span></div>
        </div>
        <div class="congruence-row">
          <span class="congruence-label">Congruence</span>
          <div class="congruence-track"><div class="congruence-fill" id="congruence-fill" style="width:0%"></div></div>
          <span class="congruence-pct" id="congruence-pct">—%</span>
        </div>
        <div class="multimodal-insight" id="multimodal-insight"></div>
      </div>
      <hr/>

      <h3>🔍 Behavior Analysis</h3>
      <div id="analysis-container"></div>
      <hr/>

      <div class="footer" id="footer-text">Last updated: N/A | Data source: /data</div>
    </div>
  </main>
  <script src="app.js?v=sdk"></script>
</body>
</html>`;
}

function getDashboardCSS() {
  return `:root {
  --bg-color: #0a0a0f;
  --card-color: #13131a;
  --text-color: #efeff1;
  --text-secondary: #adadb8;
  --border-color: #24242e;
  --accent-primary: #9146FF;
  --accent-secondary: #00E6CB;
  --neon-cyan: #00f0ff;
  --neon-pink: #ff2d9b;
  --neon-green: #00ff88;
}
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 8px rgba(255, 0, 60, 0.4), 0 0 20px rgba(255, 0, 60, 0.15); }
  50% { box-shadow: 0 0 16px rgba(255, 0, 60, 0.7), 0 0 40px rgba(255, 0, 60, 0.3); }
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}
@keyframes border-glow {
  0%, 100% { border-color: rgba(0, 240, 255, 0.3); box-shadow: 0 0 15px rgba(0, 240, 255, 0.08), inset 0 0 15px rgba(0, 0, 0, 0.5); }
  50% { border-color: rgba(145, 70, 255, 0.5); box-shadow: 0 0 25px rgba(145, 70, 255, 0.12), inset 0 0 15px rgba(0, 0, 0, 0.5); }
}
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Inter', -apple-system, sans-serif;
  background-color: var(--bg-color);
  background-image: radial-gradient(ellipse at 20% 50%, rgba(145, 70, 255, 0.04) 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(0, 240, 255, 0.03) 0%, transparent 50%);
  color: var(--text-color);
  line-height: 1.5;
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}
h1 { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem; color: var(--text-color); }
h3 { font-size: 1.4rem; font-weight: 600; margin: 2rem 0 1rem; color: var(--text-color); }
.sub-header { color: var(--text-secondary); font-size: 1rem; margin-bottom: 2rem; }
hr { border: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--border-color), transparent); margin: 2rem 0; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 1.5rem; }
.metric-card {
  background: linear-gradient(145deg, var(--card-color) 0%, #0f0f18 100%);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.3s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(145, 70, 255, 0.1); border-color: rgba(145, 70, 255, 0.3); }
.metric-label { color: var(--text-secondary); font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem; }
.metric-value { color: var(--text-color); font-weight: 700; font-size: 1.8rem; }
.signals-container { display: flex; flex-wrap: wrap; gap: 10px; }
.signal-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 0.95rem; background-color: var(--card-color); color: var(--text-color); border: 1px solid var(--border-color); transition: all 0.3s ease; }
.signal-badge.on { border-color: var(--accent-primary); background: linear-gradient(135deg, rgba(145, 70, 255, 0.2), rgba(0, 230, 203, 0.1)); box-shadow: 0 0 12px rgba(145, 70, 255, 0.15); }
.signal-badge.off { color: var(--text-secondary); background-color: var(--card-color); }
.breakdown-container { display: flex; flex-direction: column; gap: 12px; }
.score-bar-row { display: flex; align-items: center; }
.score-label { width: 120px; font-weight: 600; color: var(--text-secondary); font-size: 0.9rem; text-transform: capitalize; }
.score-track { flex: 1; height: 12px; background-color: var(--border-color); border-radius: 6px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); transition: width 0.3s; }
.score-pct { width: 60px; text-align: right; color: var(--text-secondary); font-size: 0.85rem; font-weight: 600; }
.analysis-box { background: linear-gradient(145deg, var(--card-color), #0f0f18); border: 1px solid var(--border-color); border-left: 4px solid var(--accent-primary); border-radius: 12px; padding: 1.2rem; font-weight: 600; font-size: 1.05rem; margin-bottom: 1rem; color: var(--text-color); }
.footer { color: var(--text-secondary); font-size: 0.85rem; margin-top: 2rem; }
.status-indicator { position: absolute; top: 2rem; right: 2rem; padding: 6px 14px; border-radius: 20px; background-color: var(--card-color); color: var(--text-secondary); font-weight: 600; font-size: 0.85rem; border: 1px solid var(--border-color); transition: all 0.3s ease; }
.dashboard-main { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; align-items: start; }
.video-wrapper {
  position: relative;
  background-color: #0d0d14;
  border: 2px solid rgba(0, 240, 255, 0.2);
  border-radius: 16px;
  overflow: hidden;
  animation: border-glow 4s ease-in-out infinite;
}
.video-header {
  background: linear-gradient(90deg, #1a1a2e 0%, #0d0d14 100%);
  color: var(--text-color);
  padding: 0.6rem 1.2rem;
  font-weight: 700;
  font-size: 0.85rem;
  letter-spacing: 1.5px;
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid rgba(0, 240, 255, 0.1);
}
.video-header::before {
  content: '';
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #ff3040;
  animation: pulse-dot 1.5s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(255, 48, 64, 0.6);
}
.live-stream-img {
  width: 100%;
  aspect-ratio: 4/3;
  display: block;
  object-fit: contain;
  background: radial-gradient(ellipse at center, #141428 0%, #0a0a14 100%);
  will-change: contents;
  image-rendering: auto;
}
.confidence-box {
  background: linear-gradient(145deg, var(--card-color) 0%, #0f0f1a 100%);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 2.5rem;
  color: var(--text-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  position: relative;
  overflow: hidden;
}
.confidence-box::before {
  content: '';
  position: absolute;
  top: 0; left: -100%; width: 200%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(145, 70, 255, 0.03), transparent);
  animation: shimmer 6s ease-in-out infinite;
}
.confidence-box:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(145, 70, 255, 0.15); }
.conf-info { display: flex; flex-direction: column; position: relative; z-index: 1; }
.conf-label { font-size: 0.95rem; font-weight: 600; color: var(--text-secondary); letter-spacing: 1px; margin-bottom: 0.5rem; text-transform: uppercase; }
.conf-level { font-size: 2.8rem; font-weight: 800; line-height: 1; text-transform: uppercase; background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.conf-score { font-size: 4.8rem; font-weight: 800; line-height: 1; background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; position: relative; z-index: 1; }
.timer-container { display: flex; gap: 1.5rem; margin: 1.5rem 0; }
.timer-box { flex: 1; background: linear-gradient(145deg, var(--card-color), #0f0f18); border: 1px solid var(--border-color); border-top: 3px solid var(--accent-primary); border-radius: 12px; padding: 1rem 1.5rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
.timer-label { font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); letter-spacing: 2px; margin-bottom: 0.5rem; }
.timer-value { font-size: 2.2rem; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1; }
.neon-blue { color: var(--accent-secondary); text-shadow: 0 0 10px rgba(0, 230, 203, 0.35), 0 0 30px rgba(0, 230, 203, 0.1); }
.neon-red { color: #FF4A4A; text-shadow: 0 0 10px rgba(255, 74, 74, 0.4), 0 0 30px rgba(255, 74, 74, 0.15); }
.chart-container { background: linear-gradient(145deg, var(--card-color), #0f0f18); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; height: 220px; width: 100%; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
@media (max-width: 1100px) { .dashboard-main { grid-template-columns: 1fr; } }
@media (max-width: 900px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .metrics-grid { grid-template-columns: 1fr; } .conf-level { font-size: 1.8rem; } .conf-score { font-size: 3rem; } }
.voice-activity-box { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.voice-header { display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem; }
.voice-mic-icon { font-size: 1.4rem; transition: all 0.3s ease; }
.voice-mic-icon.speaking { animation: pulse-mic 1s ease-in-out infinite; filter: drop-shadow(0 0 6px rgba(0, 230, 203, 0.6)); }
@keyframes pulse-mic { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.15); } }
.voice-status { font-weight: 600; font-size: 0.95rem; color: var(--text-secondary); transition: color 0.3s ease; }
.voice-status.active { color: var(--accent-secondary); }
.voice-energy-track { height: 8px; background-color: var(--border-color); border-radius: 4px; overflow: hidden; margin-bottom: 0.8rem; }
.voice-energy-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--accent-secondary) 0%, #00ff88 100%); transition: width 0.3s ease; box-shadow: 0 0 8px rgba(0, 230, 203, 0.3); }
.voice-details { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-secondary); }
.voice-details strong { color: var(--text-color); }
.transcript-panel { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.transcript-header { padding: 0.6rem 1.2rem; font-weight: 700; font-size: 0.85rem; letter-spacing: 1px; color: var(--text-color); background-color: rgba(145, 70, 255, 0.08); border-bottom: 1px solid var(--border-color); }
.transcript-body { max-height: 200px; overflow-y: auto; padding: 0.8rem 1.2rem; scroll-behavior: smooth; }
.transcript-entry { display: flex; gap: 10px; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.9rem; animation: fadeIn 0.3s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
.transcript-ts { color: var(--accent-primary); font-weight: 600; font-size: 0.8rem; min-width: 65px; flex-shrink: 0; }
.transcript-text { color: var(--text-color); line-height: 1.4; }
.transcript-placeholder { color: var(--text-secondary); font-size: 0.9rem; font-style: italic; }
.multimodal-state-box { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.emotion-fusion-row { display: flex; justify-content: space-around; margin-bottom: 1rem; gap: 1rem; }
.fusion-source { text-align: center; flex: 1; padding: 0.6rem; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid var(--border-color); }
.fusion-label { display: block; font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); letter-spacing: 0.5px; margin-bottom: 0.3rem; text-transform: uppercase; }
.fusion-value { display: block; font-size: 1.1rem; font-weight: 700; color: var(--text-color); text-transform: capitalize; }
.congruence-row { display: flex; align-items: center; gap: 12px; margin-bottom: 0.8rem; }
.congruence-label { font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); min-width: 130px; }
.congruence-track { flex: 1; height: 10px; background-color: var(--border-color); border-radius: 5px; overflow: hidden; }
.congruence-fill { height: 100%; border-radius: 5px; background: linear-gradient(90deg, #FF4A4A 0%, #FFB84A 30%, var(--accent-secondary) 70%, #00ff88 100%); transition: width 0.5s ease; }
.congruence-pct { font-size: 0.9rem; font-weight: 700; color: var(--text-color); min-width: 45px; text-align: right; }
.multimodal-insight { background: rgba(145, 70, 255, 0.06); border: 1px solid rgba(145, 70, 255, 0.2); border-left: 4px solid var(--accent-primary); border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.9rem; color: var(--text-color); line-height: 1.5; }
.multimodal-insight:empty { display: none; }
.multimodal-insight .insight-label { font-weight: 700; color: var(--accent-primary); font-size: 0.8rem; letter-spacing: 0.5px; display: block; margin-bottom: 0.3rem; }`;
}

function getDashboardJS() {
  return `const FETCH_INTERVAL = 400;
const DATA_URL = "/data";
const statusIndicator = document.getElementById("status-indicator");
const metricEng = document.getElementById("metric-eng");
const metricTension = document.getElementById("metric-tension");
const metricEye = document.getElementById("metric-eye");
const metricPosture = document.getElementById("metric-posture");
const metricGaze = document.getElementById("metric-gaze");
const metricGestures = document.getElementById("metric-gestures");
const metricHead = document.getElementById("metric-head");
const metricBlinks = document.getElementById("metric-blinks");
const signalsContainer = document.getElementById("signals-container");
const breakdownContainer = document.getElementById("breakdown-container");
const analysisContainer = document.getElementById("analysis-container");
const footerText = document.getElementById("footer-text");
const voiceStatus = document.getElementById("voice-status");
const voiceMicIcon = document.getElementById("voice-mic-icon");
const voiceEnergyFill = document.getElementById("voice-energy-fill");
const voiceRms = document.getElementById("voice-rms");
const voiceTs = document.getElementById("voice-ts");
const audioEmotionLabel = document.getElementById("audio-emotion-label");
const audioEmotionConf = document.getElementById("audio-emotion-conf");
const audioBreakdownContainer = document.getElementById("audio-breakdown-container");
const metricValence = document.getElementById("metric-valence");
const metricArousal = document.getElementById("metric-arousal");
const metricDominance = document.getElementById("metric-dominance");
const metricVadQuad = document.getElementById("metric-vad-quad");
const transcriptBody = document.getElementById("transcript-body");
const fusionVision = document.getElementById("fusion-vision");
const fusionAudio = document.getElementById("fusion-audio");
const fusionCombined = document.getElementById("fusion-combined");
const congruenceFill = document.getElementById("congruence-fill");
const congruencePct = document.getElementById("congruence-pct");
const multimodalInsight = document.getElementById("multimodal-insight");

let sessionSeconds = 0, awaySeconds = 0, isFaceDetected = false, attentionChart = null;
let awayLogs = [], currentAwayStart = null, lastAudioUpdateTime = 0;

function initChart() {
  const ctx = document.getElementById('attentionChart').getContext('2d');
  attentionChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array(50).fill(''),
      datasets: [{ label: 'Attention', data: Array(50).fill(100), borderColor: '#00E6CB', backgroundColor: 'rgba(0,230,203,0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: { y: { min: 0, max: 100, grid: { color: '#2f2f35' } }, x: { grid: { display: false } } },
      plugins: { legend: { display: false } }
    }
  });
}

function formatTime(s) {
  const hh = Math.floor(s/3600).toString().padStart(2,'0');
  const mm = Math.floor((s%3600)/60).toString().padStart(2,'0');
  const ss = (s%60).toString().padStart(2,'0');
  return hh+':'+mm+':'+ss;
}

setInterval(() => {
  sessionSeconds++;
  if (!isFaceDetected) {
    awaySeconds++;
    if (!currentAwayStart) currentAwayStart = new Date();
  } else {
    if (currentAwayStart) {
      const awayEnd = new Date();
      const dur = Math.round((awayEnd - currentAwayStart)/1000);
      if (dur >= 3) awayLogs.unshift({ start: currentAwayStart, end: awayEnd, duration: dur });
      currentAwayStart = null;
    }
  }
  renderAwayLogs();
  document.getElementById('session-time').textContent = formatTime(sessionSeconds);
  document.getElementById('away-time').textContent = formatTime(awaySeconds);
}, 1000);

async function fetchData() {
  try {
    const res = await fetch(DATA_URL);
    const json = await res.json();
    if (json.status === "waiting" || (!json.report && !json.audioEmotion)) { statusIndicator.textContent = "Waiting for data..."; isFaceDetected = false; return; }
    const stale = json.lastUpdated && (Date.now() - json.lastUpdated > 2500);
    if (stale && !json.audioEmotion) { statusIndicator.textContent = "Camera Feed Frozen"; statusIndicator.style.color = "#FF4A4A"; statusIndicator.style.borderColor = "#FF4A4A"; isFaceDetected = false; }
    else { statusIndicator.textContent = "LIVE"; statusIndicator.style.color = "var(--text-color)"; statusIndicator.style.borderColor = "var(--text-color)"; }
    if (json.report) updateVisionDashboard(json.report);
    if (json.audioEmotion) { lastAudioUpdateTime = Date.now(); updateAudioDashboard(json.audioEmotion); }
    if (json.report && json.audioEmotion) updateMultimodalFusion(json.report, json.audioEmotion);
  } catch (err) { statusIndicator.textContent = "Connection lost"; statusIndicator.style.color = "var(--text-secondary)"; isFaceDetected = false; }
}

function updateVisionDashboard(payload) {
  const report = payload.data || {};
  const sig = payload.sig || {};
  isFaceDetected = (sig.face_detected === true || String(sig.face_detected).toLowerCase() === "true");
  if (report.summary) {
    document.getElementById("conf-level").textContent = report.summary.focus_level || "UNKNOWN";
    document.getElementById("conf-score").textContent = Math.round((report.summary.attention_score||0)*100)+'%';
  } else { document.getElementById("conf-level").textContent = "WAITING..."; document.getElementById("conf-score").textContent = "--%"; }
  if (attentionChart) { const cs = report.summary ? Math.round((report.summary.attention_score||0)*100) : 0; attentionChart.data.datasets[0].data.shift(); attentionChart.data.datasets[0].data.push(cs); attentionChart.update(); }
  const eng = sig.engagement_score || 5; const tension = sig.micro_tension_score || 0; const eye = sig.eye_contact_score || 0;
  metricEng.textContent = eng+'/10'; metricTension.textContent = tension+'/10'; metricEye.textContent = Math.round(eye*100)+'%'; metricPosture.textContent = sig.posture || "Closed";
  metricGaze.textContent = sig.gaze || "—"; metricGestures.textContent = sig.gestures || "0"; metricHead.textContent = sig.head_pose || "—"; metricBlinks.textContent = Math.round(sig.blinks_per_minute || 0).toString();
  let signalsHtml = "";
  signalsHtml += createBadge("Smile: "+(sig.smile_label||'-'), sig.smile_genuine);
  signalsHtml += createBadge("Brow: "+(sig.brow_label||'-'), sig.brow_label === 'RAISED');
  signalsHtml += createBadge("Lip: "+(sig.lip_label||'-'), sig.lip_label === 'RELAXED');
  signalsHtml += createBadge("Nodding", sig.nodding);
  signalsHtml += createBadge("Head Shake", sig.head_shake);
  signalsContainer.innerHTML = signalsHtml;
  const emoAll = sig.emotion_all || {};
  if (Object.keys(emoAll).length > 0 && sig.emotion !== "loading" && sig.emotion !== "analyzing") {
    const sorted = Object.entries(emoAll).sort((a,b) => b[1]-a[1]);
    let barsHtml = "";
    for (const [name, val] of sorted) { barsHtml += '<div class="score-bar-row"><span class="score-label">'+name+'</span><div class="score-track"><div class="score-fill" style="width:'+Math.max(2,val)+'%"></div></div><span class="score-pct">'+val.toFixed(1)+'%</span></div>'; }
    breakdownContainer.innerHTML = barsHtml;
  } else { breakdownContainer.innerHTML = '<div style="color:var(--text-secondary);font-size:0.9rem;">Loading breakdown...</div>'; }
  let analysisHtml = "";
  if (eng >= 7) analysisHtml += '<div class="analysis-box">✅ Excellent Engagement ('+eng+'/10)</div>';
  else if (eng >= 4) analysisHtml += '<div class="analysis-box">⚠️ Moderate Engagement ('+eng+'/10)</div>';
  else analysisHtml += '<div class="analysis-box">🚨 Low Engagement ('+eng+'/10)</div>';
  if (tension >= 6) analysisHtml += '<div class="analysis-box">😰 High Tension ('+tension+'/10)</div>';
  // Multimodal behavioral insights from report
  const mm = report.multimodal_emotion;
  if (mm && mm.behavioral_insights) {
    for (const insight of mm.behavioral_insights) {
      const icon = insight.includes("congruence") || insight.includes("genuinely") ? "✅" : insight.includes("mismatch") || insight.includes("stress") ? "🚨" : "⚠️";
      analysisHtml += '<div class="analysis-box">' + icon + ' ' + insight + '</div>';
    }
  }
  analysisContainer.innerHTML = analysisHtml;
  footerText.textContent = 'Last updated: '+new Date().toLocaleTimeString()+' | Data source: /data';
}

function updateAudioDashboard(audioPayload) {
  const ae = audioPayload.audio_emotion; if (!ae) return;
  const isSilence = (ae.status === "silence"); const rms = ae.rms || 0;
  if (isSilence) { voiceStatus.textContent = "Listening..."; voiceStatus.classList.remove("active"); voiceMicIcon.classList.remove("speaking"); }
  else { voiceStatus.textContent = "Speech Detected"; voiceStatus.classList.add("active"); voiceMicIcon.classList.add("speaking"); }
  voiceEnergyFill.style.width = Math.min(100, Math.round(rms * 1000)) + '%';
  voiceRms.textContent = rms.toFixed(4); voiceTs.textContent = ae.timestamp || "—";
  if (!isSilence && ae.label) { audioEmotionLabel.textContent = ae.label.toUpperCase(); audioEmotionConf.textContent = Math.round((ae.confidence||0)*100)+'%'; }
  else { audioEmotionLabel.textContent = isSilence ? "LISTENING..." : "WAITING..."; audioEmotionConf.textContent = "—%"; }
  if (!isSilence && ae.all_scores && Object.keys(ae.all_scores).length > 0) {
    const sorted = Object.entries(ae.all_scores).sort((a,b) => b[1]-a[1]);
    let barsHtml = "";
    for (const [emo, score] of sorted) { const pct = Math.round(score*100); barsHtml += '<div class="score-bar-row"><span class="score-label">'+emo+'</span><div class="score-track"><div class="score-fill" style="width:'+Math.max(2,pct)+'%;background:linear-gradient(90deg,#00E6CB,#00ff88)"></div></div><span class="score-pct">'+pct+'%</span></div>'; }
    audioBreakdownContainer.innerHTML = barsHtml;
  }
  if (!isSilence) {
    metricValence.textContent = (ae.valence >= 0 ? '+' : '') + (ae.valence||0).toFixed(3);
    metricArousal.textContent = (ae.arousal >= 0 ? '+' : '') + (ae.arousal||0).toFixed(3);
    metricDominance.textContent = (ae.dominance >= 0 ? '+' : '') + (ae.dominance||0).toFixed(3);
    metricVadQuad.textContent = ae.vad_quadrant || "—";
  }
  const transcript = audioPayload.transcript;
  if (transcript && transcript.length > 0) {
    let html = "";
    for (const entry of transcript) { html += '<div class="transcript-entry"><span class="transcript-ts">'+entry.ts+'</span><span class="transcript-text">'+entry.text+'</span></div>'; }
    transcriptBody.innerHTML = html; transcriptBody.scrollTop = transcriptBody.scrollHeight;
  }
}

function updateMultimodalFusion(visionPayload, audioPayload) {
  const report = visionPayload.data || {};
  const ae = audioPayload.audio_emotion || {};
  const mm = report.multimodal_emotion;
  if (mm && mm.fused_emotion) {
    fusionVision.textContent = capitalize(mm.vision_emotion || "—");
    fusionAudio.textContent = capitalize(mm.audio_emotion || "—");
    fusionCombined.textContent = capitalize(mm.fused_emotion);
    const cong = Math.round((mm.congruence||0)*100);
    congruenceFill.style.width = cong+'%'; congruencePct.textContent = cong+'%';
    const insights = mm.behavioral_insights || [];
    if (insights.length > 0) {
      let html = '<span class="insight-label">🧠 AI BEHAVIORAL INSIGHTS</span>';
      for (const i of insights) { const icon = i.includes("congruence")||i.includes("genuinely") ? "✅" : i.includes("mismatch")||i.includes("stress") ? "🚨" : "⚠️"; html += '<div style="margin:4px 0;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05)">' + icon + ' ' + i + '</div>'; }
      multimodalInsight.innerHTML = html;
    }
  } else {
    const audioLabel = (ae.status !== "silence" && ae.label) ? ae.label : "—";
    fusionVision.textContent = "—"; fusionAudio.textContent = capitalize(audioLabel); fusionCombined.textContent = capitalize(audioLabel);
  }
}

function capitalize(s) { if (!s||s==="—") return "—"; return s.charAt(0).toUpperCase()+s.slice(1).toLowerCase(); }

function createBadge(label, active) {
  const cls = active ? "on" : "off";
  const icon = active ? "✅" : "❌";
  return '<span class="signal-badge '+cls+'">'+icon+' '+label+'</span>';
}

function renderAwayLogs() {
  const container = document.getElementById('away-log-container');
  if (!container) return;
  if (awayLogs.length === 0 && !currentAwayStart) { container.innerHTML = '<div style="color:var(--text-secondary);font-size:0.9rem;">No away intervals recorded.</div>'; return; }
  let html = '';
  if (currentAwayStart) html += '<div class="analysis-box" style="border-left-color:#FF4A4A;padding:0.8rem 1.2rem;margin-bottom:0.5rem;font-size:0.95rem;">🔴 <span style="font-weight:700">Currently Away</span> (since '+currentAwayStart.toLocaleTimeString()+')</div>';
  for (const log of awayLogs.slice(0,10)) { html += '<div class="analysis-box" style="padding:0.8rem 1.2rem;margin-bottom:0.5rem;font-size:0.95rem;">🛑 Away: <span style="color:var(--text-secondary)">'+log.start.toLocaleTimeString()+' - '+log.end.toLocaleTimeString()+'</span><span style="float:right;font-weight:700;">'+log.duration+'s</span></div>'; }
  container.innerHTML = html;
}

setInterval(fetchData, FETCH_INTERVAL);
initChart();
fetchData();
`;
}

module.exports = { getDashboardHTML, getDashboardCSS, getDashboardJS };
