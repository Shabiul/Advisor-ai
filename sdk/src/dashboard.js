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
        <div class="video-header">🔴 LIVE STREAM</div>
        <img class="live-stream-img" id="live-stream" src="/video_feed" alt="Establishing connection..." onerror="this.onerror=null; this.alt='Video stream offline.'; this.src=''; this.style.backgroundColor='#1a1a2e'; this.style.minHeight='360px';" />
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
      <h3>🔍 Behavior Analysis</h3>
      <div id="analysis-container"></div>
      <div class="footer" id="footer-text">Last updated: N/A | Data source: /data</div>
    </div>
  </main>
  <script src="app.js?v=sdk"></script>
</body>
</html>`;
}

function getDashboardCSS() {
  return `:root {
  --bg-color: #0e0e10;
  --card-color: #18181b;
  --text-color: #efeff1;
  --text-secondary: #adadb8;
  --border-color: #2f2f35;
  --accent-primary: #9146FF;
  --accent-secondary: #00E6CB;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Inter', -apple-system, sans-serif;
  background-color: var(--bg-color);
  color: var(--text-color);
  line-height: 1.5;
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}
h1 { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem; color: var(--text-color); }
h3 { font-size: 1.4rem; font-weight: 600; margin: 2rem 0 1rem; color: var(--text-color); }
.sub-header { color: var(--text-secondary); font-size: 1rem; margin-bottom: 2rem; }
hr { border: 0; height: 1px; background-color: var(--border-color); margin: 2rem 0; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 1.5rem; }
.metric-card {
  background-color: var(--card-color);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1); }
.metric-label { color: var(--text-secondary); font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem; }
.metric-value { color: var(--text-color); font-weight: 700; font-size: 1.8rem; }
.signals-container { display: flex; flex-wrap: wrap; gap: 10px; }
.signal-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 0.95rem; background-color: var(--card-color); color: var(--text-color); border: 1px solid var(--border-color); }
.signal-badge.on { border-color: var(--accent-primary); background-color: rgba(145, 70, 255, 0.15); }
.signal-badge.off { color: var(--text-secondary); background-color: var(--card-color); }
.breakdown-container { display: flex; flex-direction: column; gap: 12px; }
.score-bar-row { display: flex; align-items: center; }
.score-label { width: 120px; font-weight: 600; color: var(--text-secondary); font-size: 0.9rem; text-transform: capitalize; }
.score-track { flex: 1; height: 12px; background-color: var(--border-color); border-radius: 6px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); transition: width 0.3s; }
.score-pct { width: 60px; text-align: right; color: var(--text-secondary); font-size: 0.85rem; font-weight: 600; }
.analysis-box { background-color: var(--card-color); border: 1px solid var(--border-color); border-left: 4px solid var(--accent-primary); border-radius: 12px; padding: 1.2rem; font-weight: 600; font-size: 1.05rem; margin-bottom: 1rem; color: var(--text-color); }
.footer { color: var(--text-secondary); font-size: 0.85rem; margin-top: 2rem; }
.status-indicator { position: absolute; top: 2rem; right: 2rem; padding: 6px 14px; border-radius: 20px; background-color: var(--card-color); color: var(--text-secondary); font-weight: 600; font-size: 0.85rem; border: 1px solid var(--border-color); }
.dashboard-main { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; align-items: start; }
.video-wrapper { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05); }
.video-header { background: var(--text-color); color: #0e0e10; padding: 0.6rem 1.2rem; font-weight: 700; font-size: 0.85rem; letter-spacing: 1.5px; }
.live-stream-img { width: 100%; aspect-ratio: 16/9; display: block; object-fit: cover; background-color: #1a1a2e; }
.confidence-box { background: var(--card-color); border: 1px solid var(--border-color); border-radius: 16px; padding: 2.5rem; color: var(--text-color); display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.4); transition: transform 0.3s ease, box-shadow 0.3s ease; }
.confidence-box:hover { transform: translateY(-2px); box-shadow: 0 12px 30px -5px rgba(145, 70, 255, 0.2); }
.conf-info { display: flex; flex-direction: column; }
.conf-label { font-size: 0.95rem; font-weight: 600; color: var(--text-secondary); letter-spacing: 1px; margin-bottom: 0.5rem; text-transform: uppercase; }
.conf-level { font-size: 2.8rem; font-weight: 800; line-height: 1; text-transform: uppercase; background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.conf-score { font-size: 4.8rem; font-weight: 800; line-height: 1; background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.timer-container { display: flex; gap: 1.5rem; margin: 1.5rem 0; }
.timer-box { flex: 1; background-color: var(--card-color); border: 1px solid var(--border-color); border-top: 3px solid var(--accent-primary); border-radius: 12px; padding: 1rem 1.5rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.timer-label { font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); letter-spacing: 2px; margin-bottom: 0.5rem; }
.timer-value { font-size: 2.2rem; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1; }
.neon-blue { color: var(--accent-secondary); text-shadow: 0 0 10px rgba(0, 230, 203, 0.25); }
.neon-red { color: #FF4A4A; text-shadow: 0 0 10px rgba(255, 74, 74, 0.3); }
.chart-container { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; height: 220px; width: 100%; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
@media (max-width: 1100px) { .dashboard-main { grid-template-columns: 1fr; } }
@media (max-width: 900px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .metrics-grid { grid-template-columns: 1fr; } .conf-level { font-size: 1.8rem; } .conf-score { font-size: 3rem; } }`;
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

let sessionSeconds = 0, awaySeconds = 0, isFaceDetected = false, attentionChart = null;
let awayLogs = [], currentAwayStart = null;

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
    const stale = json.lastUpdated && (Date.now() - json.lastUpdated > 2500);
    if (json.status === "waiting" || !json.report) { statusIndicator.textContent = "Waiting for data..."; isFaceDetected = false; return; }
    if (stale) { statusIndicator.textContent = "Camera Feed Frozen"; statusIndicator.style.color = "#FF4A4A"; statusIndicator.style.borderColor = "#FF4A4A"; isFaceDetected = false; }
    else { statusIndicator.textContent = "LIVE"; statusIndicator.style.color = "var(--text-color)"; statusIndicator.style.borderColor = "var(--text-color)"; }
    updateDashboard(json.report);
  } catch (err) { statusIndicator.textContent = "Connection lost"; statusIndicator.style.color = "var(--text-secondary)"; isFaceDetected = false; }
}

function updateDashboard(payload) {
  const report = payload.data || {};
  const sig = payload.sig || {};
  isFaceDetected = (sig.face_detected === true || String(sig.face_detected).toLowerCase() === "true");

  if (report.summary) {
    document.getElementById("conf-level").textContent = report.summary.focus_level || "UNKNOWN";
    document.getElementById("conf-score").textContent = Math.round((report.summary.attention_score||0)*100)+'%';
  } else {
    document.getElementById("conf-level").textContent = "WAITING...";
    document.getElementById("conf-score").textContent = "--%";
  }

  if (attentionChart) {
    const cs = report.summary ? Math.round((report.summary.attention_score||0)*100) : 0;
    attentionChart.data.datasets[0].data.shift();
    attentionChart.data.datasets[0].data.push(cs);
    attentionChart.update();
  }

  const eng = sig.engagement_score || 5;
  const tension = sig.micro_tension_score || 0;
  const eye = sig.eye_contact_score || 0;
  metricEng.textContent = eng+'/10';
  metricTension.textContent = tension+'/10';
  metricEye.textContent = Math.round(eye*100)+'%';
  metricPosture.textContent = sig.posture || "Closed";
  metricGaze.textContent = sig.gaze || "—";
  metricGestures.textContent = sig.gestures || "0";
  metricHead.textContent = sig.head_pose || "—";
  metricBlinks.textContent = Math.round(sig.blinks_per_minute || 0).toString();

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
    for (const [name, val] of sorted) {
      barsHtml += '<div class="score-bar-row"><span class="score-label">'+name+'</span><div class="score-track"><div class="score-fill" style="width:'+Math.max(2,val)+'%"></div></div><span class="score-pct">'+val.toFixed(1)+'%</span></div>';
    }
    breakdownContainer.innerHTML = barsHtml;
  } else { breakdownContainer.innerHTML = '<div style="color:var(--text-secondary);font-size:0.9rem;">Loading breakdown...</div>'; }

  let analysisHtml = "";
  if (eng >= 7) analysisHtml += '<div class="analysis-box">✅ Excellent Engagement ('+eng+'/10)</div>';
  else if (eng >= 4) analysisHtml += '<div class="analysis-box">⚠️ Moderate Engagement ('+eng+'/10)</div>';
  else analysisHtml += '<div class="analysis-box">🚨 Low Engagement ('+eng+'/10)</div>';
  if (tension >= 6) analysisHtml += '<div class="analysis-box">😰 High Tension ('+tension+'/10)</div>';
  analysisContainer.innerHTML = analysisHtml;
  footerText.textContent = 'Last updated: '+new Date().toLocaleTimeString()+' | Data source: /data';
}

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
  for (const log of awayLogs.slice(0,10)) {
    html += '<div class="analysis-box" style="padding:0.8rem 1.2rem;margin-bottom:0.5rem;font-size:0.95rem;">🛑 Away: <span style="color:var(--text-secondary)">'+log.start.toLocaleTimeString()+' - '+log.end.toLocaleTimeString()+'</span><span style="float:right;font-weight:700;">'+log.duration+'s</span></div>';
  }
  container.innerHTML = html;
}

setInterval(fetchData, FETCH_INTERVAL);
initChart();
fetchData();`;
}

module.exports = { getDashboardHTML, getDashboardCSS, getDashboardJS };
