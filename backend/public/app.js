/**
 * Trusted Advisor AI — Strict Theme Dashboard App
 * ================================================
 * Fetches /data every 400ms and updates DOM to match app.py layout 
 * with strict #0F0F0F, #2C2C2C, #EDEDED, #8A8A8A palette.
 */

const FETCH_INTERVAL = 400;
const DATA_URL = "/data";

// DOM Elements
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

// Globals
let sessionSeconds = 0;
let awaySeconds = 0;
let isFaceDetected = false;
let attentionChart = null;

let awayLogs = [];
let currentAwayStart = null;

function initChart() {
  const ctx = document.getElementById('attentionChart').getContext('2d');
  attentionChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array(50).fill(''),
      datasets: [{
        label: 'Attention',
        data: Array(50).fill(100),
        borderColor: '#00E6CB',
        backgroundColor: 'rgba(0, 230, 203, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        y: { min: 0, max: 100, grid: { color: '#2f2f35' } },
        x: { grid: { display: false } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function formatTime(s) {
  const hh = Math.floor(s / 3600).toString().padStart(2, '0');
  const mm = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
  const ss = (s % 60).toString().padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

setInterval(() => {
  sessionSeconds++;
  if (!isFaceDetected) {
    awaySeconds++;
    if (!currentAwayStart) {
      currentAwayStart = new Date();
    }
  } else {
    if (currentAwayStart) {
      const awayEnd = new Date();
      const dur = Math.round((awayEnd - currentAwayStart)/1000);
      if (dur >= 3) {
        awayLogs.unshift({
          start: currentAwayStart,
          end: awayEnd,
          duration: dur
        });
      }
      currentAwayStart = null;
    }
  }
  
  renderAwayLogs();

  document.getElementById('session-time').textContent = formatTime(sessionSeconds);
  document.getElementById('away-time').textContent = formatTime(awaySeconds);
}, 1000);

// Fetch Loop
async function fetchData() {
  try {
    const res = await fetch(DATA_URL);
    const json = await res.json();

    const stale = json.lastUpdated && (Date.now() - json.lastUpdated > 2500);

    if (json.status === "waiting" || !json.report) {
      statusIndicator.textContent = "Waiting for data...";
      isFaceDetected = false;
      return;
    }

    if (stale) {
      statusIndicator.textContent = "Camera Feed Frozen";
      statusIndicator.style.color = "#FF4A4A";
      statusIndicator.style.borderColor = "#FF4A4A";
      isFaceDetected = false;
    } else {
      statusIndicator.textContent = "LIVE";
      statusIndicator.style.color = "var(--text-color)";
      statusIndicator.style.borderColor = "var(--text-color)";
    }
    
    updateDashboard(json.report);
  } catch (err) {
    statusIndicator.textContent = "Connection lost";
    statusIndicator.style.color = "var(--text-secondary)";
    statusIndicator.style.borderColor = "var(--text-secondary)";
    isFaceDetected = false;
  }
}

function updateDashboard(payload) {
  // payload is { mode, data, sig } based on our app.js assumption 
  // actually the backend stores `json.report` which equals the POST body.
  // The POST body is `{"mode": "PROCTORING", "data": report, "sig": sig}`.
  // So payload = {"mode": "PROCTORING", "data": report, "sig": sig}
  
  const report = payload.data || {};
  const sig = payload.sig || {};

  // Guarantee strict boolean or explicitly true string
  isFaceDetected = (sig.face_detected === true || String(sig.face_detected).toLowerCase() === "true");

  // 0. Primary Confidence UI
  if (report.summary) {
    const focus = report.summary.focus_level || "UNKNOWN";
    const attnScore = report.summary.attention_score || 0;
    document.getElementById("conf-level").textContent = focus;
    document.getElementById("conf-score").textContent = `${Math.round(attnScore * 100)}%`;
  } else {
    document.getElementById("conf-level").textContent = "WAITING...";
    document.getElementById("conf-score").textContent = "--%";
  }

  // Chart Update
  if (attentionChart) {
    const currentScore = report.summary ? Math.round((report.summary.attention_score || 0) * 100) : 0;
    attentionChart.data.datasets[0].data.shift();
    attentionChart.data.datasets[0].data.push(currentScore);
    attentionChart.update();
  }

  // 2. Top Metrics
  const eng = sig.engagement_score || 5;
  const tension = sig.micro_tension_score || 0;
  const eye = sig.eye_contact_score || 0;
  const posture = sig.posture || "Closed";

  metricEng.textContent = `${eng}/10`;
  metricTension.textContent = `${tension}/10`;
  metricEye.textContent = `${Math.round(eye * 100)}%`;
  metricPosture.textContent = posture;

  // 3. Second Metrics
  metricGaze.textContent = sig.gaze || "—";
  metricGestures.textContent = sig.gestures || "0";
  metricHead.textContent = sig.head_pose || "—";
  metricBlinks.textContent = Math.round(sig.blinks_per_minute || 0).toString();

  // 4. Facial Signals
  let signalsHtml = "";
  signalsHtml += createBadge(`Smile: ${sig.smile_label || '-'}`, sig.smile_genuine);
  signalsHtml += createBadge(`Brow: ${sig.brow_label || '-'}`, sig.brow_label === 'RAISED');
  signalsHtml += createBadge(`Lip: ${sig.lip_label || '-'}`, sig.lip_label === 'RELAXED');
  signalsHtml += createBadge(`Nodding`, sig.nodding);
  signalsHtml += createBadge(`Head Shake`, sig.head_shake);
  signalsContainer.innerHTML = signalsHtml;

  // 5. Emotion Breakdown
  const emoAll = sig.emotion_all || {};
  if (Object.keys(emoAll).length > 0 && sig.emotion !== "loading" && sig.emotion !== "analyzing") {
    // Sort emotions by value descending
    const sortedEmos = Object.entries(emoAll).sort((a, b) => b[1] - a[1]);
    let barsHtml = "";
    for (const [emoName, emoVal] of sortedEmos) {
      const width = Math.max(2, emoVal);
      barsHtml += `
        <div class="score-bar-row">
          <span class="score-label">${emoName}</span>
          <div class="score-track">
            <div class="score-fill" style="width: ${width}%;"></div>
          </div>
          <span class="score-pct">${emoVal.toFixed(1)}%</span>
        </div>
      `;
    }
    breakdownContainer.innerHTML = barsHtml;
  } else {
    breakdownContainer.innerHTML = '<div style="color:var(--text-secondary); font-size:0.9rem;">Loading breakdown...</div>';
  }

  // 6. Behavior Analysis
  let analysisHtml = "";
  if (eng >= 7) {
    analysisHtml += `<div class="analysis-box">✅ Excellent Engagement (${eng}/10) — Confident, well-projected non-verbal communication.</div>`;
  } else if (eng >= 4) {
    analysisHtml += `<div class="analysis-box">⚠️ Moderate Engagement (${eng}/10) — Try more eye contact, open posture, and gestures.</div>`;
  } else {
    analysisHtml += `<div class="analysis-box">🚨 Low Engagement (${eng}/10) — Significant improvement needed in body language and expression.</div>`;
  }

  if (tension >= 6) {
    analysisHtml += `<div class="analysis-box">😰 High Tension (${tension}/10) — Signs of stress detected. Try relaxing your brow and jaw.</div>`;
  }
  
  analysisContainer.innerHTML = analysisHtml;

  // 7. Footer
  const now = new Date();
  footerText.textContent = `Last updated: ${now.toLocaleTimeString()} | Data source: /data`;
}

function createBadge(label, active) {
  const cls = active ? "on" : "off";
  const icon = active ? "✅" : "❌";
  return `<span class="signal-badge ${cls}">${icon} ${label}</span>`;
}

function renderAwayLogs() {
  const container = document.getElementById('away-log-container');
  if (!container) return;
  
  if (awayLogs.length === 0 && !currentAwayStart) {
    container.innerHTML = '<div style="color:var(--text-secondary); font-size:0.9rem;">No away intervals recorded.</div>';
    return;
  }
  
  let html = '';
  if (currentAwayStart) {
      html += `<div class="analysis-box" style="border-left-color: #FF4A4A; padding: 0.8rem 1.2rem; margin-bottom: 0.5rem; font-size:0.95rem;">
        🔴 <span style="font-weight:700">Currently Away</span> (since ${currentAwayStart.toLocaleTimeString()})
      </div>`;
  }
  
  for (const log of awayLogs.slice(0, 10)) {
      html += `<div class="analysis-box" style="padding: 0.8rem 1.2rem; margin-bottom: 0.5rem; font-size:0.95rem;">
        🛑 Away: <span style="color:var(--text-secondary)">${log.start.toLocaleTimeString()} - ${log.end.toLocaleTimeString()}</span>
        <span style="float:right; font-weight:700;">${log.duration}s</span>
      </div>`;
  }
  
  container.innerHTML = html;
}

setInterval(fetchData, FETCH_INTERVAL);
initChart();
fetchData();
