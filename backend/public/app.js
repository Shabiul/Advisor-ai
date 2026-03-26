/**
 * Trusted Advisor AI — Dashboard App
 * ====================================
 * Fetches /data every 400ms and updates all panels smoothly.
 */

const FETCH_INTERVAL = 400;
const DATA_URL = "/data";

// ── DOM References ───────────────────────────────────────────────────

const statusBadge = document.getElementById("status-badge");
const pulseDot = document.getElementById("pulse-dot");
const statusText = document.getElementById("status-text");

const scoreValue = document.getElementById("score-value");
const ringFill = document.getElementById("ring-fill");
const focusLabel = document.getElementById("focus-label");
const stabilityTag = document.getElementById("stability-tag");

const metricOffscreen = document.getElementById("metric-offscreen");
const metricHeaddown = document.getElementById("metric-headdown");
const metricFacemissing = document.getElementById("metric-facemissing");
const metricSlouch = document.getElementById("metric-slouch");
const barOffscreen = document.getElementById("bar-offscreen");
const barHeaddown = document.getElementById("bar-headdown");
const barFacemissing = document.getElementById("bar-facemissing");
const barSlouch = document.getElementById("bar-slouch");

const bodyPosture = document.getElementById("body-posture");
const bodyNeck = document.getElementById("body-neck");
const bodyShoulders = document.getElementById("body-shoulders");
const bodyRecommendation = document.getElementById("body-recommendation");

const alertCount = document.getElementById("alert-count");
const alertsList = document.getElementById("alerts-list");
const episodesList = document.getElementById("episodes-list");

// ── State ────────────────────────────────────────────────────────────

let isLive = false;

// ── Fetch Loop ───────────────────────────────────────────────────────

async function fetchData() {
  try {
    const res = await fetch(DATA_URL);
    const json = await res.json();

    if (json.status === "waiting") {
      setDisconnected("Waiting for data…");
      return;
    }

    setLive();
    updateDashboard(json.report);
  } catch (err) {
    setDisconnected("Connection lost");
  }
}

function setLive() {
  if (!isLive) {
    isLive = true;
    statusBadge.classList.add("live");
    pulseDot.classList.add("live");
    statusText.textContent = "LIVE";
  }
}

function setDisconnected(msg) {
  isLive = false;
  statusBadge.classList.remove("live");
  pulseDot.classList.remove("live");
  statusText.textContent = msg;
}

// ── Update Functions ─────────────────────────────────────────────────

function updateDashboard(report) {
  if (!report) return;

  updateAttention(report.summary);
  updateMetrics(report.metrics);
  updateBody(report.body_analysis, report.recommendation);
  updateAlerts(report.live_alerts || []);
  updateEpisodes(report.look_away_episodes || []);
}

function updateAttention(summary) {
  if (!summary) return;

  const score = summary.attention_score || 0;
  const focus = summary.focus_level || "N/A";
  const stability = summary.stability || "";

  // Score number
  scoreValue.textContent = Math.round(score * 100);

  // Ring
  const circumference = 326.73;
  const offset = circumference * (1 - score);
  ringFill.style.strokeDashoffset = offset;

  // Ring color
  if (score >= 0.7) {
    ringFill.style.stroke = "var(--accent-green)";
    scoreValue.style.color = "var(--accent-green)";
  } else if (score >= 0.4) {
    ringFill.style.stroke = "var(--accent-amber)";
    scoreValue.style.color = "var(--accent-amber)";
  } else {
    ringFill.style.stroke = "var(--accent-red)";
    scoreValue.style.color = "var(--accent-red)";
  }

  // Focus label
  focusLabel.textContent = focus;
  focusLabel.className = "focus-label " + focus;

  // Stability
  stabilityTag.textContent = stability ? `Stability: ${stability}` : "";
}

function updateMetrics(metrics) {
  if (!metrics) return;

  setMetric(metricOffscreen, barOffscreen, metrics.off_screen_time);
  setMetric(metricHeaddown, barHeaddown, metrics.head_down_time);
  setMetric(metricFacemissing, barFacemissing, metrics.face_missing_time);
  setMetric(metricSlouch, barSlouch, metrics.slouch_time);
}

function setMetric(valueEl, barEl, percentStr) {
  if (!percentStr) {
    valueEl.textContent = "0%";
    barEl.style.width = "0%";
    return;
  }

  valueEl.textContent = percentStr;
  const num = parseInt(percentStr, 10);
  barEl.style.width = num + "%";

  // Color coding
  if (num >= 30) {
    valueEl.style.color = "var(--accent-red)";
    barEl.style.background = "var(--accent-red)";
  } else if (num >= 15) {
    valueEl.style.color = "var(--accent-amber)";
    barEl.style.background = "var(--accent-amber)";
  } else {
    valueEl.style.color = "var(--accent-green)";
    barEl.style.background = "var(--accent-green)";
  }
}

function updateBody(body, recommendation) {
  if (!body) return;

  setBodyValue(bodyPosture, body.posture);
  setBodyValue(bodyNeck, body.neck);
  setBodyValue(bodyShoulders, body.shoulders);

  bodyRecommendation.textContent = recommendation || "—";
  if (recommendation === "GOOD") {
    bodyRecommendation.className = "body-value recommendation ok";
  } else if (recommendation === "ACCEPTABLE") {
    bodyRecommendation.className = "body-value recommendation warn";
  } else {
    bodyRecommendation.className = "body-value recommendation bad";
  }
}

function setBodyValue(el, value) {
  el.textContent = value || "—";
  const bad = ["SLOUCHED", "FORWARD_HEAD", "DOWN", "DROPPED", "TILTED"];
  const ok = ["UPRIGHT", "STRAIGHT", "ACTIVE", "NEUTRAL", "RELAXED"];

  if (bad.includes(value)) {
    el.className = "body-value bad";
  } else if (ok.includes(value)) {
    el.className = "body-value ok";
  } else {
    el.className = "body-value warn";
  }
}

function updateAlerts(alerts) {
  const count = alerts.filter(a => a.active).length;
  alertCount.textContent = count;

  if (alerts.length === 0) {
    alertsList.innerHTML = '<div class="empty-state">No active alerts</div>';
    return;
  }

  let html = "";
  for (const alert of alerts) {
    const cls = alert.active ? "active" : "completed";
    const typeName = formatEventType(alert.type);

    html += `
      <div class="alert-item ${cls}">
        <span class="alert-type">${typeName}</span>
        <span class="alert-time">${alert.start} → ${alert.end}</span>
        <span class="alert-duration">${alert.duration}</span>
        ${alert.active ? '<span class="alert-active-badge">LIVE</span>' : ''}
      </div>
    `;
  }

  alertsList.innerHTML = html;
}

function updateEpisodes(episodes) {
  if (episodes.length === 0) {
    episodesList.innerHTML = '<div class="empty-state">No look-away events recorded</div>';
    return;
  }

  let html = `
    <div class="episode-item" style="opacity:0.5; font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em;">
      <span>Duration</span>
      <span>Start</span>
      <span>End</span>
      <span>Timeline</span>
    </div>
  `;

  for (const ep of episodes) {
    const durSec = parseFloat(ep.duration);
    const barWidth = Math.min(durSec / 10 * 100, 100);
    const cls = ep.active ? "active" : "";

    html += `
      <div class="episode-item ${cls}">
        <span class="episode-duration">${ep.duration}</span>
        <span class="episode-start">${ep.start}</span>
        <span class="episode-end">${ep.end}</span>
        <div class="episode-bar">
          <div class="episode-bar-fill" style="width:${barWidth}%"></div>
        </div>
      </div>
    `;
  }

  episodesList.innerHTML = html;
}

// ── Helpers ──────────────────────────────────────────────────────────

function formatEventType(type) {
  const map = {
    LOOK_AWAY: "👀 Look Away",
    LOOK_DOWN: "⬇️ Look Down",
    FACE_MISSING: "🚫 Face Missing",
    SLOUCH: "🪑 Slouch",
    SHOULDER_DROP: "💪 Shoulder Drop",
    ARMS_CROSSED: "🤞 Arms Crossed",
  };
  return map[type] || type;
}

// ── Init ─────────────────────────────────────────────────────────────

setInterval(fetchData, FETCH_INTERVAL);
fetchData();
