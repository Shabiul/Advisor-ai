/**
 * Trusted Advisor AI — Dashboard App (v3.0 — Multimodal)
 * ======================================================
 * Fetches /data every 400ms and updates DOM with vision + audio emotion data.
 * Palette: #0F0F0F, #2C2C2C, #EDEDED, #8A8A8A
 */

const FETCH_INTERVAL = 400;
const DATA_URL = "/data";

// ── DOM Elements ─────────────────────────────────────────────────────

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

// Audio emotion DOM
const voiceMicIcon = document.getElementById("voice-mic-icon");
const voiceStatus = document.getElementById("voice-status");
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

// ── Globals ──────────────────────────────────────────────────────────

let sessionSeconds = 0;
let awaySeconds = 0;
let isFaceDetected = false;
let attentionChart = null;

let awayLogs = [];
let currentAwayStart = null;

// Track last audio update time for staleness detection
let lastAudioUpdateTime = 0;


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


// ── Fetch Loop ───────────────────────────────────────────────────────

async function fetchData() {
  try {
    const res = await fetch(DATA_URL);
    const json = await res.json();

    if (json.status === "waiting" || (!json.report && !json.audioEmotion)) {
      statusIndicator.textContent = "Waiting for data...";
      isFaceDetected = false;
      return;
    }

    const stale = json.lastUpdated && (Date.now() - new Date(json.lastUpdated).getTime() > 5000);

    if (stale && !json.audioEmotion) {
      statusIndicator.textContent = "Camera Feed Frozen";
      statusIndicator.style.color = "#FF4A4A";
      statusIndicator.style.borderColor = "#FF4A4A";
      isFaceDetected = false;
    } else {
      statusIndicator.textContent = "LIVE";
      statusIndicator.style.color = "var(--text-color)";
      statusIndicator.style.borderColor = "var(--text-color)";
    }

    // Update vision dashboard
    if (json.report) {
      updateVisionDashboard(json.report);
    }

    // Update audio emotion dashboard
    if (json.audioEmotion) {
      lastAudioUpdateTime = Date.now();
      updateAudioDashboard(json.audioEmotion);
    } else {
      // Check staleness of audio data
      if (lastAudioUpdateTime > 0 && (Date.now() - lastAudioUpdateTime > 5000)) {
        voiceStatus.textContent = "Audio service disconnected";
        voiceStatus.classList.remove("active");
        voiceMicIcon.classList.remove("speaking");
      }
    }

    // Update multimodal fusion
    if (json.report && json.audioEmotion) {
      updateMultimodalFusion(json.report, json.audioEmotion);
    }

  } catch (err) {
    statusIndicator.textContent = "Connection lost";
    statusIndicator.style.color = "var(--text-secondary)";
    statusIndicator.style.borderColor = "var(--text-secondary)";
    isFaceDetected = false;
  }
}


// ── Vision Dashboard ─────────────────────────────────────────────────

function updateVisionDashboard(payload) {
  const report = payload.data || {};
  const sig = payload.sig || {};

  isFaceDetected = (sig.face_detected === true || String(sig.face_detected).toLowerCase() === "true");

  // Primary Confidence UI
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

  // Top Metrics
  const eng = sig.engagement_score || 5;
  const tension = sig.micro_tension_score || 0;
  const eye = sig.eye_contact_score || 0;
  const posture = sig.posture || "Closed";

  metricEng.textContent = `${eng}/10`;
  metricTension.textContent = `${tension}/10`;
  metricEye.textContent = `${Math.round(eye * 100)}%`;
  metricPosture.textContent = posture;

  // Second Metrics
  metricGaze.textContent = sig.gaze || "—";
  metricGestures.textContent = sig.gestures || "0";
  metricHead.textContent = sig.head_pose || "—";
  metricBlinks.textContent = Math.round(sig.blinks_per_minute || 0).toString();

  // Facial Signals
  let signalsHtml = "";
  signalsHtml += createBadge(`Smile: ${sig.smile_label || '-'}`, sig.smile_genuine);
  signalsHtml += createBadge(`Brow: ${sig.brow_label || '-'}`, sig.brow_label === 'RAISED');
  signalsHtml += createBadge(`Lip: ${sig.lip_label || '-'}`, sig.lip_label === 'RELAXED');
  signalsHtml += createBadge(`Nodding`, sig.nodding);
  signalsHtml += createBadge(`Head Shake`, sig.head_shake);
  signalsContainer.innerHTML = signalsHtml;

  // Emotion Breakdown (vision-based from sig)
  const emoAll = sig.emotion_all || {};
  if (Object.keys(emoAll).length > 0 && sig.emotion !== "loading" && sig.emotion !== "analyzing") {
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

  // Behavior Analysis
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

  // Footer
  const now = new Date();
  footerText.textContent = `Last updated: ${now.toLocaleTimeString()} | Data source: /data`;
}


// ── Audio Emotion Dashboard ──────────────────────────────────────────

function updateAudioDashboard(audioPayload) {
  const ae = audioPayload.audio_emotion;
  if (!ae) return;

  const isSilence = (ae.status === "silence");
  const rms = ae.rms || 0;

  // Voice Activity
  if (isSilence) {
    voiceStatus.textContent = "Listening... (waiting for speech)";
    voiceStatus.classList.remove("active");
    voiceMicIcon.classList.remove("speaking");
  } else {
    voiceStatus.textContent = "Speech Detected";
    voiceStatus.classList.add("active");
    voiceMicIcon.classList.add("speaking");
  }

  // RMS energy bar (scale: 0-0.1 maps to 0-100%)
  const energyPct = Math.min(100, Math.round(rms * 1000));
  voiceEnergyFill.style.width = `${energyPct}%`;
  voiceRms.textContent = rms.toFixed(4);
  voiceTs.textContent = ae.timestamp || "—";

  // Audio Emotion label + confidence
  if (!isSilence && ae.label) {
    audioEmotionLabel.textContent = ae.label.toUpperCase();
    audioEmotionConf.textContent = `${Math.round((ae.confidence || 0) * 100)}%`;
  } else {
    audioEmotionLabel.textContent = isSilence ? "LISTENING..." : "WAITING...";
    audioEmotionConf.textContent = "—%";
  }

  // Audio category breakdown bars
  if (!isSilence && ae.all_scores && Object.keys(ae.all_scores).length > 0) {
    const sorted = Object.entries(ae.all_scores).sort((a, b) => b[1] - a[1]);
    let barsHtml = "";
    for (const [emo, score] of sorted) {
      const pct = Math.round(score * 100);
      const width = Math.max(2, pct);
      barsHtml += `
        <div class="score-bar-row">
          <span class="score-label">${emo}</span>
          <div class="score-track">
            <div class="score-fill" style="width: ${width}%;"></div>
          </div>
          <span class="score-pct">${pct}%</span>
        </div>
      `;
    }
    audioBreakdownContainer.innerHTML = barsHtml;
  }

  // VAD Dimensions
  if (!isSilence) {
    metricValence.textContent = formatVAD(ae.valence);
    metricArousal.textContent = formatVAD(ae.arousal);
    metricDominance.textContent = formatVAD(ae.dominance);
    metricVadQuad.textContent = ae.vad_quadrant || "—";
  }

  // Transcript
  const transcript = audioPayload.transcript;
  if (transcript && transcript.length > 0) {
    let html = "";
    for (const entry of transcript) {
      html += `
        <div class="transcript-entry">
          <span class="transcript-ts">${entry.ts}</span>
          <span class="transcript-text">${escapeHtml(entry.text)}</span>
        </div>
      `;
    }
    transcriptBody.innerHTML = html;
    // Auto-scroll to bottom
    transcriptBody.scrollTop = transcriptBody.scrollHeight;
  }
}

function formatVAD(val) {
  if (val === undefined || val === null) return "—";
  const num = parseFloat(val);
  return (num >= 0 ? "+" : "") + num.toFixed(3);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}


// ── Multimodal Fusion ────────────────────────────────────────────────

function updateMultimodalFusion(visionPayload, audioPayload) {
  const sig = visionPayload.sig || {};
  const report = visionPayload.data || {};
  const ae = audioPayload.audio_emotion || {};

  // Check if report has pre-computed multimodal fusion from main.py
  const serverMultimodal = report.multimodal_emotion;

  if (serverMultimodal && serverMultimodal.fused_emotion) {
    // Use server-side fusion (richer — uses facial signal analysis)
    fusionVision.textContent = capitalize(serverMultimodal.vision_emotion || "—");
    fusionAudio.textContent = capitalize(serverMultimodal.audio_emotion || "—");
    fusionCombined.textContent = capitalize(serverMultimodal.fused_emotion);

    const congruence = Math.round((serverMultimodal.congruence || 0) * 100);
    congruenceFill.style.width = `${congruence}%`;
    congruencePct.textContent = `${congruence}%`;

    // Render behavioral insights from server
    const insights = serverMultimodal.behavioral_insights || [];
    if (insights.length > 0) {
      let html = '<span class="insight-label">🧠 AI BEHAVIORAL INSIGHTS</span>';
      for (const insight of insights) {
        const icon = insight.includes("congruence") || insight.includes("genuinely") || insight.includes("receptive") ? "✅"
                   : insight.includes("mismatch") || insight.includes("suppressing") || insight.includes("stress") ? "🚨"
                   : "⚠️";
        html += `<div style="margin:4px 0;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">${icon} ${insight}</div>`;
      }
      multimodalInsight.innerHTML = html;
    }
    return;
  }

  // Fallback: client-side fusion
  const visionEmotion = sig.emotion || "—";
  const audioEmotion = (ae.status !== "silence" && ae.label) ? ae.label : "—";
  const fused = audioPayload.fused_emotion || null;

  fusionVision.textContent = capitalize(visionEmotion);
  fusionAudio.textContent = capitalize(audioEmotion);

  if (fused && fused.label) {
    fusionCombined.textContent = capitalize(fused.label);
  } else if (audioEmotion !== "—") {
    fusionCombined.textContent = capitalize(audioEmotion);
  } else {
    fusionCombined.textContent = capitalize(visionEmotion);
  }

  let congruence = 0;
  if (visionEmotion !== "—" && audioEmotion !== "—") {
    if (visionEmotion.toLowerCase() === audioEmotion.toLowerCase()) {
      congruence = 100;
    } else {
      congruence = calculateCongruence(visionEmotion, audioEmotion);
    }
  }

  congruenceFill.style.width = `${congruence}%`;
  congruencePct.textContent = `${congruence}%`;

  if (visionEmotion !== "—" && audioEmotion !== "—") {
    if (congruence >= 80) {
      multimodalInsight.innerHTML = `
        <span class="insight-label">MULTIMODAL INSIGHT</span>
        ✅ Strong congruence — facial expression and vocal tone align on <strong>${capitalize(audioEmotion)}</strong>.
      `;
    } else if (congruence >= 40) {
      multimodalInsight.innerHTML = `
        <span class="insight-label">MULTIMODAL INSIGHT</span>
        ⚠️ Mixed signals — face shows <strong>${capitalize(visionEmotion)}</strong> but voice indicates <strong>${capitalize(audioEmotion)}</strong>. Client may be masking emotions.
      `;
    } else {
      multimodalInsight.innerHTML = `
        <span class="insight-label">MULTIMODAL INSIGHT</span>
        🚨 Emotional incongruence — strong mismatch between visual (<strong>${capitalize(visionEmotion)}</strong>) and vocal (<strong>${capitalize(audioEmotion)}</strong>) signals. Investigate further.
      `;
    }
  } else {
    multimodalInsight.innerHTML = "";
  }
}

function capitalize(str) {
  if (!str || str === "—") return "—";
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

// Emotion valence groups for congruence scoring
const EMOTION_VALENCE = {
  happy: 1, calm: 0.7, neutral: 0.5, surprised: 0.4,
  sad: -0.3, fear: -0.5, fearful: -0.5, angry: -0.7,
  disgust: -0.8
};

function calculateCongruence(emo1, emo2) {
  const v1 = EMOTION_VALENCE[emo1.toLowerCase()] ?? 0;
  const v2 = EMOTION_VALENCE[emo2.toLowerCase()] ?? 0;
  const diff = Math.abs(v1 - v2);
  // Max possible diff is 1.8 (happy vs disgust); scale to 0-100
  return Math.round(Math.max(0, (1 - diff / 1.8)) * 100);
}


// ── Helpers ──────────────────────────────────────────────────────────

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


// ── Init ─────────────────────────────────────────────────────────────

setInterval(fetchData, FETCH_INTERVAL);
initChart();
fetchData();
