"""
Trusted Advisor AI — Streamlit Dashboard
==========================================
Reads live_data.json (enriched signals) and renders real-time behavioural metrics.
"""

import streamlit as st
import json
import os
import time
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "ta_face_signals.json")

st.set_page_config(
    page_title="Trusted Advisor AI",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a8b2d1 !important; font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6f1ff !important; font-weight: 700;
    }

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem;
    }
    .sub-header { color: #8892b0; font-size: 1rem; margin-bottom: 1.5rem; }

    .signal-badge {
        display: inline-block; padding: 6px 18px; border-radius: 20px;
        font-weight: 600; font-size: 0.9rem; margin: 4px;
    }
    .signal-on {
        background: rgba(100,255,218,0.15); color: #64ffda;
        border: 1px solid rgba(100,255,218,0.3);
    }
    .signal-off {
        background: rgba(255,107,107,0.12); color: #ff6b6b;
        border: 1px solid rgba(255,107,107,0.25);
    }

    .analysis-box {
        border-radius: 12px; padding: 20px; margin: 10px 0;
        font-weight: 600; font-size: 1.05rem;
    }
    .analysis-success {
        background: rgba(100,255,218,0.1); border-left: 4px solid #64ffda; color: #64ffda;
    }
    .analysis-warning {
        background: rgba(255,193,7,0.1); border-left: 4px solid #ffc107; color: #ffc107;
    }
    .analysis-danger {
        background: rgba(255,107,107,0.1); border-left: 4px solid #ff6b6b; color: #ff6b6b;
    }

    .emotion-chip {
        display: inline-block; padding: 8px 20px; border-radius: 25px;
        font-weight: 700; font-size: 1.1rem; margin: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; box-shadow: 0 2px 8px rgba(102,126,234,0.3);
    }
    .score-bar {
        display: flex; align-items: center; margin: 6px 0;
    }
    .score-label {
        width: 100px; font-weight: 600; color: #a8b2d1; font-size: 0.85rem;
    }
    .score-fill {
        height: 10px; border-radius: 5px; transition: width 0.3s;
    }
    .score-track {
        flex: 1; height: 10px; background: rgba(255,255,255,0.08);
        border-radius: 5px; overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_data() -> dict:
    defaults = {
        "emotion": "loading", "emotion_conf": 0, "emotion_all": {},
        "gaze": "-", "eye_contact_score": 0.5,
        "head_pose": "-", "nodding": False, "head_shake": False,
        "blinks_per_minute": 0,
        "brow_score": 0.5, "brow_label": "-",
        "lip_score": 0.5, "lip_label": "-",
        "smile_genuine": False, "smile_label": "-",
        "micro_tension_score": 0, "engagement_score": 5,
        "posture": "Closed", "gestures": 0,
        "timestamp": 0,
        "current_looking_away_seconds": 0.0,
        "total_away_time_seconds": 0.0,
        "total_away_events": 0,
        "proctor_alert": False,
    }
    if not os.path.exists(DATA_FILE):
        return defaults
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    except (json.JSONDecodeError, IOError):
        return defaults


# ---------------------------------------------------------------------------
# Session state for history
# ---------------------------------------------------------------------------
if "engagement_history" not in st.session_state:
    st.session_state.engagement_history = []
if "tension_history" not in st.session_state:
    st.session_state.tension_history = []

data = load_data()

st.session_state.engagement_history.append(data["engagement_score"])
st.session_state.tension_history.append(data["micro_tension_score"])
for hist in [st.session_state.engagement_history, st.session_state.tension_history]:
    if len(hist) > 60:
        del hist[:-60]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🧠 Trusted Advisor AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Behavioral Intelligence Dashboard</div>', unsafe_allow_html=True)
if data.get("proctor_alert") == True:
    st.error("🚨 PROCTOR ALERT: User has been looking away from the screen for too long! ")

# ---------------------------------------------------------------------------
# Emotion banners (Visual + Audio + Multimodal)
# ---------------------------------------------------------------------------
emo = data["emotion"].upper()
emo_conf = data["emotion_conf"]
audio_emo = data.get("audio_emotion", {})
multimodal_emo = data.get("multimodal_emotion", {})

chips_html = '<div style="text-align:center; margin: 10px 0 20px 0;">'
chips_html += f'<span class="emotion-chip" style="background:linear-gradient(135deg,#667eea,#764ba2)">👁️ {emo} — {emo_conf:.0f}%</span>'
if audio_emo.get("label"):
    a_label = audio_emo["label"].upper()
    a_conf = audio_emo.get("confidence", 0) * 100
    chips_html += f'<span class="emotion-chip" style="background:linear-gradient(135deg,#f093fb,#f5576c)">🎙️ {a_label} — {a_conf:.0f}%</span>'
if multimodal_emo.get("label"):
    m_label = multimodal_emo["label"].upper()
    m_conf = multimodal_emo.get("confidence", 0) * 100
    chips_html += f'<span class="emotion-chip" style="background:linear-gradient(135deg,#43e97b,#38f9d7)">🧠 {m_label} — {m_conf:.0f}%</span>'
chips_html += '</div>'
st.markdown(chips_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top metrics (4 cols)
# ---------------------------------------------------------------------------
st.markdown("### 📊 Key Metrics")
c1, c2, c3, c4 = st.columns(4)

eng = data["engagement_score"]
c1.metric("⚡ Engagement", f"{eng}/10")
c2.metric("😰 Tension", f"{data['micro_tension_score']}/10")
ec_pct = int(data["eye_contact_score"] * 100)
c3.metric("👁️ Eye Contact", f"{ec_pct}%")
c4.metric("🧍 Posture", data["posture"])

# ---------------------------------------------------------------------------
# Second row of metrics
# ---------------------------------------------------------------------------
c5, c6, c7, c8 = st.columns(4)
c5.metric("👀 Gaze", data["gaze"])
c6.metric("🤚 Gestures", data["gestures"])
c7.metric("😊 Head Pose", data["head_pose"])
c8.metric("👁️ Blinks/min", f"{data['blinks_per_minute']:.0f}")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Custom Focus Timers (Your newly engineered logic!)
# ---------------------------------------------------------------------------
st.markdown("### ⏱️ Focus & Attention Tracking")
t1, t2, t3 = st.columns(3)

curr_away = data.get("current_looking_away_seconds", 0)
tot_away = data.get("total_away_time_seconds", 0)
away_events = data.get("total_away_events", 0)

t1.metric("Current Away Time", f"{curr_away:.1f}s")
t2.metric("Total Distraction Time", f"{tot_away:.1f}s")
t3.metric("Look-Away Events", away_events)

# ---------------------------------------------------------------------------
# NEW: Distraction Extraction Log for Proctor
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True) 
st.markdown("#### 📝 Distraction Log (Extracted Events)")

history = data.get("away_history", [])
if len(history) > 0:
    # Convert the raw JSON data into a clean pandas table
    df = pd.DataFrame(history)
    
    # Rename the columns so they look professional for the proctor
    df.columns = ["Session Timestamp (Seconds)", "Duration of Distraction (Seconds)"]
    
    # Display the table on the dashboard
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("No distraction events recorded yet.")

# ---------------------------------------------------------------------------
# Facial Signals
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 😊 Facial & Body Signals")


def signal_badge(label, active):
    cls = "signal-on" if active else "signal-off"
    icon = "✅" if active else "❌"
    return f'<span class="signal-badge {cls}">{icon} {label}</span>'


signals_html = "".join([
    signal_badge(f"Smile: {data['smile_label']}", data["smile_genuine"]),
    signal_badge(f"Brow: {data['brow_label']}", data["brow_label"] == "RAISED"),
    signal_badge(f"Lip: {data['lip_label']}", data["lip_label"] == "RELAXED"),
    signal_badge("Nodding", data["nodding"]),
    signal_badge("Head Shake", data["head_shake"]),
])
st.markdown(signals_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Emotion breakdown (if available)
# ---------------------------------------------------------------------------
emo_all = data.get("emotion_all", {})
if emo_all and data["emotion"] not in ["loading", "analyzing"]:
    st.markdown("---")
    st.markdown("### 🎭 Emotion Breakdown")
    sorted_emos = sorted(emo_all.items(), key=lambda x: x[1], reverse=True)

    emo_colors = {
        "happy": "#64ffda", "neutral": "#a8b2d1", "sad": "#74b9ff",
        "angry": "#ff6b6b", "surprise": "#feca57", "fear": "#a29bfe",
        "disgust": "#fd79a8",
    }
    bars_html = ""
    for emo_name, emo_val in sorted_emos:
        color = emo_colors.get(emo_name, "#a8b2d1")
        width = max(2, emo_val)
        bars_html += (
            f'<div class="score-bar">'
            f'<span class="score-label">{emo_name.title()}</span>'
            f'<div class="score-track">'
            f'<div class="score-fill" style="width:{width}%;background:{color}"></div>'
            f'</div>'
            f'<span style="color:#8892b0;font-size:0.8rem;margin-left:8px;">{emo_val:.1f}%</span>'
            f'</div>'
        )
    st.markdown(bars_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Audio Emotion Breakdown (from acoustic bridge)
# ---------------------------------------------------------------------------
audio_emo = data.get("audio_emotion", {})
audio_all = audio_emo.get("all_scores", {})
if audio_all:
    st.markdown("---")
    st.markdown("### 🎙️ Acoustic Emotion (Voice)")

    emo_colors_audio = {
        "happy": "#64ffda", "neutral": "#a8b2d1", "sad": "#74b9ff",
        "angry": "#ff6b6b", "surprised": "#feca57", "fear": "#a29bfe",
        "disgust": "#fd79a8", "calm": "#00cec9", "fearful": "#a29bfe",
    }
    sorted_audio = sorted(audio_all.items(), key=lambda x: x[1], reverse=True)
    bars_html_a = ""
    for emo_name, emo_val in sorted_audio:
        color = emo_colors_audio.get(emo_name, "#a8b2d1")
        width = max(2, emo_val * 100)
        bars_html_a += (
            f'<div class="score-bar">'
            f'<span class="score-label">{emo_name.title()}</span>'
            f'<div class="score-track">'
            f'<div class="score-fill" style="width:{width}%;background:{color}"></div>'
            f'</div>'
            f'<span style="color:#8892b0;font-size:0.8rem;margin-left:8px;">{emo_val:.0%}</span>'
            f'</div>'
        )
    st.markdown(bars_html_a, unsafe_allow_html=True)

    # VAD Dimensional scores
    v = audio_emo.get("valence", 0)
    a = audio_emo.get("arousal", 0)
    d = audio_emo.get("dominance", 0)
    quad = audio_emo.get("vad_quadrant", "")

    vad_cols = st.columns(4)
    vad_cols[0].metric("Valence", f"{v:+.2f}")
    vad_cols[1].metric("Arousal", f"{a:+.2f}")
    vad_cols[2].metric("Dominance", f"{d:+.2f}")
    vad_cols[3].metric("VAD Quadrant", quad or "-")

# ---------------------------------------------------------------------------
# Live Transcript (from Canary-Qwen)
# ---------------------------------------------------------------------------
transcript = data.get("live_transcript") or []
if transcript:
    st.markdown("---")
    st.markdown("### 📝 Live Transcript")
    transcript_html = '<div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px;max-height:200px;overflow-y:auto;">'
    for entry in transcript:
        if isinstance(entry, dict):
            ts = entry.get("ts", "")
            txt = entry.get("text", "")
            transcript_html += f'<div style="margin:4px 0;"><span style="color:#667eea;font-weight:600;">[{ts}]</span> <span style="color:#e6f1ff;">{txt}</span></div>'
    transcript_html += '</div>'
    st.markdown(transcript_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Behavior Analysis
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Behavior Analysis")

tension = data["micro_tension_score"]
if eng >= 7:
    st.markdown(
        f'<div class="analysis-box analysis-success">'
        f'✅ Excellent Engagement ({eng}/10) — Confident, well-projected non-verbal communication.'
        f'</div>',
        unsafe_allow_html=True,
    )
elif eng >= 4:
    st.markdown(
        f'<div class="analysis-box analysis-warning">'
        f'⚠️ Moderate Engagement ({eng}/10) — Try more eye contact, open posture, and gestures.'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="analysis-box analysis-danger">'
        f'🚨 Low Engagement ({eng}/10) — Significant improvement needed in body language and expression.'
        f'</div>',
        unsafe_allow_html=True,
    )

if tension >= 6:
    st.markdown(
        f'<div class="analysis-box analysis-danger">'
        f'😰 High Tension ({tension}/10) — Signs of stress detected. Try relaxing your brow and jaw.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 🔀 Multimodal Behavioral Insights (Face + Voice Fusion)
# ---------------------------------------------------------------------------
multimodal = data.get("multimodal_emotion", {})
if multimodal and multimodal.get("fused_emotion"):
    st.markdown("---")
    st.markdown("### 🔀 Multimodal Behavioral Intelligence")

    # Fusion overview chips
    vision_emo = multimodal.get("vision_emotion", "—").upper()
    audio_emo = multimodal.get("audio_emotion", "—").upper()
    fused_emo = multimodal.get("fused_emotion", "—").upper()
    fused_conf = multimodal.get("fused_confidence", 0)
    congruence = multimodal.get("congruence", 0)
    congruence_level = multimodal.get("congruence_level", "—")

    # Fusion comparison row
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("👁️ Face Emotion", vision_emo)
    fc2.metric("🎙️ Voice Emotion", audio_emo)
    fc3.metric("🧠 Fused Emotion", f"{fused_emo} ({fused_conf:.0%})")

    # Congruence metric
    cong_color = "#64ffda" if congruence >= 0.7 else "#ffc107" if congruence >= 0.4 else "#ff6b6b"
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px;margin:10px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<span style="color:#a8b2d1;font-weight:600;">Emotional Congruence</span>'
        f'<span style="color:{cong_color};font-weight:700;font-size:1.2rem;">{congruence:.0%} ({congruence_level})</span>'
        f'</div>'
        f'<div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;overflow:hidden;">'
        f'<div style="width:{congruence*100}%;height:100%;background:{cong_color};border-radius:4px;transition:width 0.3s;"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # VAD dimensions from audio
    vad = multimodal.get("vad", {})
    if vad:
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("💓 Valence", f"{vad.get('valence', 0):+.2f}")
        vc2.metric("⚡ Arousal", f"{vad.get('arousal', 0):+.2f}")
        vc3.metric("👑 Dominance", f"{vad.get('dominance', 0):+.2f}")

    # Behavioral insights (the key value-add)
    insights = multimodal.get("behavioral_insights", [])
    if insights:
        st.markdown("#### 🧠 AI Behavioral Insights")
        for insight in insights:
            if "congruence" in insight.lower() or "genuinely" in insight.lower() or "receptive" in insight.lower():
                box_class = "analysis-success"
            elif "mismatch" in insight.lower() or "suppressing" in insight.lower() or "stress" in insight.lower():
                box_class = "analysis-danger"
            else:
                box_class = "analysis-warning"

            st.markdown(
                f'<div class="analysis-box {box_class}">{insight}</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Communication Breakdown Chart
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Communication Breakdown")

chart_data = pd.DataFrame({
    "Category": ["Body Language", "Tone", "Words"],
    "Impact (%)": [55, 38, 7],
})
st.bar_chart(chart_data.set_index("Category"), use_container_width=True)

# ---------------------------------------------------------------------------
# Engagement & Tension Timeline
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📉 Engagement & Tension Timeline")

timeline_df = pd.DataFrame({
    "Engagement": st.session_state.engagement_history,
    "Tension": st.session_state.tension_history,
})
st.line_chart(timeline_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
ts = data.get("timestamp", 0)
ts_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "N/A"
st.caption(f"Last updated: {ts_str} | Data source: `{DATA_FILE}`")

# ---------------------------------------------------------------------------
# Auto-refresh ~1s
# ---------------------------------------------------------------------------
time.sleep(1)
st.rerun()
