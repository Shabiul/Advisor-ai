/**
 * Trusted Advisor SDK — Behavioral Interpreter (Multimodal v3.0)
 * ===============================================================
 * Mode-based interpretation engine with multimodal intelligence.
 *
 * Vision-only:
 *   - interpret(report)     → vision-only flags + assessment
 *   - getRiskScore(report)  → 0-100 risk score
 *
 * Multimodal (vision + audio):
 *   - interpretMultimodal(report, audioData)  → fused analysis
 *   - getEmotionalState(report, audioData)    → combined emotion
 *   - detectInconsistencies(report, audioData) → cross-modal conflicts
 *
 * Supports: PROCTORING, MEETING modes.
 */

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

// Emotion valence mapping for congruence scoring
const EMOTION_VALENCE = {
  happy: 1.0, calm: 0.7, neutral: 0.5, surprised: 0.4,
  sad: -0.3, fear: -0.5, fearful: -0.5,
  angry: -0.7, disgust: -0.8,
};

// Fusion weights: how much to trust each modality
const FUSION_WEIGHTS = {
  PROCTORING: { vision: 0.6, audio: 0.4 }, // watching matters more
  MEETING:    { vision: 0.4, audio: 0.6 }, // voice matters more in meetings
};

// Known cross-modal inconsistency patterns
const INCONSISTENCY_PATTERNS = [
  { vision: "happy",   audio: "angry",   severity: "HIGH",   label: "MASKED_ANGER",      message: "Smiling face but angry voice — possible suppressed frustration." },
  { vision: "happy",   audio: "sad",     severity: "HIGH",   label: "MASKED_SADNESS",    message: "Smiling face but sad voice — social masking of negative emotions." },
  { vision: "happy",   audio: "fear",    severity: "HIGH",   label: "NERVOUS_SMILE",     message: "Smiling face but fearful voice — potential anxiety or nervous laughter." },
  { vision: "neutral", audio: "angry",   severity: "MEDIUM", label: "HIDDEN_FRUSTRATION", message: "Flat expression but angry voice — internally frustrated." },
  { vision: "neutral", audio: "happy",   severity: "LOW",    label: "RESTRAINED_JOY",    message: "Neutral face but happy voice — restrained positive emotion." },
  { vision: "angry",   audio: "happy",   severity: "MEDIUM", label: "CONFLICTED",        message: "Angry face but happy voice — conflicting emotional signals." },
  { vision: "sad",     audio: "happy",   severity: "MEDIUM", label: "FORCED_POSITIVITY",  message: "Sad face but happy voice — forced positivity." },
  { vision: "angry",   audio: "calm",    severity: "LOW",    label: "CONTROLLED_ANGER",  message: "Angry face but calm voice — controlled frustration." },
];

// ─────────────────────────────────────────────────────────────────────
//  INTERPRETER
// ─────────────────────────────────────────────────────────────────────

class Interpreter {
  /**
   * @param {string} mode - "PROCTORING" or "MEETING"
   * @param {Object} [customThresholds] - Override default thresholds
   */
  constructor(mode = "PROCTORING", customThresholds = null) {
    this.mode = mode.toUpperCase();
    this.thresholds = customThresholds || THRESHOLDS[this.mode];
    if (!this.thresholds) {
      throw new Error(`Unknown mode '${mode}'. Supported: PROCTORING, MEETING.`);
    }
  }

  // ── EXISTING: Vision-only ──────────────────────────────────────────

  /**
   * Interpret a cumulative report and return flags + assessment.
   * @param {Object} report - Report from CumulativeReporter
   * @returns {Object} { flags, suspicion_level?, engagement_level?, recommendation }
   */
  interpret(report) {
    const summary = report.summary || {};
    const metrics = report.metrics || {};
    const alerts = report.alerts || [];
    const body = report.body_analysis || {};
    const t = this.thresholds;

    const flags = [];

    // Attention check
    if ((summary.attention_score || 1) < t.attention_low) {
      flags.push({
        severity: this.mode === "PROCTORING" ? "HIGH" : "MEDIUM",
        message: `Attention score (${summary.attention_score}) is below threshold (${t.attention_low}).`,
      });
    }

    // Off-screen time
    const offScreen = this._parsePercent(metrics.off_screen_time);
    if (offScreen > t.off_screen_warn) {
      flags.push({
        severity: this.mode === "PROCTORING" ? "HIGH" : "LOW",
        message: `Off-screen time (${metrics.off_screen_time}) exceeds limit (${(t.off_screen_warn * 100).toFixed(0)}%).`,
      });
    }

    // Head down
    const headDown = this._parsePercent(metrics.head_down_time);
    if (headDown > t.head_down_warn) {
      flags.push({
        severity: "MEDIUM",
        message: `Head-down time (${metrics.head_down_time}) exceeds limit (${(t.head_down_warn * 100).toFixed(0)}%).`,
      });
    }

    // Face missing
    const faceMissing = this._parsePercent(metrics.face_missing_time);
    if (faceMissing > t.face_missing_warn) {
      flags.push({
        severity: this.mode === "PROCTORING" ? "HIGH" : "MEDIUM",
        message: `Face missing (${metrics.face_missing_time}) exceeds limit (${(t.face_missing_warn * 100).toFixed(0)}%).`,
      });
    }

    // Alert count
    if (alerts.length >= t.alert_count_limit) {
      flags.push({
        severity: "HIGH",
        message: `${alerts.length} behavioral alerts detected (limit: ${t.alert_count_limit}).`,
      });
    }

    // Body posture
    if (body.posture === "SLOUCHED") {
      flags.push({ severity: "LOW", message: "Subject appears slouched." });
    }
    if (body.neck === "FORWARD_HEAD") {
      flags.push({ severity: "LOW", message: "Forward head posture detected." });
    }

    const result = {
      flags,
      recommendation: report.recommendation || "REVIEW_REQUIRED",
    };

    // Mode-specific assessment
    if (this.mode === "PROCTORING") {
      const highCount = flags.filter((f) => f.severity === "HIGH").length;
      if (highCount >= 3) result.suspicion_level = "HIGH";
      else if (highCount >= 1) result.suspicion_level = "MODERATE";
      else result.suspicion_level = "LOW";
    }

    if (this.mode === "MEETING") {
      const score = summary.attention_score || 0;
      if (score >= 0.75) result.engagement_level = "HIGHLY_ENGAGED";
      else if (score >= 0.50) result.engagement_level = "ENGAGED";
      else if (score >= 0.30) result.engagement_level = "PARTIALLY_ENGAGED";
      else result.engagement_level = "DISENGAGED";
    }

    return result;
  }

  /**
   * Quick convenience: returns an overall risk score from 0–100.
   * @param {Object} report
   * @returns {number}
   */
  getRiskScore(report) {
    const { flags } = this.interpret(report);
    let score = 0;
    for (const flag of flags) {
      if (flag.severity === "HIGH") score += 30;
      else if (flag.severity === "MEDIUM") score += 15;
      else score += 5;
    }
    return Math.min(100, score);
  }

  // ── NEW: Multimodal (Vision + Audio) ───────────────────────────────

  /**
   * Full multimodal interpretation. Fuses vision report with audio data.
   * @param {Object} report - Vision report from CumulativeReporter
   * @param {Object} audioData - Audio emotion payload from emo_service
   * @returns {Object} Complete multimodal interpretation
   */
  interpretMultimodal(report, audioData) {
    // Start with vision interpretation
    const visionResult = this.interpret(report);
    const ae = (audioData && audioData.audio_emotion) ? audioData.audio_emotion : null;

    if (!ae || ae.status === "silence" || !ae.label) {
      // No audio data — return vision-only with marker
      return { ...visionResult, modality: "vision_only", multimodal: null };
    }

    const emotionalState = this.getEmotionalState(report, audioData);
    const inconsistencies = this.detectInconsistencies(report, audioData);
    const audioFlags = this._getAudioFlags(ae);

    // Merge audio flags into vision flags
    const mergedFlags = [...visionResult.flags, ...audioFlags, ...inconsistencies.map(i => ({
      severity: i.severity,
      message: `[MULTIMODAL] ${i.message}`,
    }))];

    // Recalculate risk with both modalities
    const weights = FUSION_WEIGHTS[this.mode] || FUSION_WEIGHTS.PROCTORING;
    const visionRisk = this.getRiskScore(report);
    const audioRisk = this._getAudioRiskScore(ae, inconsistencies);
    const fusedRisk = Math.round(visionRisk * weights.vision + audioRisk * weights.audio);

    const result = {
      ...visionResult,
      flags: mergedFlags,
      modality: "multimodal",
      fused_risk_score: Math.min(100, fusedRisk),
      emotional_state: emotionalState,
      inconsistencies,
      multimodal: {
        vision_emotion: emotionalState.vision_emotion,
        audio_emotion: emotionalState.audio_emotion,
        fused_emotion: emotionalState.fused_emotion,
        congruence: emotionalState.congruence,
        congruence_level: emotionalState.congruence_level,
        behavioral_insights: emotionalState.insights,
        vad: {
          valence: ae.valence || 0,
          arousal: ae.arousal || 0,
          dominance: ae.dominance || 0,
          quadrant: ae.vad_quadrant || "",
        },
      },
    };

    // Override engagement level with multimodal data
    if (this.mode === "MEETING") {
      const attScore = (report.summary || {}).attention_score || 0;
      const audioPositive = (ae.valence || 0) > 0.1;
      const fused = attScore * weights.vision + (audioPositive ? 0.8 : 0.3) * weights.audio;
      if (fused >= 0.7) result.engagement_level = "HIGHLY_ENGAGED";
      else if (fused >= 0.5) result.engagement_level = "ENGAGED";
      else if (fused >= 0.3) result.engagement_level = "PARTIALLY_ENGAGED";
      else result.engagement_level = "DISENGAGED";
    }

    // Override suspicion with multimodal
    if (this.mode === "PROCTORING") {
      const highCount = mergedFlags.filter((f) => f.severity === "HIGH").length;
      if (highCount >= 4) result.suspicion_level = "CRITICAL";
      else if (highCount >= 2) result.suspicion_level = "HIGH";
      else if (highCount >= 1) result.suspicion_level = "MODERATE";
      else result.suspicion_level = "LOW";
    }

    return result;
  }

  /**
   * Combined emotional state from vision + audio.
   * @param {Object} report - Vision report
   * @param {Object} audioData - Audio emotion payload
   * @returns {Object} { vision_emotion, audio_emotion, fused_emotion, congruence, insights }
   */
  getEmotionalState(report, audioData) {
    const ae = (audioData && audioData.audio_emotion) ? audioData.audio_emotion : {};
    const mm = report.multimodal_emotion || {};

    // If main.py already computed multimodal fusion, use it
    if (mm && mm.fused_emotion) {
      return {
        vision_emotion: mm.vision_emotion || "neutral",
        audio_emotion: mm.audio_emotion || ae.label || "neutral",
        fused_emotion: mm.fused_emotion,
        fused_confidence: mm.fused_confidence || 0,
        congruence: mm.congruence || 0,
        congruence_level: mm.congruence_level || "UNKNOWN",
        insights: mm.behavioral_insights || [],
      };
    }

    // Fallback: compute here if main.py didn't
    const audioLabel = (ae.label || "neutral").toLowerCase();
    const audioConf = ae.confidence || 0;
    const visionEmotion = this._deriveVisionEmotion(report);

    const vVal = EMOTION_VALENCE[visionEmotion] || 0;
    const aVal = EMOTION_VALENCE[audioLabel] || 0;
    const diff = Math.abs(vVal - aVal);
    const congruence = round(Math.max(0, 1 - diff / 1.8), 2);

    let fusedEmotion, fusedConf;
    if (visionEmotion === audioLabel) {
      fusedEmotion = audioLabel;
      fusedConf = Math.min(1, audioConf * 1.2);
    } else if (congruence > 0.6) {
      fusedEmotion = audioConf > 0.5 ? audioLabel : visionEmotion;
      fusedConf = audioConf;
    } else {
      fusedEmotion = audioLabel; // voice harder to fake
      fusedConf = audioConf * 0.8;
    }

    const insights = [];
    if (congruence >= 0.8) {
      insights.push(`Strong emotional congruence — face and voice both indicate ${fusedEmotion}.`);
    } else if (congruence >= 0.5) {
      insights.push(`Mixed signals — face reads ${visionEmotion} but voice indicates ${audioLabel}. Possible social masking.`);
    } else {
      insights.push(`Emotional mismatch — face shows ${visionEmotion}, voice reveals ${audioLabel}. Client may be suppressing true emotions.`);
    }

    if ((ae.arousal || 0) > 0.3 && (report.engagement_score || 5) <= 4) {
      insights.push("High vocal arousal but low visual engagement — possible internal distress or frustration.");
    }
    if (audioLabel === "happy" && (report.engagement_score || 5) >= 7) {
      insights.push("Positive vocal tone with high engagement — client is genuinely receptive and comfortable.");
    }

    return {
      vision_emotion: visionEmotion,
      audio_emotion: audioLabel,
      fused_emotion: fusedEmotion,
      fused_confidence: round(fusedConf, 2),
      congruence,
      congruence_level: congruence >= 0.7 ? "HIGH" : congruence >= 0.4 ? "MEDIUM" : "LOW",
      insights,
    };
  }

  /**
   * Detect cross-modal emotional inconsistencies.
   * @param {Object} report - Vision report
   * @param {Object} audioData - Audio emotion payload
   * @returns {Array} List of detected inconsistencies
   */
  detectInconsistencies(report, audioData) {
    const ae = (audioData && audioData.audio_emotion) ? audioData.audio_emotion : {};
    if (!ae.label || ae.status === "silence") return [];

    const visionEmotion = this._deriveVisionEmotion(report);
    const audioLabel = ae.label.toLowerCase();
    const detected = [];

    for (const pattern of INCONSISTENCY_PATTERNS) {
      if (pattern.vision === visionEmotion && pattern.audio === audioLabel) {
        detected.push({
          ...pattern,
          audio_confidence: ae.confidence || 0,
          timestamp: Date.now(),
        });
      }
    }

    // Dynamic inconsistencies not in fixed patterns
    const vVal = EMOTION_VALENCE[visionEmotion] || 0;
    const aVal = EMOTION_VALENCE[audioLabel] || 0;
    const valDiff = Math.abs(vVal - aVal);

    if (valDiff > 1.0 && detected.length === 0) {
      detected.push({
        severity: "MEDIUM",
        label: "VALENCE_CONFLICT",
        message: `Large valence gap between face (${visionEmotion}: ${vVal.toFixed(1)}) and voice (${audioLabel}: ${aVal.toFixed(1)}).`,
        vision: visionEmotion,
        audio: audioLabel,
      });
    }

    return detected;
  }

  // ── Private helpers ────────────────────────────────────────────────

  _parsePercent(str) {
    if (!str) return 0;
    const val = parseInt(str, 10);
    return isNaN(val) ? 0 : val / 100;
  }

  /** Derive a single emotion label from vision signal data. */
  _deriveVisionEmotion(report) {
    // Use pre-computed if available
    if (report.multimodal_emotion && report.multimodal_emotion.vision_emotion) {
      return report.multimodal_emotion.vision_emotion;
    }

    // Derive from face signals in the sig dict (if passed through)
    const sig = report._sig || {};
    const smile = sig.smile_genuine;
    const smileLabel = sig.smile_label || "";
    const browLabel = sig.brow_label || "";
    const lipLabel = sig.lip_label || "";
    const tension = sig.micro_tension_score || 0;
    const engagement = sig.engagement_score || report.engagement_score || 5;

    if (smile && smileLabel === "GENUINE") return "happy";
    if (browLabel === "FURROWED" && tension >= 6) return "angry";
    if (browLabel === "RAISED" && ["SLIGHTLY_OPEN", "SPEAKING"].includes(lipLabel)) return "surprised";
    if (tension >= 5 && lipLabel === "COMPRESSED") return "fear";
    if (engagement <= 3 && !smile) return "sad";
    return "neutral";
  }

  /** Generate audio-specific behavioral flags. */
  _getAudioFlags(ae) {
    const flags = [];
    const label = (ae.label || "").toLowerCase();
    const arousal = ae.arousal || 0;
    const valence = ae.valence || 0;
    const conf = ae.confidence || 0;

    // High-confidence negative vocal emotion
    if (conf > 0.6 && ["angry", "fear", "disgust"].includes(label)) {
      flags.push({
        severity: "HIGH",
        message: `Strong negative vocal emotion detected: ${label} (${Math.round(conf * 100)}% confidence).`,
      });
    }

    // Vocal stress indicators
    if (arousal > 0.4 && valence < -0.2) {
      flags.push({
        severity: "MEDIUM",
        message: `Vocal stress detected — high arousal (${arousal.toFixed(2)}) with negative valence (${valence.toFixed(2)}).`,
      });
    }

    // Sudden emotional shift would be tracked over time, but for now:
    if (arousal > 0.5) {
      flags.push({
        severity: "LOW",
        message: `Elevated vocal arousal (${arousal.toFixed(2)}) — heightened emotional intensity.`,
      });
    }

    return flags;
  }

  /** Audio-based risk scoring. */
  _getAudioRiskScore(ae, inconsistencies) {
    let score = 0;
    const label = (ae.label || "").toLowerCase();
    const conf = ae.confidence || 0;

    // Negative emotions
    if (["angry", "fear", "disgust"].includes(label)) score += conf * 40;
    else if (label === "sad") score += conf * 25;

    // Inconsistencies
    for (const inc of inconsistencies) {
      if (inc.severity === "HIGH") score += 25;
      else if (inc.severity === "MEDIUM") score += 15;
      else score += 5;
    }

    return Math.min(100, Math.round(score));
  }
}

// Helper
function round(val, decimals) {
  return Math.round(val * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

module.exports = {
  Interpreter,
  THRESHOLDS,
  FUSION_WEIGHTS,
  INCONSISTENCY_PATTERNS,
  EMOTION_VALENCE,
};
