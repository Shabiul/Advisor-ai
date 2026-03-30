/**
 * Trusted Advisor SDK — Behavioral Interpreter
 * ===============================================
 * Mode-based interpretation engine that mirrors the backend's
 * /analyze logic. Can be used standalone (offline) or to validate
 * server-side interpretations.
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

  _parsePercent(str) {
    if (!str) return 0;
    return parseInt(String(str).replace("%", ""), 10) / 100;
  }
}

module.exports = { Interpreter, THRESHOLDS };
