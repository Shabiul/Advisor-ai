/**
 * Trusted Advisor SDK — Attention Tracker
 * =========================================
 * Maintains a rolling time-series of attention scores and computes
 * derived analytics: timeline history, focus streaks, averages.
 *
 * Events emitted:
 *   "score"        → { score, focusLevel, timestamp }
 *   "focus_change" → { from, to, timestamp }
 *   "streak"       → { type, durationSec }
 */

const { EventEmitter } = require("./events");

class AttentionTracker extends EventEmitter {
  /**
   * @param {Object} [options]
   * @param {number} [options.maxHistory=300] - Max data points to retain
   * @param {Object} [options.thresholds] - Custom focus level thresholds
   */
  constructor(options = {}) {
    super();
    this.maxHistory = options.maxHistory || 300;
    this.thresholds = Object.assign(
      { high: 0.70, medium: 0.40 },
      options.thresholds || {}
    );

    this._history = [];           // [{ score, focusLevel, timestamp }]
    this._lastFocusLevel = null;
    this._streakStart = Date.now();
    this._streakType = null;
  }

  /**
   * Push a new attention score from the pipeline.
   * @param {number} score - Attention score between 0.0 and 1.0
   */
  push(score) {
    const clamped = Math.max(0, Math.min(1, score));
    const focusLevel = this._classify(clamped);
    const timestamp = Date.now();

    const entry = { score: clamped, focusLevel, timestamp };
    this._history.push(entry);

    // Evict old entries
    while (this._history.length > this.maxHistory) {
      this._history.shift();
    }

    this.emit("score", entry);

    // Detect focus level transitions
    if (this._lastFocusLevel && focusLevel !== this._lastFocusLevel) {
      this.emit("focus_change", {
        from: this._lastFocusLevel,
        to: focusLevel,
        timestamp,
      });

      // Emit streak for the completed segment
      const streakDuration = Math.round((timestamp - this._streakStart) / 1000);
      if (streakDuration >= 3) {
        this.emit("streak", {
          type: this._lastFocusLevel,
          durationSec: streakDuration,
        });
      }
      this._streakStart = timestamp;
    }

    this._lastFocusLevel = focusLevel;
    this._streakType = focusLevel;
  }

  /**
   * Get the full attention timeline.
   * @returns {Array<{score: number, focusLevel: string, timestamp: number}>}
   */
  getTimeline() {
    return [...this._history];
  }

  /**
   * Get the most recent N scores as an array of numbers.
   * @param {number} [count=50]
   * @returns {number[]}
   */
  getRecentScores(count = 50) {
    return this._history.slice(-count).map((e) => e.score);
  }

  /**
   * Calculate mean attention over the full history or last N entries.
   * @param {number} [lastN] - If provided, only average the last N entries
   * @returns {number}
   */
  getAverage(lastN) {
    const slice = lastN ? this._history.slice(-lastN) : this._history;
    if (slice.length === 0) return 0;
    return slice.reduce((sum, e) => sum + e.score, 0) / slice.length;
  }

  /**
   * Get the current focus level.
   * @returns {string|null}
   */
  getCurrentLevel() {
    return this._lastFocusLevel;
  }

  /**
   * Get current streak info.
   * @returns {{ type: string, durationSec: number }}
   */
  getCurrentStreak() {
    return {
      type: this._streakType || "NONE",
      durationSec: Math.round((Date.now() - this._streakStart) / 1000),
    };
  }

  /**
   * Reset all tracked data.
   */
  reset() {
    this._history = [];
    this._lastFocusLevel = null;
    this._streakStart = Date.now();
    this._streakType = null;
  }

  _classify(score) {
    if (score >= this.thresholds.high) return "HIGH";
    if (score >= this.thresholds.medium) return "MEDIUM";
    return "LOW";
  }
}

module.exports = { AttentionTracker };
