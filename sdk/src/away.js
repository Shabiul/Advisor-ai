/**
 * Trusted Advisor SDK — Away Tracker
 * ====================================
 * Tracks user presence/absence via the `face_detected` signal.
 * Maintains a timestamped log of away intervals with durations.
 *
 * Events emitted:
 *   "away_start"  → { startTime: Date }
 *   "away_end"    → { startTime: Date, endTime: Date, durationSec: number }
 *   "tick"        → { sessionSec, awaySec, focusSec, isAway }
 */

const { EventEmitter } = require("./events");

class AwayTracker extends EventEmitter {
  /**
   * @param {Object} [options]
   * @param {number} [options.minAwayDuration=3] - Minimum seconds to log an away interval
   * @param {number} [options.maxLogs=100] - Maximum away log entries to retain
   */
  constructor(options = {}) {
    super();
    this.minAwayDuration = options.minAwayDuration !== undefined ? options.minAwayDuration : 3;
    this.maxLogs = options.maxLogs || 100;

    this._sessionStart = Date.now();
    this._sessionSeconds = 0;
    this._awaySeconds = 0;
    this._isAway = false;
    this._currentAwayStart = null;
    this._awayLogs = [];
    this._tickTimer = null;
  }

  /**
   * Start the internal 1-second tick loop.
   * @returns {AwayTracker} this
   */
  start() {
    if (this._tickTimer) return this;
    this._sessionStart = Date.now();
    this._tickTimer = setInterval(() => this._tick(), 1000);
    return this;
  }

  /**
   * Stop the tick loop.
   */
  stop() {
    if (this._tickTimer) {
      clearInterval(this._tickTimer);
      this._tickTimer = null;
    }
    // Close any open away interval
    if (this._currentAwayStart) {
      this._closeAwayInterval();
    }
  }

  /**
   * Update presence state from signal data.
   * Called externally whenever a new signal payload arrives.
   * @param {boolean} faceDetected
   */
  update(faceDetected) {
    const wasAway = this._isAway;
    this._isAway = !faceDetected;

    if (this._isAway && !wasAway) {
      // Transition: present → away
      this._currentAwayStart = new Date();
      this.emit("away_start", { startTime: this._currentAwayStart });
    } else if (!this._isAway && wasAway) {
      // Transition: away → present
      this._closeAwayInterval();
    }
  }

  /**
   * Get all logged away intervals.
   * @returns {Array<{start: Date, end: Date, durationSec: number}>}
   */
  getAwayLogs() {
    return [...this._awayLogs];
  }

  /**
   * Get timing summary.
   * @returns {{ sessionSec: number, awaySec: number, focusSec: number, awayPercent: number }}
   */
  getSummary() {
    const focusSec = this._sessionSeconds - this._awaySeconds;
    return {
      sessionSec: this._sessionSeconds,
      awaySec: this._awaySeconds,
      focusSec,
      awayPercent: this._sessionSeconds > 0
        ? Math.round((this._awaySeconds / this._sessionSeconds) * 100)
        : 0,
    };
  }

  /**
   * Whether the user is currently away.
   * @returns {boolean}
   */
  isCurrentlyAway() {
    return this._isAway;
  }

  /**
   * Format seconds as HH:MM:SS.
   * @param {number} s
   * @returns {string}
   */
  static formatTime(s) {
    const hh = Math.floor(s / 3600).toString().padStart(2, "0");
    const mm = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
    const ss = (s % 60).toString().padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  /**
   * Reset all tracked data.
   */
  reset() {
    this._sessionStart = Date.now();
    this._sessionSeconds = 0;
    this._awaySeconds = 0;
    this._isAway = false;
    this._currentAwayStart = null;
    this._awayLogs = [];
  }

  // ── Internal ──────────────────────────────────────────────────────

  _tick() {
    this._sessionSeconds++;
    if (this._isAway) {
      this._awaySeconds++;
    }

    this.emit("tick", {
      sessionSec: this._sessionSeconds,
      awaySec: this._awaySeconds,
      focusSec: this._sessionSeconds - this._awaySeconds,
      isAway: this._isAway,
    });
  }

  _closeAwayInterval() {
    if (!this._currentAwayStart) return;

    const endTime = new Date();
    const durationSec = Math.round((endTime - this._currentAwayStart) / 1000);

    if (durationSec >= this.minAwayDuration) {
      const log = {
        start: this._currentAwayStart,
        end: endTime,
        durationSec,
      };

      this._awayLogs.unshift(log);

      // Cap log size
      while (this._awayLogs.length > this.maxLogs) {
        this._awayLogs.pop();
      }

      this.emit("away_end", log);
    }

    this._currentAwayStart = null;
  }
}

module.exports = { AwayTracker };
