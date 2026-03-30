/**
 * Trusted Advisor SDK — Gesture Counter
 * =======================================
 * Cumulative session-based gesture counter.
 * Tracks total gesture instances (rising-edge detection)
 * and provides per-minute rates.
 *
 * Events emitted:
 *   "gesture" → { total, timestamp }
 */

const { EventEmitter } = require("./events");

class GestureCounter extends EventEmitter {
  constructor() {
    super();
    this._total = 0;
    this._timestamps = [];   // Timestamps of each gesture event
    this._prevState = false;  // Was hand detected on last update?
    this._sessionStart = Date.now();
  }

  /**
   * Update with current hand detection state.
   * Increments counter only on rising edge (no hands → hands detected).
   * @param {number|boolean} handsDetected - Number of hands or boolean
   */
  update(handsDetected) {
    const currentState = typeof handsDetected === "number"
      ? handsDetected > 0
      : Boolean(handsDetected);

    if (currentState && !this._prevState) {
      // Rising edge — new gesture event
      this._total++;
      const ts = Date.now();
      this._timestamps.push(ts);
      this.emit("gesture", { total: this._total, timestamp: ts });
    }

    this._prevState = currentState;
  }

  /**
   * Get the total cumulative gesture count for this session.
   * @returns {number}
   */
  getTotal() {
    return this._total;
  }

  /**
   * Get gestures per minute over the session.
   * @returns {number}
   */
  getRate() {
    const elapsedMin = (Date.now() - this._sessionStart) / 60000;
    return elapsedMin > 0 ? Math.round(this._total / elapsedMin) : 0;
  }

  /**
   * Get gesture timestamps.
   * @returns {number[]}
   */
  getTimestamps() {
    return [...this._timestamps];
  }

  /**
   * Get recent gesture count within the last N seconds.
   * @param {number} [seconds=60]
   * @returns {number}
   */
  getRecent(seconds = 60) {
    const cutoff = Date.now() - seconds * 1000;
    return this._timestamps.filter((ts) => ts >= cutoff).length;
  }

  /**
   * Reset the counter.
   */
  reset() {
    this._total = 0;
    this._timestamps = [];
    this._prevState = false;
    this._sessionStart = Date.now();
  }
}

module.exports = { GestureCounter };
