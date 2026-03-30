/**
 * Trusted Advisor SDK — Signal Store
 * =====================================
 * Centralized store for the latest raw signal payload from the
 * Python pipeline. Provides reactive access, diffing, and history.
 *
 * Events emitted:
 *   "update"       → { sig, report, timestamp }
 *   "face_lost"    → { timestamp }
 *   "face_found"   → { timestamp }
 *   "emotion"      → { emotion, confidence }
 */

const { EventEmitter } = require("./events");

class SignalStore extends EventEmitter {
  /**
   * @param {Object} [options]
   * @param {number} [options.maxHistory=60] - Max signal snapshots to retain
   */
  constructor(options = {}) {
    super();
    this.maxHistory = options.maxHistory || 60;

    this._latest = null;        // { sig, report, timestamp }
    this._history = [];
    this._prevFaceDetected = null;
  }

  /**
   * Ingest a new payload from the backend.
   * @param {Object} payload - The `report` object from GET /data → { mode, data, sig }
   */
  ingest(payload) {
    const sig = payload.sig || {};
    const report = payload.data || {};
    const timestamp = Date.now();

    const entry = { sig, report, timestamp };
    this._latest = entry;
    this._history.push(entry);

    while (this._history.length > this.maxHistory) {
      this._history.shift();
    }

    this.emit("update", entry);

    // Face detection transitions
    const faceNow = sig.face_detected === true;
    if (this._prevFaceDetected !== null) {
      if (!faceNow && this._prevFaceDetected) {
        this.emit("face_lost", { timestamp });
      }
      if (faceNow && !this._prevFaceDetected) {
        this.emit("face_found", { timestamp });
      }
    }
    this._prevFaceDetected = faceNow;

    // Emotion events
    if (sig.emotion && sig.emotion !== "loading" && sig.emotion !== "analyzing") {
      this.emit("emotion", {
        emotion: sig.emotion,
        confidence: sig.emotion_conf || 0,
      });
    }
  }

  /**
   * Get the latest signal dictionary.
   * @returns {Object|null}
   */
  getSignals() {
    return this._latest ? this._latest.sig : null;
  }

  /**
   * Get the latest cumulative report.
   * @returns {Object|null}
   */
  getReport() {
    return this._latest ? this._latest.report : null;
  }

  /**
   * Get a specific signal value.
   * @param {string} key - Signal key (e.g. "gaze", "head_pose")
   * @param {*} [defaultValue] - Fallback if missing
   * @returns {*}
   */
  get(key, defaultValue = null) {
    if (!this._latest) return defaultValue;
    return this._latest.sig[key] !== undefined ? this._latest.sig[key] : defaultValue;
  }

  /**
   * Get recent signal history.
   * @param {number} [count]
   * @returns {Array}
   */
  getHistory(count) {
    return count ? this._history.slice(-count) : [...this._history];
  }

  /**
   * Check if face is currently detected.
   * @returns {boolean}
   */
  isFaceDetected() {
    return this._prevFaceDetected === true;
  }

  /**
   * Get structured body language data.
   * @returns {Object}
   */
  getBodyLanguage() {
    const sig = this.getSignals();
    if (!sig) return {};
    return sig.body_language || {};
  }

  /**
   * Get facial analysis data.
   * @returns {Object}
   */
  getFacialSignals() {
    const sig = this.getSignals();
    if (!sig) return {};
    return {
      smile: { genuine: sig.smile_genuine, label: sig.smile_label },
      brow: { score: sig.brow_score, label: sig.brow_label },
      lip: { score: sig.lip_score, label: sig.lip_label },
      face: sig.face || {},
    };
  }

  /**
   * Get engagement metrics.
   * @returns {{ engagement: number, tension: number, eyeContact: number, blinksPerMinute: number }}
   */
  getMetrics() {
    const sig = this.getSignals();
    if (!sig) return { engagement: 0, tension: 0, eyeContact: 0, blinksPerMinute: 0 };
    return {
      engagement: sig.engagement_score || 0,
      tension: sig.micro_tension_score || 0,
      eyeContact: sig.eye_contact_score || 0,
      blinksPerMinute: sig.blinks_per_minute || 0,
    };
  }

  /**
   * Reset the store.
   */
  reset() {
    this._latest = null;
    this._history = [];
    this._prevFaceDetected = null;
  }
}

module.exports = { SignalStore };
