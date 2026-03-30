/**
 * Trusted Advisor SDK — Session (v2)
 * =====================================
 * High-level orchestrator that ties together all SDK modules:
 *   - ApiClient: backend communication
 *   - SignalStore: signal state management
 *   - AttentionTracker: rolling attention timeline
 *   - AwayTracker: presence & away interval logging
 *   - GestureCounter: session-based gesture counting
 *   - Interpreter: offline behavioral analysis
 *
 * Usage:
 *   const { createSession } = require("trusted-advisor-sdk");
 *
 *   const session = createSession({
 *     backendUrl: "http://localhost:3000",
 *     mode: "PROCTORING",
 *     pollInterval: 400,
 *   });
 *
 *   session.on("update",       (data) => { ... });
 *   session.on("away_end",     (log)  => { ... });
 *   session.on("focus_change", (chg)  => { ... });
 *   session.on("gesture",      (g)    => { ... });
 *   session.on("alert",        (flags)=> { ... });
 *   session.on("error",        (err)  => { ... });
 *
 *   session.start();
 *   // ... later ...
 *   const summary = session.getSummary();
 *   session.stop();
 */

const { EventEmitter } = require("./events");
const { ApiClient } = require("./api");
const { SignalStore } = require("./signals");
const { AttentionTracker } = require("./attention");
const { AwayTracker } = require("./away");
const { GestureCounter } = require("./gestures");
const { Interpreter } = require("./interpreter");

class Session extends EventEmitter {
  /**
   * @param {Object} options
   * @param {string} [options.backendUrl="http://localhost:3000"] - Backend URL
   * @param {string} [options.mode="PROCTORING"] - Operating mode
   * @param {number} [options.pollInterval=400] - Data fetch interval in ms
   * @param {number} [options.minAwayDuration=3] - Min seconds for away logging
   * @param {Object} [options.apiOptions] - Extra options for ApiClient
   */
  constructor(options = {}) {
    super();

    const {
      backendUrl = "http://localhost:3000",
      mode = "PROCTORING",
      pollInterval = 400,
      minAwayDuration = 3,
      apiOptions = {},
    } = options;

    this.mode = mode.toUpperCase();
    this.pollInterval = pollInterval;

    // Sub-modules
    this.api = new ApiClient(backendUrl, apiOptions);
    this.signals = new SignalStore();
    this.attention = new AttentionTracker();
    this.away = new AwayTracker({ minAwayDuration });
    this.gestures = new GestureCounter();
    this.interpreter = new Interpreter(this.mode);

    // Internal state
    this._running = false;
    this._pollTimer = null;
    this._sessionStart = null;

    // Wire up internal event forwarding
    this._wireEvents();
  }

  /**
   * Start the session: begins polling backend + tracking time.
   * @returns {Session} this
   */
  start() {
    if (this._running) return this;
    this._running = true;
    this._sessionStart = Date.now();

    this.away.start();
    this._pollTimer = setInterval(() => this._poll(), this.pollInterval);
    this._poll(); // immediate first fetch

    this.emit("status", "started");
    return this;
  }

  /**
   * Stop the session: stops polling and all trackers.
   */
  stop() {
    if (!this._running) return;
    this._running = false;

    if (this._pollTimer) {
      clearInterval(this._pollTimer);
      this._pollTimer = null;
    }

    this.away.stop();
    this.emit("status", "stopped");
  }

  /**
   * Get comprehensive session summary.
   * @returns {Object}
   */
  getSummary() {
    const awaySummary = this.away.getSummary();
    const streak = this.attention.getCurrentStreak();
    const signals = this.signals.getSignals();
    const report = this.signals.getReport();

    return {
      mode: this.mode,
      sessionDurationSec: Math.round((Date.now() - (this._sessionStart || Date.now())) / 1000),
      attention: {
        current: this.attention.getCurrentLevel(),
        average: Math.round(this.attention.getAverage() * 100),
        streak: streak,
        timeline: this.attention.getRecentScores(50),
      },
      presence: awaySummary,
      awayLogs: this.away.getAwayLogs(),
      gestures: {
        total: this.gestures.getTotal(),
        perMinute: this.gestures.getRate(),
        recent60s: this.gestures.getRecent(60),
      },
      metrics: this.signals.getMetrics(),
      facial: this.signals.getFacialSignals(),
      body: this.signals.getBodyLanguage(),
      interpretation: report ? this.interpreter.interpret(report) : null,
      riskScore: report ? this.interpreter.getRiskScore(report) : 0,
      latestSignals: signals,
    };
  }

  /**
   * Get the raw latest signals.
   * @returns {Object|null}
   */
  getSignals() {
    return this.signals.getSignals();
  }

  /**
   * Get the latest report.
   * @returns {Object|null}
   */
  getReport() {
    return this.signals.getReport();
  }

  /**
   * Check if backend is alive.
   * @returns {Promise<boolean>}
   */
  async healthCheck() {
    return this.api.healthCheck();
  }

  /**
   * Manually push a report (for non-polling usage).
   * @param {Object} payload - { mode, data, sig }
   */
  ingest(payload) {
    this._processPayload(payload);
  }

  // ── Internal ──────────────────────────────────────────────────────

  async _poll() {
    if (!this._running) return;

    try {
      const json = await this.api.getData();

      if (json.status === "waiting" || !json.report) {
        this.away.update(false); // No data = away
        return;
      }

      this._processPayload(json.report);
    } catch (err) {
      this.away.update(false); // Connection lost = away
      this.emit("error", err);
    }
  }

  _processPayload(payload) {
    const sig = payload.sig || {};
    const report = payload.data || {};

    // 1. Signal Store
    this.signals.ingest(payload);

    // 2. Face detection → Away tracker
    const faceDetected = sig.face_detected === true;
    this.away.update(faceDetected);

    // 3. Attention score → Attention tracker
    if (report.summary && report.summary.attention_score !== undefined) {
      this.attention.push(report.summary.attention_score);
    }

    // 4. Gesture counter
    if (sig.gestures !== undefined) {
      // If the Python side already counts cumulatively, just store;
      // otherwise use rising-edge detection
      this.gestures.update(sig.gestures);
    }

    // 5. Interpretation alerts
    if (report.summary) {
      const interp = this.interpreter.interpret(report);
      if (interp.flags && interp.flags.length > 0) {
        this.emit("alert", interp.flags);
      }
    }
  }

  _wireEvents() {
    // Forward sub-module events up to the session level
    this.signals.on("update", (data) => this.emit("update", data));
    this.signals.on("face_lost", (data) => this.emit("face_lost", data));
    this.signals.on("face_found", (data) => this.emit("face_found", data));
    this.signals.on("emotion", (data) => this.emit("emotion", data));

    this.attention.on("score", (data) => this.emit("attention", data));
    this.attention.on("focus_change", (data) => this.emit("focus_change", data));
    this.attention.on("streak", (data) => this.emit("streak", data));

    this.away.on("away_start", (data) => this.emit("away_start", data));
    this.away.on("away_end", (data) => this.emit("away_end", data));
    this.away.on("tick", (data) => this.emit("tick", data));

    this.gestures.on("gesture", (data) => this.emit("gesture", data));
  }
}

/**
 * Factory function.
 * @param {Object} options
 * @returns {Session}
 */
function createSession(options) {
  return new Session(options);
}

module.exports = { Session, createSession };
