/**
 * Trusted Advisor SDK — Session
 * ==============================
 * High-level session manager that orchestrates data capture, polling,
 * and event emission for consumers of the SDK.
 *
 * Usage:
 *   const session = createSession({ backendUrl, mode, pollInterval });
 *   session.on("update", (report) => { ... });
 *   session.on("alert",  (alerts) => { ... });
 *   session.start();
 *   session.pushReport(reportData);
 *   // later …
 *   session.stop();
 *   const finalReport = session.getReport();
 */

const { ApiClient } = require("./api");
const { EventEmitter } = require("./events");

class Session extends EventEmitter {
  constructor({ backendUrl = "http://localhost:3000", mode = "PROCTORING", pollInterval = 5000 } = {}) {
    super();
    this.api = new ApiClient(backendUrl);
    this.mode = mode.toUpperCase();
    this.pollInterval = pollInterval;

    this._running = false;
    this._timer = null;
    this._latestReport = null;
    this._reportQueue = [];
  }

  start() {
    if (this._running) return;
    this._running = true;
    this.emit("status", "started");
    this._poll();
    this._timer = setInterval(() => this._poll(), this.pollInterval);
  }

  stop() {
    this._running = false;
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    this.emit("status", "stopped");
  }

  pushReport(report) {
    this._reportQueue.push(report);
  }

  getReport() {
    return this._latestReport;
  }

  async _poll() {
    if (!this._running) return;

    const report = this._reportQueue.shift();
    if (!report) return;

    try {
      const result = await this.api.analyze(this.mode, report);
      this._latestReport = result;

      this.emit("update", result);

      const flags = result?.interpretation?.flags || [];
      if (flags.length > 0) {
        this.emit("alert", flags);
      }
    } catch (err) {
      this.emit("error", err);
    }
  }
}

function createSession(options) {
  return new Session(options);
}

module.exports = { Session, createSession };
