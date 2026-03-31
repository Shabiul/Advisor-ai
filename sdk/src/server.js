/**
 * Trusted Advisor SDK — Embedded Server
 * ========================================
 * The complete backend server as a module. Includes:
 *   - POST /analyze    — receive + interpret behavioral reports
 *   - GET  /data       — return latest report for dashboard
 *   - POST /video_frame — receive JPEG frames from Python
 *   - GET  /video_feed  — MJPEG stream to browser
 *   - Static dashboard  — full Twitch-style UI served from memory
 *
 * Usage:
 *   const { createServer } = require("trusted-advisor-sdk");
 *   const server = createServer({ port: 3000 });
 *   server.start();
 */

const http = require("http");
const { URL } = require("url");
const fs = require("fs");
const path = require("path");
const { Interpreter, THRESHOLDS } = require("./interpreter");
const { EventEmitter } = require("./events");
const { getDashboardHTML, getDashboardCSS, getDashboardJS } = require("./dashboard");

class AdvisorServer extends EventEmitter {
  /**
   * @param {Object} [options]
   * @param {number} [options.port=3000] - Port to listen on
   * @param {string} [options.host="0.0.0.0"] - Host to bind to
   * @param {boolean} [options.serveDashboard=true] - Whether to serve the web dashboard
   */
  constructor(options = {}) {
    super();
    this.port = options.port || 3000;
    this.host = options.host || "0.0.0.0";
    this.serveDashboard = options.serveDashboard !== false;

    this._latestReport = null;
    this._lastUpdated = null;
    this._videoClients = [];
    this._httpServer = null;
  }

  /**
   * Start the HTTP server.
   * @returns {Promise<AdvisorServer>}
   */
  start() {
    return new Promise((resolve, reject) => {
      this._httpServer = http.createServer((req, res) => this._handleRequest(req, res));

      this._httpServer.listen(this.port, this.host, () => {
        console.log(`[Trusted Advisor SDK] Server running on http://localhost:${this.port}`);
        console.log(`[Trusted Advisor SDK] Dashboard: http://localhost:${this.port}/`);
        console.log(`[Trusted Advisor SDK] Waiting for Python pipeline data...`);
        this.emit("started", { port: this.port });
        resolve(this);
      });

      this._httpServer.on("error", (err) => {
        this.emit("error", err);
        reject(err);
      });
    });
  }

  /**
   * Stop the HTTP server.
   * @returns {Promise<void>}
   */
  stop() {
    return new Promise((resolve) => {
      if (this._httpServer) {
        // Close all video stream clients
        this._videoClients.forEach((c) => { try { c.end(); } catch (e) {} });
        this._videoClients = [];

        this._httpServer.close(() => {
          this.emit("stopped");
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  /**
   * Get latest stored report.
   * @returns {Object|null}
   */
  getLatestReport() {
    return this._latestReport;
  }

  /**
   * Get latest signal data.
   * @returns {Object|null}
   */
  getLatestSignals() {
    return this._latestReport ? this._latestReport.sig : null;
  }

  // ── Request Router ──────────────────────────────────────────────────

  _handleRequest(req, res) {
    const parsedUrl = new URL(req.url, `http://${req.headers.host}`);
    const pathname = parsedUrl.pathname;

    // CORS
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    // Route
    if (req.method === "POST" && pathname === "/analyze") {
      this._handleAnalyze(req, res);
    } else if (req.method === "POST" && pathname === "/video_frame") {
      this._handleVideoFrame(req, res);
    } else if (req.method === "GET" && pathname === "/video_feed") {
      this._handleVideoFeed(req, res);
    } else if (req.method === "GET" && pathname === "/data") {
      this._handleData(req, res);
    } else if (req.method === "GET" && pathname === "/recordings") {
      this._handleRecordings(req, res);
    } else if (this.serveDashboard && req.method === "GET") {
      this._handleStatic(pathname, res);
    } else {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Not found" }));
    }
  }

  // ── POST /analyze ───────────────────────────────────────────────────

  _handleAnalyze(req, res) {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      try {
        const parsed = JSON.parse(body);
        const { mode, data } = parsed;

        if (!mode || !data) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'mode' or 'data' in request body." }));
          return;
        }

        const upperMode = mode.toUpperCase();
        const thresholds = THRESHOLDS[upperMode];

        if (!thresholds) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: `Unknown mode '${mode}'. Supported: PROCTORING, MEETING.` }));
          return;
        }

        // Store
        this._latestReport = parsed;
        this._lastUpdated = Date.now();

        // Interpret
        const interpreter = new Interpreter(upperMode);
        const interpretation = interpreter.interpret(data);

        // Emit event
        this.emit("analyze", { mode: upperMode, data, sig: parsed.sig, interpretation });

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ mode: upperMode, interpretation }));
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Invalid JSON: " + err.message }));
      }
    });
  }

  // ── POST /video_frame ───────────────────────────────────────────────

  _handleVideoFrame(req, res) {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      const frame = Buffer.concat(chunks);
      if (!frame.length) {
        res.writeHead(400);
        res.end("Empty frame");
        return;
      }

      // Broadcast to all SSE video clients
      const header = `--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: ${frame.length}\r\n\r\n`;
      this._videoClients = this._videoClients.filter((client) => {
        try {
          client.write(header);
          client.write(frame);
          client.write("\r\n");
          return true;
        } catch (e) {
          return false;
        }
      });

      res.writeHead(200);
      res.end("OK");
    });
  }

  // ── GET /video_feed ─────────────────────────────────────────────────

  _handleVideoFeed(req, res) {
    res.writeHead(200, {
      "Content-Type": "multipart/x-mixed-replace; boundary=FRAME",
      "Cache-Control": "no-cache, private",
      "Connection": "keep-alive",
      "Pragma": "no-cache",
    });
    this._videoClients.push(res);

    req.on("close", () => {
      this._videoClients = this._videoClients.filter((c) => c !== res);
    });
  }

  // ── GET /data ───────────────────────────────────────────────────────

  _handleData(req, res) {
    res.writeHead(200, { "Content-Type": "application/json" });

    if (!this._latestReport) {
      res.end(JSON.stringify({ status: "waiting", message: "No data received yet." }));
      return;
    }

    res.end(JSON.stringify({
      status: "live",
      lastUpdated: this._lastUpdated,
      report: this._latestReport,
    }));
  }

  // ── GET /recordings ─────────────────────────────────────────────────

  _handleRecordings(req, res) {
    // Search for recordings directory in likely locations
    const candidates = [
      path.resolve(__dirname, "..", "..", "recordings"),        // sdk/../recordings
      path.resolve(__dirname, "..", "..", "..", "recordings"),  // project root
      path.resolve(process.cwd(), "recordings"),               // cwd
    ];

    let recordingsDir = null;
    for (const dir of candidates) {
      if (fs.existsSync(dir)) {
        recordingsDir = dir;
        break;
      }
    }

    if (!recordingsDir) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ sessions: [], message: "No recordings directory found." }));
      return;
    }

    try {
      const entries = fs.readdirSync(recordingsDir, { withFileTypes: true });
      const sessions = [];

      for (const entry of entries) {
        if (!entry.isDirectory()) continue;

        const sessionPath = path.join(recordingsDir, entry.name);
        const metaPath = path.join(sessionPath, "metadata.json");

        let metadata = {};
        if (fs.existsSync(metaPath)) {
          try {
            metadata = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
          } catch (e) {}
        }

        const hasVideo = fs.existsSync(path.join(sessionPath, "video.mp4")) ||
                         fs.existsSync(path.join(sessionPath, "video.avi"));
        const hasSignals = fs.existsSync(path.join(sessionPath, "signals.jsonl"));

        sessions.push({
          name: entry.name,
          path: sessionPath,
          has_video: hasVideo,
          has_signals: hasSignals,
          duration: metadata.duration_seconds || null,
          total_frames: metadata.total_frames || null,
          start_time: metadata.start_time || null,
          end_time: metadata.end_time || null,
        });
      }

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ sessions, count: sessions.length }));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Failed to list recordings: " + err.message }));
    }
  }

  // ── Static Dashboard ───────────────────────────────────────────────

  _handleStatic(pathname, res) {
    if (pathname === "/" || pathname === "/index.html") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(getDashboardHTML());
    } else if (pathname === "/style.css") {
      res.writeHead(200, { "Content-Type": "text/css; charset=utf-8" });
      res.end(getDashboardCSS());
    } else if (pathname === "/app.js" || pathname.startsWith("/app.js")) {
      res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
      res.end(getDashboardJS());
    } else {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not found");
    }
  }
}

/**
 * Factory function.
 * @param {Object} [options]
 * @returns {AdvisorServer}
 */
function createServer(options) {
  return new AdvisorServer(options);
}

module.exports = { AdvisorServer, createServer };
