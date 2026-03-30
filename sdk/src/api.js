/**
 * Trusted Advisor SDK — API Client
 * ==================================
 * HTTP communication layer for the Trusted Advisor AI backend.
 * Supports: POST /analyze, GET /data, POST /video_frame
 *
 * Uses native Node.js `http` module — zero dependencies.
 */

const http = require("http");
const https = require("https");
const { URL } = require("url");

class ApiClient {
  /**
   * @param {string} baseUrl - Backend URL (e.g. "http://localhost:3000")
   * @param {Object} [options]
   * @param {number} [options.timeout=5000] - Request timeout in ms
   * @param {Object} [options.headers] - Additional headers to include
   */
  constructor(baseUrl = "http://localhost:3000", options = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeout = options.timeout || 5000;
    this.extraHeaders = options.headers || {};
  }

  /**
   * POST behavioral report to /analyze
   * @param {string} mode - "PROCTORING" or "MEETING"
   * @param {Object} reportData - The cumulative report from the Python pipeline
   * @param {Object} [sig] - Raw signal dictionary
   * @returns {Promise<Object>} Server interpretation response
   */
  async analyze(mode, reportData, sig = {}) {
    const payload = { mode, data: reportData, sig };
    return this._post("/analyze", payload);
  }

  /**
   * GET latest data from /data
   * @returns {Promise<Object>} { status, lastUpdated, report }
   */
  async getData() {
    return this._get("/data");
  }

  /**
   * POST a raw JPEG frame to /video_frame
   * @param {Buffer} jpegBuffer - Raw JPEG image bytes
   * @returns {Promise<number>} HTTP status code
   */
  async postVideoFrame(jpegBuffer) {
    return new Promise((resolve, reject) => {
      const url = new URL(this.baseUrl + "/video_frame");
      const transport = url.protocol === "https:" ? https : http;

      const req = transport.request(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          method: "POST",
          headers: {
            "Content-Type": "image/jpeg",
            "Content-Length": jpegBuffer.length,
            ...this.extraHeaders,
          },
          timeout: this.timeout,
        },
        (res) => {
          let body = "";
          res.on("data", (chunk) => (body += chunk));
          res.on("end", () => resolve(res.statusCode));
        }
      );

      req.on("error", reject);
      req.on("timeout", () => {
        req.destroy();
        reject(new Error("Video frame POST timed out"));
      });

      req.write(jpegBuffer);
      req.end();
    });
  }

  /**
   * Health check — pings GET /data to verify backend is alive
   * @returns {Promise<boolean>} true if reachable
   */
  async healthCheck() {
    try {
      const data = await this._get("/data");
      return data !== null;
    } catch {
      return false;
    }
  }

  // ── Internal HTTP helpers ──────────────────────────────────────────

  _post(path, body) {
    return new Promise((resolve, reject) => {
      const url = new URL(this.baseUrl + path);
      const transport = url.protocol === "https:" ? https : http;
      const payload = JSON.stringify(body);

      const req = transport.request(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(payload),
            ...this.extraHeaders,
          },
          timeout: this.timeout,
        },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => {
            if (res.statusCode >= 400) {
              reject(new Error(`[ApiClient] ${res.statusCode}: ${data}`));
            } else {
              try {
                resolve(JSON.parse(data));
              } catch {
                resolve(data);
              }
            }
          });
        }
      );

      req.on("error", reject);
      req.on("timeout", () => {
        req.destroy();
        reject(new Error(`POST ${path} timed out`));
      });

      req.write(payload);
      req.end();
    });
  }

  _get(path) {
    return new Promise((resolve, reject) => {
      const url = new URL(this.baseUrl + path);
      const transport = url.protocol === "https:" ? https : http;

      const req = transport.request(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          method: "GET",
          headers: this.extraHeaders,
          timeout: this.timeout,
        },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => {
            try {
              resolve(JSON.parse(data));
            } catch {
              resolve(data);
            }
          });
        }
      );

      req.on("error", reject);
      req.on("timeout", () => {
        req.destroy();
        reject(new Error(`GET ${path} timed out`));
      });

      req.end();
    });
  }
}

module.exports = { ApiClient };
